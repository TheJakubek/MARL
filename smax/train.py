"""Training loop for SMAX 2s3z with VDN/QMIX + indep/correlated exploration.

Single-env training (n_envs=1). Could be vmap'd, but we want to focus on
exploration ablation, not throughput, and a single env is fine on GPU at
~5k transitions/sec.

For each transition:
  1. Forward QNet on per-agent obs (with agent_id + role one-hot).
  2. Pick actions via exploration (independent or correlated copula).
  3. Step env, store transition.
  4. Every train_every steps and once buffer is warm: sample, compute TD, gradient step.
  5. Every target_sync steps: copy online -> target.

Logs (returned in result dict):
  episode_returns: list of float, one per finished episode
  episode_wins:    list of bool/0-1
  steps_done:      total env steps
  corr_matrix_log: list of (step, R) snapshots, only for correlated runs
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario

from smax.buffer import ReplayBuffer
from smax.exploration import correlation_matrix, select_correlated, select_independent
from smax.mixers import QMixMixer, VDNMixer
from smax.qnet import QNet, augment_obs


# Unit-type indices in jaxmarl's smax_env.SMAX (see unit_type_names list).
# 0:marine 1:marauder 2:stalker 3:zealot 4:zergling 5:hydralisk
ROLE_TO_IDX = {"stalker": 2, "zealot": 3}
N_ROLES = 2  # we only have stalker/zealot in 2s3z


def _build_role_oh(scenario_unit_types: jnp.ndarray) -> jnp.ndarray:
    """Map global unit_type idx (0..5) to local role idx (0=stalker, 1=zealot)."""
    # In 2s3z, ally unit types are first 5: [stalker, stalker, zealot, zealot, zealot].
    role_local = jnp.where(
        scenario_unit_types[:5] == ROLE_TO_IDX["stalker"], 0, 1
    )
    return jax.nn.one_hot(role_local, N_ROLES)


@dataclass
class Config:
    seed: int = 0
    total_steps: int = 200_000
    buffer_cap: int = 50_000
    batch_size: int = 64
    warmup: int = 2_000
    train_every: int = 8
    target_sync: int = 500
    gamma: float = 0.99
    lr: float = 3e-4
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal_steps: int = 50_000
    hidden_size: int = 128
    exploration: str = "independent"          # or "correlated"
    similarity: str = "obs"                    # or "q_values" or "hidden"
    mixer: str = "vdn"                         # or "qmix"
    qmix_embed: int = 32
    log_corr_every: int = 5_000


def epsilon(step: int, cfg: Config) -> float:
    frac = min(1.0, step / cfg.eps_anneal_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _stack_obs(obs_dict, agents):
    return jnp.stack([obs_dict[a] for a in agents], axis=0)


def _stack_avail(avail_dict, agents):
    return jnp.stack([avail_dict[a] for a in agents], axis=0)


def select_features(
    obs: jnp.ndarray,        # (n_agents, obs_dim_aug)
    q: jnp.ndarray,          # (n_agents, n_actions)
    hidden: jnp.ndarray,     # (n_agents, hidden_dim)
    similarity: str,
) -> jnp.ndarray:
    if similarity == "obs":
        return obs
    elif similarity == "q_values":
        return q
    elif similarity == "hidden":
        return hidden
    else:
        raise ValueError(f"unknown similarity: {similarity}")


class TrainState(train_state.TrainState):
    target_qnet_params: dict
    mixer_params: dict
    target_mixer_params: dict


def train(cfg: Config):
    key = jax.random.PRNGKey(cfg.seed)

    # --- Env ---
    scenario = map_name_to_scenario("2s3z")
    env = make(
        "HeuristicEnemySMAX",
        scenario=scenario,
        use_self_play_reward=False,
        walls_cause_death=True,
        see_enemy_actions=False,
    )
    agents = env.agents
    n_agents = len(agents)
    n_actions = env.action_space(agents[0]).n

    # Probe shapes from one reset.
    key, k_reset = jax.random.split(key)
    obs0, state0 = env.reset(k_reset)
    obs_dim = obs0[agents[0]].shape[0]
    state_dim = obs0["world_state"].shape[0]

    # Roles (stalker/zealot) — fixed for 2s3z scenario.
    # scenario.unit_types is a jnp array of length num_allies + num_enemies.
    role_oh = _build_role_oh(jnp.asarray(scenario.unit_types))
    agent_id_oh = jnp.eye(n_agents)
    aug_dim = obs_dim + n_agents + N_ROLES

    print(f"[cfg] n_agents={n_agents}  n_actions={n_actions}  "
          f"obs_dim={obs_dim}  state_dim={state_dim}  aug_dim={aug_dim}")

    # --- Networks ---
    qnet = QNet(n_actions=n_actions, hidden_sizes=(cfg.hidden_size, cfg.hidden_size))
    if cfg.mixer == "vdn":
        mixer = VDNMixer()
        mixer_state_dim = state_dim
    elif cfg.mixer == "qmix":
        mixer = QMixMixer(n_agents=n_agents, embed_dim=cfg.qmix_embed)
        mixer_state_dim = state_dim
    else:
        raise ValueError(cfg.mixer)

    # Init params.
    key, k_q, k_m = jax.random.split(key, 3)
    dummy_aug = jnp.zeros((aug_dim,))
    qnet_params = qnet.init(k_q, dummy_aug)
    target_qnet_params = qnet_params

    dummy_q_chosen = jnp.zeros((n_agents,))
    dummy_state = jnp.zeros((mixer_state_dim,))
    mixer_params = mixer.init(k_m, dummy_q_chosen, dummy_state)
    target_mixer_params = mixer_params

    # --- Optimizer ---
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.lr),
    )
    # Combine qnet + mixer params for a single optimizer.
    opt_params = {"qnet": qnet_params, "mixer": mixer_params}
    opt_state = tx.init(opt_params)

    # --- Replay ---
    buf = ReplayBuffer(
        cap=cfg.buffer_cap,
        n_agents=n_agents,
        obs_dim=aug_dim,         # we store augmented obs
        state_dim=state_dim,
        n_actions=n_actions,
    )

    # --- JIT-compiled forward ---
    @jax.jit
    def forward_qnet(params, aug_obs_batch):
        # aug_obs_batch: (n_agents, aug_dim)
        # returns (n_agents, n_actions), (n_agents, hidden)
        return jax.vmap(lambda x: qnet.apply(params, x, return_hidden=True))(
            aug_obs_batch
        )

    @jax.jit
    def loss_fn_jit(params, batch):
        # Batched TD loss.
        # batch fields are np arrays converted to jnp.
        obs_b = batch["obs"]                  # (B, n_agents, aug_dim)
        state_b = batch["state"]              # (B, state_dim)
        actions_b = batch["actions"]          # (B, n_agents)
        rewards_b = batch["rewards"]          # (B,)
        next_obs_b = batch["next_obs"]        # (B, n_agents, aug_dim)
        next_state_b = batch["next_state"]
        next_avail_b = batch["next_avail"]    # (B, n_agents, n_actions)
        done_b = batch["done"]                # (B,)

        def online_q(params_q, obs):
            return jax.vmap(qnet.apply, in_axes=(None, 0))(params_q, obs)

        # Q(s,a) for each agent.
        q_all = jax.vmap(online_q, in_axes=(None, 0))(params["qnet"], obs_b)
        # q_all: (B, n_agents, n_actions). Gather chosen.
        q_chosen = jnp.take_along_axis(
            q_all, actions_b[..., None], axis=-1
        ).squeeze(-1)                          # (B, n_agents)

        # Target Q: max over avail with target net.
        q_next_all = jax.vmap(online_q, in_axes=(None, 0))(
            params["target_qnet"], next_obs_b
        )
        masked = jnp.where(
            next_avail_b.astype(jnp.bool_), q_next_all, -1e9
        )
        q_next_max = jnp.max(masked, axis=-1)   # (B, n_agents)

        # Mix.
        q_tot = mixer.apply(params["mixer"], q_chosen, state_b).squeeze(-1)
        q_tot_next = mixer.apply(
            params["target_mixer"], q_next_max, next_state_b
        ).squeeze(-1)

        # Bellman target.
        target = rewards_b + cfg.gamma * (1.0 - done_b) * q_tot_next
        target = jax.lax.stop_gradient(target)

        return jnp.mean((q_tot - target) ** 2)

    @jax.jit
    def update(opt_params, opt_state, target_qnet_params, target_mixer_params, batch):
        all_params = {
            **opt_params,
            "target_qnet": target_qnet_params,
            "target_mixer": target_mixer_params,
        }
        loss, grads = jax.value_and_grad(loss_fn_jit)(all_params, batch)
        # Only update non-target params.
        upd_grads = {"qnet": grads["qnet"], "mixer": grads["mixer"]}
        updates, opt_state = tx.update(upd_grads, opt_state, opt_params)
        opt_params = optax.apply_updates(opt_params, updates)
        return opt_params, opt_state, loss

    # --- Run ---
    rng_np = np.random.default_rng(cfg.seed)
    key, k = jax.random.split(key)
    obs_dict, state = env.reset(k)

    ep_return = 0.0
    ep_returns, ep_wins = [], []
    corr_log = []
    losses = []

    obs_arr = _stack_obs(obs_dict, agents)                     # (n_agents, obs_dim)
    aug_obs = augment_obs(obs_arr, agent_id_oh, role_oh)       # (n_agents, aug_dim)

    for step in range(cfg.total_steps):
        eps = epsilon(step, cfg)

        # Forward.
        q, hidden = forward_qnet(opt_params["qnet"], aug_obs)
        avail = _stack_avail(env.get_avail_actions(state), agents)

        # Pick actions.
        feats = select_features(aug_obs, q, hidden, cfg.similarity)
        key, k_act = jax.random.split(key)
        if cfg.exploration == "independent":
            actions = select_independent(k_act, q, avail, eps)
        else:
            actions = select_correlated(k_act, q, avail, feats, eps)

        # Optionally log correlation matrix snapshot.
        if cfg.exploration == "correlated" and step % cfg.log_corr_every == 0:
            R = correlation_matrix(feats)
            corr_log.append((step, np.asarray(R)))

        # Step env.
        actions_dict = {a: actions[i] for i, a in enumerate(agents)}
        key, k_step = jax.random.split(key)
        next_obs_dict, next_state, rewards, dones, info = env.step(
            k_step, state, actions_dict
        )

        # Shared reward (all agents identical for SMAX).
        r = float(rewards[agents[0]])
        done = bool(dones["__all__"])
        ep_return += r

        # Stack next.
        next_obs_arr = _stack_obs(next_obs_dict, agents)
        next_aug_obs = augment_obs(next_obs_arr, agent_id_oh, role_oh)
        next_avail = _stack_avail(env.get_avail_actions(next_state), agents)
        world_state_now = obs_dict["world_state"]
        world_state_next = next_obs_dict["world_state"]

        buf.add(
            obs=np.asarray(aug_obs),
            state=np.asarray(world_state_now),
            actions=np.asarray(actions),
            reward=r,
            next_obs=np.asarray(next_aug_obs),
            next_state=np.asarray(world_state_next),
            next_avail=np.asarray(next_avail),
            done=done,
        )

        # Train step.
        if buf.size >= cfg.warmup and step % cfg.train_every == 0:
            batch_np = buf.sample(cfg.batch_size, rng_np)
            batch_jax = {k: jnp.asarray(v) for k, v in batch_np.items()}
            opt_params, opt_state, loss = update(
                opt_params, opt_state, target_qnet_params, target_mixer_params,
                batch_jax,
            )
            losses.append(float(loss))

        # Sync targets.
        if step > 0 and step % cfg.target_sync == 0:
            target_qnet_params = opt_params["qnet"]
            target_mixer_params = opt_params["mixer"]

        # Episode bookkeeping.
        if done:
            ep_returns.append(ep_return)
            # SMAX info doesn't always include explicit "win" — we infer from
            # final reward magnitude (won_battle_bonus=1.0 added on win).
            ep_wins.append(1 if ep_return > 0.5 else 0)
            ep_return = 0.0
            key, k_reset = jax.random.split(key)
            obs_dict, state = env.reset(k_reset)
            obs_arr = _stack_obs(obs_dict, agents)
            aug_obs = augment_obs(obs_arr, agent_id_oh, role_oh)
        else:
            obs_dict, state = next_obs_dict, next_state
            obs_arr = next_obs_arr
            aug_obs = next_aug_obs

        if step % 5000 == 0:
            recent_n = max(1, len(ep_returns) // 5)
            recent_ret = (
                float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            )
            recent_win = (
                float(np.mean(ep_wins[-50:])) if ep_wins else 0.0
            )
            print(
                f"step={step:>7d}  eps={eps:.3f}  "
                f"episodes={len(ep_returns):>4d}  "
                f"recent_return={recent_ret:.2f}  win={recent_win:.2f}  "
                f"buf={buf.size}",
                flush=True,
            )

    return {
        "episode_returns": np.array(ep_returns, dtype=np.float32),
        "episode_wins": np.array(ep_wins, dtype=np.float32),
        "losses": np.array(losses, dtype=np.float32),
        "corr_log": corr_log,
        "config": cfg.__dict__,
    }
