"""Training loop for VDN with pluggable exploration strategy."""

import numpy as np
import torch

from agent import MARLLearner, ReplayBuffer
from env import CoordinatedSwitches
from exploration import CorrelatedEpsilonGreedy, IndependentEpsilonGreedy
from hallway_env import Hallway
from lbf_env import LBFAdapter


def epsilon_schedule(episode: int, n_episodes: int, eps_start=1.0, eps_end=0.3):
    decay_episodes = int(0.8 * n_episodes)
    frac = min(1.0, episode / max(1, decay_episodes))
    return eps_start + (eps_end - eps_start) * frac


def make_env(env_kind: str, seed: int):
    if env_kind == "switches":
        return CoordinatedSwitches(seed=seed)
    if env_kind == "lbf":
        return LBFAdapter(seed=seed)
    if env_kind == "hallway":
        return Hallway(seed=seed)
    raise ValueError(env_kind)


def train(
    exploration_kind: str,
    env_kind: str = "lbf",
    mixer_kind: str = "vdn",
    similarity_source: str = "obs",
    n_episodes: int = 600,
    seed: int = 0,
    batch_size: int = 128,
    buffer_capacity: int = 50_000,
    learning_starts: int = 2000,
    update_every: int = 4,
    log_every: int = 20,
    verbose: bool = True,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = make_env(env_kind, seed=seed)
    learner = MARLLearner(
        n_agents=env.N_AGENTS,
        obs_dim=env.obs_dim,
        n_actions=env.N_ACTIONS,
        mixer_kind=mixer_kind,
    )
    buf = ReplayBuffer(capacity=buffer_capacity, n_agents=env.N_AGENTS)

    if exploration_kind == "independent":
        strategy = IndependentEpsilonGreedy(env.N_AGENTS, env.N_ACTIONS, rng=rng)
    elif exploration_kind == "correlated":
        strategy = CorrelatedEpsilonGreedy(
            env.N_AGENTS,
            env.N_ACTIONS,
            rng=rng,
            similarity_source=similarity_source,
        )
    else:
        raise ValueError(exploration_kind)

    episode_returns = []
    episode_successes = []

    total_steps = 0
    for ep in range(n_episodes):
        obs = env.reset()
        epsilon = epsilon_schedule(ep, n_episodes)
        ep_return = 0.0
        success = False

        need_hidden = (
            exploration_kind == "correlated" and similarity_source == "hidden"
        )
        done = False
        while not done:
            if need_hidden:
                q_list, h_list = learner.q_values(obs, return_hidden=True)
                actions = strategy.select(q_list, obs, epsilon, hidden_list=h_list)
            else:
                q_list = learner.q_values(obs)
                actions = strategy.select(q_list, obs, epsilon)

            next_obs, reward, done, info = env.step(actions)
            buf.push(obs, actions, reward, next_obs, float(done))

            obs = next_obs
            ep_return += reward
            total_steps += 1
            success = info.get("success", False) or success

            if (
                len(buf) >= max(batch_size, learning_starts)
                and total_steps % update_every == 0
            ):
                batch = buf.sample(batch_size)
                learner.update(batch)

        episode_returns.append(ep_return)
        episode_successes.append(int(success))

        if verbose and (ep + 1) % log_every == 0:
            recent_r = np.mean(episode_returns[-log_every:])
            recent_s = np.mean(episode_successes[-log_every:])
            print(
                f"[{exploration_kind}] ep {ep + 1:4d}  eps={epsilon:.2f}  "
                f"return={recent_r:+.3f}  success_rate={recent_s:.2f}"
            )

    return {
        "returns": np.array(episode_returns),
        "successes": np.array(episode_successes),
    }


if __name__ == "__main__":
    import sys

    kind = sys.argv[1] if len(sys.argv) > 1 else "independent"
    train(kind)
