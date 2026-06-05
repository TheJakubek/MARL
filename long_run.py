"""Single training run with on-disk progress checkpoints.

Designed for batch use: every parameter is on the CLI, output goes to a single
.npz file. Use run_grid.py to launch many of these in parallel.
"""

import argparse
import sys
import time

import numpy as np
import torch

from agent import BalancedReplayBuffer, MARLLearner, ReplayBuffer
from exploration import CorrelatedEpsilonGreedy, IndependentEpsilonGreedy
from train import epsilon_schedule, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploration", choices=["independent", "correlated"], required=True)
    ap.add_argument("--mixer", choices=["vdn", "qmix"], default="vdn")
    ap.add_argument("--similarity", choices=["obs", "q_values", "hidden"], default="obs",
                    help="Source for correlation matrix (only used with --exploration correlated).")
    ap.add_argument("--env", choices=["switches", "lbf", "hallway"], default="lbf")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Path to .npz output file.")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect).")
    ap.add_argument("--balanced-buffer", action="store_true",
                    help="Use BalancedReplayBuffer (oversample success transitions in batches).")
    ap.add_argument("--parameter-sharing", action="store_true",
                    help="Share QNet across agents + agent_id one-hot in obs.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = make_env(args.env, seed=args.seed)
    learner = MARLLearner(
        n_agents=env.N_AGENTS,
        obs_dim=env.obs_dim,
        n_actions=env.N_ACTIONS,
        mixer_kind=args.mixer,
        device=args.device,
        parameter_sharing=args.parameter_sharing,
    )
    if args.balanced_buffer:
        buf = BalancedReplayBuffer(capacity=50_000, n_agents=env.N_AGENTS)
    else:
        buf = ReplayBuffer(capacity=50_000, n_agents=env.N_AGENTS)

    if args.exploration == "independent":
        strategy = IndependentEpsilonGreedy(env.N_AGENTS, env.N_ACTIONS, rng=rng)
    else:
        strategy = CorrelatedEpsilonGreedy(
            env.N_AGENTS,
            env.N_ACTIONS,
            rng=rng,
            similarity_source=args.similarity,
        )

    returns = []
    successes = []
    total_steps = 0
    learning_starts = 2000
    update_every = 4
    batch_size = 128
    t0 = time.time()

    config = dict(
        exploration=args.exploration,
        mixer=args.mixer,
        similarity=args.similarity,
        env=args.env,
        seed=args.seed,
        episodes=args.episodes,
        device=str(learner.device),
    )

    need_hidden = (
        args.exploration == "correlated" and args.similarity == "hidden"
    )
    for ep in range(args.episodes):
        obs = env.reset()
        eps = epsilon_schedule(ep, args.episodes)
        ep_r = 0.0
        success = False
        done = False
        while not done:
            if need_hidden:
                qs, hs = learner.q_values(obs, return_hidden=True)
                actions = strategy.select(qs, obs, eps, hidden_list=hs)
            else:
                qs = learner.q_values(obs)
                actions = strategy.select(qs, obs, eps)
            next_obs, r, done, info = env.step(actions)
            buf.push(obs, actions, r, next_obs, float(done))
            obs = next_obs
            ep_r += r
            total_steps += 1
            success = info.get("success", False) or success
            if (
                len(buf) >= max(batch_size, learning_starts)
                and total_steps % update_every == 0
            ):
                learner.update(buf.sample(batch_size))
        returns.append(ep_r)
        successes.append(int(success))

        if (ep + 1) % args.save_every == 0:
            elapsed = time.time() - t0
            recent = float(np.mean(successes[-args.save_every :]))
            buf_total = len(buf)
            buf_success = buf.success_count()
            buf_pct = (buf_success / buf_total) if buf_total > 0 else 0.0
            np.savez(
                args.out,
                returns=np.array(returns),
                successes=np.array(successes),
                episodes_done=ep + 1,
                elapsed_sec=elapsed,
                final_buf_total=buf_total,
                final_buf_success=buf_success,
                **{f"cfg_{k}": str(v) for k, v in config.items()},
            )
            tag = f"{args.exploration[:4]}/{args.mixer}/{args.similarity}/s{args.seed}"
            print(
                f"[{tag}] ep={ep + 1}/{args.episodes}  "
                f"eps={eps:.2f}  recent_success={recent:.2f}  "
                f"buf={buf_success}/{buf_total} ({100 * buf_pct:.2f}%)  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    np.savez(
        args.out,
        returns=np.array(returns),
        successes=np.array(successes),
        episodes_done=args.episodes,
        elapsed_sec=time.time() - t0,
        **{f"cfg_{k}": str(v) for k, v in config.items()},
    )
    print(f"DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
