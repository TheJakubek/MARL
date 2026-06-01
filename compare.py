"""Run independent vs correlated exploration over multiple seeds, plot mean +/- std."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from train import train


def smooth(x, window=20):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def run_many(kind: str, env_kind: str, n_episodes: int, seeds: list[int]):
    runs = []
    for s in seeds:
        print(f"=== {kind} seed={s} ===")
        out = train(
            kind,
            env_kind=env_kind,
            n_episodes=n_episodes,
            seed=s,
            log_every=max(50, n_episodes // 10),
            verbose=True,
        )
        runs.append(out["returns"])
    return np.stack(runs)  # (n_seeds, n_episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="lbf", choices=["lbf", "switches"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()

    indep = run_many("independent", args.env, args.episodes, args.seeds)
    corr = run_many("correlated", args.env, args.episodes, args.seeds)

    window = max(20, args.episodes // 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in [
        ("independent eps-greedy", indep, "tab:blue"),
        ("correlated (Gaussian copula)", corr, "tab:red"),
    ]:
        smoothed = np.stack([smooth(r, window) for r in data])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean)) + window
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("episode")
    ax.set_ylabel(f"return (smoothed, window={window})")
    ax.set_title(f"VDN on {args.env}: independent vs correlated exploration ({len(args.seeds)} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
