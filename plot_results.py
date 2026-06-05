"""Plot learning curves from results/*.npz files.

Groups runs by (env, mixer, exploration, similarity) and plots mean ± std
of success rate across seeds.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(x, window=20):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def parse_tag(name: str):
    # tag format: <env>_<mixer>_<exploration>_<sim_or_na>_s<seed>
    m = re.match(r"(.+?)_(\w+?)_(\w+?)_(\w+?)_s(\d+)", name)
    if not m:
        return None
    env, mixer, expl, sim, seed_str = m.groups()
    return env, mixer, expl, sim, int(seed_str)


def load_runs(results_dir: Path):
    """Returns dict: (env, mixer, exploration, sim) -> list of success arrays."""
    grouped = defaultdict(list)
    for path in sorted(results_dir.glob("*.npz")):
        parsed = parse_tag(path.stem)
        if parsed is None:
            continue
        env, mixer, expl, sim, _seed = parsed
        d = np.load(path, allow_pickle=True)
        grouped[(env, mixer, expl, sim)].append(d["successes"])
    return grouped


def plot(grouped, env_filter=None, mixer_filter=None, out="plot.png", window=20):
    keys = sorted(grouped.keys())
    if env_filter:
        keys = [k for k in keys if k[0] == env_filter]
    if mixer_filter:
        keys = [k for k in keys if k[1] == mixer_filter]
    if not keys:
        print(f"No runs match env={env_filter} mixer={mixer_filter}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = {
        ("independent", "na"): ("tab:blue", "independent eps-greedy"),
        ("correlated", "obs"): ("tab:red", "correlated (obs)"),
        ("correlated", "q_values"): ("tab:green", "correlated (q_values)"),
        ("correlated", "hidden"): ("tab:purple", "correlated (hidden)"),
    }

    for env, mixer, expl, sim in keys:
        runs = grouped[(env, mixer, expl, sim)]
        if not runs:
            continue
        # Trim to common length, smooth, stack across seeds.
        min_len = min(len(r) for r in runs)
        smoothed = np.stack([smooth(r[:min_len], window) for r in runs])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean)) + window
        color, label = cmap.get((expl, sim), (None, f"{expl}/{sim}"))
        ax.plot(x, mean, label=f"{label} (n={len(runs)})", color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    title_env = env_filter or keys[0][0]
    title_mixer = mixer_filter or keys[0][1]
    ax.set_xlabel("episode")
    ax.set_ylabel(f"success rate (smoothed, window={window})")
    ax.set_title(f"{title_env} / {title_mixer}: exploration ablation")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--env", default=None)
    ap.add_argument("--mixer", default=None)
    ap.add_argument("--out", default="plot.png")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    grouped = load_runs(Path(args.results))
    if not grouped:
        print(f"No .npz files in {args.results}")
        return

    print("Available run groups:")
    for k, runs in sorted(grouped.items()):
        print(f"  {k}  -> {len(runs)} seed(s)")

    plot(grouped, env_filter=args.env, mixer_filter=args.mixer,
         out=args.out, window=args.window)


if __name__ == "__main__":
    main()
