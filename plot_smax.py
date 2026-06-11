"""Plot SMAX 2s3z learning curves: win rate vs training, exploration ablation.

All configs eventually reach ~100% win rate, so the interesting signal is the
*early-learning speed*. We bin each run's per-episode win flags onto a common
grid (by episode index, a proxy for training progress), average across seeds,
and overlay independent vs the three correlated variants per mixer.
"""

import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_BINS = 60
SMOOTH = 1  # bins already average ~500 episodes; extra smoothing adds edge artifacts


def load():
    groups = defaultdict(list)
    for p in sorted(glob.glob("results_smax/*.npz")):
        m = re.match(r".*smax_2s3z_(vdn|qmix)_(independent|correlated)_(\w+?)_s(\d+)",
                     Path(p).stem)
        if not m:
            continue
        mix, expl, sim, _ = m.groups()
        key = (mix, expl if expl == "independent" else f"corr-{sim}")
        d = np.load(p, allow_pickle=True)
        groups[key].append(d["episode_wins"])
    return groups


def binned_curve(runs):
    """Bin each run's win flags into N_BINS over its length, then mean across seeds."""
    mat = []
    for w in runs:
        idx = np.linspace(0, len(w), N_BINS + 1).astype(int)
        binned = np.array([w[idx[i]:idx[i + 1]].mean() for i in range(N_BINS)])
        mat.append(binned)
    mat = np.stack(mat)
    mean = mat.mean(0)
    std = mat.std(0)
    if SMOOTH > 1:
        k = np.ones(SMOOTH) / SMOOTH
        mean = np.convolve(mean, k, mode="same")
    return mean, std


CMAP = {
    "independent": ("tab:blue", "independent"),
    "corr-obs": ("tab:red", "correlated (obs)"),
    "corr-q_values": ("tab:green", "correlated (q_values)"),
    "corr-hidden": ("tab:purple", "correlated (hidden)"),
}


def main():
    groups = load()
    for mix in ["vdn", "qmix"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for expl in ["independent", "corr-obs", "corr-q_values", "corr-hidden"]:
            key = (mix, expl)
            if key not in groups:
                continue
            mean, std = binned_curve(groups[key])
            x = np.linspace(0, 100, len(mean))
            color, label = CMAP[expl]
            n = len(groups[key])
            ax.plot(x, mean, color=color, label=f"{label} (n={n})")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("training progress (% of episodes)")
        ax.set_ylabel("win rate")
        ax.set_title(f"SMAX 2s3z / {mix.upper()}: exploration ablation")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = f"plot_smax_{mix}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
