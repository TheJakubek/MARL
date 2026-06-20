"""Learned correlation matrices (end of training) for the three similarity sources.

Averaged over seeds. Text colour adapts to cell brightness so values stay
readable on the yellow (high-correlation) cells.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np

LABELS = ["stalker_0", "stalker_1", "zealot_0", "zealot_1", "zealot_2"]
SIMS = [("obs", "obs similarity"),
        ("q_values", "q_values similarity"),
        ("hidden", "hidden similarity")]


def avg_final_corr(sim):
    mats = []
    for p in sorted(glob.glob(f"results_smax/smax_2s3z_vdn_correlated_{sim}_s*.npz")):
        d = np.load(p, allow_pickle=True)
        M = d["corr_log_matrices"]
        if M.shape[0] > 0:
            mats.append(M[-1])
    return np.mean(mats, axis=0)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for ax, (sim, title) in zip(axes, SIMS):
        R = avg_final_corr(sim)
        im = ax.imshow(R, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        for i in range(5):
            for j in range(5):
                # Black text on bright (viridis > ~0.5 -> greenish/yellow) cells.
                color = "black" if R[i, j] > 0.55 else "white"
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        ax.set_title(f"SMAX 2s3z: learned correlation ({title})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("plot_smax_corr_matrix.png", dpi=120)
    print("saved plot_smax_corr_matrix.png")


if __name__ == "__main__":
    main()
