"""How the learned correlation's *role contrast* evolves over training.

The copula recomputes the correlation matrix every step from current features,
and exploration matters most early (high epsilon). So *when* the matrix carries
role structure matters. We summarise each logged snapshot by a single number:

    role_contrast = mean(within-role corr) - mean(cross-role corr)

High contrast = the method couples same-role agents (the intended behaviour);
~0 = uniform correlation (no role information used).

In 2s3z the allies are [stalker_0, stalker_1, zealot_0, zealot_1, zealot_2].
"""

import glob

import matplotlib.pyplot as plt
import numpy as np

# index pairs
WITHIN = [(0, 1), (2, 3), (2, 4), (3, 4)]          # stalker-stalker, zealot-zealot
CROSS = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]  # stalker-zealot


def contrast(R):
    w = np.mean([R[i, j] for i, j in WITHIN])
    c = np.mean([R[i, j] for i, j in CROSS])
    return w - c


def curves(sim):
    """Return (steps, mean_contrast, std_contrast) across seeds."""
    per_seed = []
    steps = None
    for p in sorted(glob.glob(f"results_smax/smax_2s3z_vdn_correlated_{sim}_s*.npz")):
        d = np.load(p, allow_pickle=True)
        M, S = d["corr_log_matrices"], d["corr_log_steps"]
        if M.shape[0] == 0:
            continue
        per_seed.append([contrast(M[k]) for k in range(M.shape[0])])
        steps = S
    arr = np.array(per_seed)
    return steps, arr.mean(0), arr.std(0)


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"obs": "tab:red", "q_values": "tab:green", "hidden": "tab:purple"}
    for sim in ["obs", "q_values", "hidden"]:
        steps, mean, std = curves(sim)
        ax.plot(steps, mean, "o-", color=colors[sim], label=f"correlated ({sim})")
        ax.fill_between(steps, mean - std, mean + std, color=colors[sim], alpha=0.15)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("role contrast  (within-role − cross-role correlation)")
    ax.set_title("SMAX 2s3z: does the correlation encode roles, and when?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("plot_smax_corr_evolution.png", dpi=120)
    print("saved plot_smax_corr_evolution.png")


if __name__ == "__main__":
    main()
