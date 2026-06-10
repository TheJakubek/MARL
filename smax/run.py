"""CLI entry point for SMAX 2s3z training.

Usage:
    python -m smax.run --exploration correlated --mixer qmix \
        --similarity hidden --seed 0 --total-steps 200000 \
        --out results_smax/qmix_corr_hidden_s0.npz
"""

import argparse
from pathlib import Path

import numpy as np

from smax.train import Config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploration", choices=["independent", "correlated"],
                    default="independent")
    ap.add_argument("--similarity", choices=["obs", "q_values", "hidden"],
                    default="obs")
    ap.add_argument("--mixer", choices=["vdn", "qmix"], default="vdn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--buffer-cap", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = Config(
        seed=args.seed,
        total_steps=args.total_steps,
        buffer_cap=args.buffer_cap,
        batch_size=args.batch_size,
        warmup=args.warmup,
        lr=args.lr,
        exploration=args.exploration,
        similarity=args.similarity,
        mixer=args.mixer,
    )

    print(f"[run] cfg={cfg.__dict__}", flush=True)
    result = train(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        episode_returns=result["episode_returns"],
        episode_wins=result["episode_wins"],
        losses=result["losses"],
        corr_log_steps=np.array([s for s, _ in result["corr_log"]], dtype=np.int32),
        corr_log_matrices=np.stack(
            [m for _, m in result["corr_log"]], axis=0
        ) if result["corr_log"] else np.zeros((0, 5, 5), dtype=np.float32),
        config=str(result["config"]),
    )
    print(f"[run] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
