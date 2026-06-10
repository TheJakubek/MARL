"""Grid runner for SMAX experiments.

Mirrors run_grid.py but for SMAX 2s3z. Idempotent: skips finished .npz files.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def make_tag(mixer, exploration, similarity, seed):
    sim = similarity if exploration == "correlated" else "na"
    return f"smax_2s3z_{mixer}_{exploration}_{sim}_s{seed}"


def is_complete(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        d = np.load(out_path, allow_pickle=True)
        return "episode_returns" in d.files and d["episode_returns"].size > 0
    except Exception:
        return False


def run_one(args_tuple):
    mixer, exploration, similarity, seed, total_steps, out_dir, log_dir = args_tuple
    tag = make_tag(mixer, exploration, similarity, seed)
    out_path = Path(out_dir) / f"{tag}.npz"
    log_path = Path(log_dir) / f"{tag}.log"

    if is_complete(out_path):
        return tag, "skipped", 0.0

    cmd = [
        sys.executable, "-m", "smax.run",
        "--exploration", exploration,
        "--similarity", similarity,
        "--mixer", mixer,
        "--seed", str(seed),
        "--total-steps", str(total_steps),
        "--out", str(out_path),
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"# {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    return tag, status, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixers", nargs="+", default=["vdn", "qmix"])
    ap.add_argument("--explorations", nargs="+", default=["independent", "correlated"])
    ap.add_argument("--similarities", nargs="+", default=["obs", "q_values", "hidden"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out-dir", default="results_smax")
    ap.add_argument("--log-dir", default="logs_smax")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    jobs = []
    for mixer, expl, seed in itertools.product(args.mixers, args.explorations, args.seeds):
        sims = args.similarities if expl == "correlated" else ["obs"]
        for sim in sims:
            jobs.append((
                mixer, expl, sim, seed, args.total_steps,
                args.out_dir, args.log_dir,
            ))

    print(f"Planned {len(jobs)} jobs (concurrency={args.concurrency}):")
    for j in jobs:
        mixer, expl, sim, seed, *_ = j
        tag = make_tag(mixer, expl, sim, seed)
        out_path = Path(args.out_dir) / f"{tag}.npz"
        marker = " [done]" if is_complete(out_path) else ""
        print(f"  {tag}{marker}")

    if args.dry_run:
        return

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            tag, status, elapsed = fut.result()
            done += 1
            print(
                f"[{done}/{len(jobs)}] {tag}  status={status}  ({elapsed:.0f}s)  "
                f"total_elapsed={(time.time() - t0):.0f}s",
                flush=True,
            )
    print(f"All done in {(time.time() - t0):.0f}s.")


if __name__ == "__main__":
    main()
