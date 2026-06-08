"""Launch a grid of long_run.py jobs in parallel with concurrency limit.

Default grid covers the main experimental dimensions for the project:
  - mixer:        vdn, qmix
  - exploration:  independent, correlated
  - similarity:   obs, q_values  (only meaningful for correlated)
  - env:          switches, lbf
  - seeds:        configurable

Outputs go to results/<tag>.npz where tag encodes the config. Reruns are
idempotent: existing complete .npz files are skipped.
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


def make_tag(env, mixer, exploration, similarity, seed):
    sim = similarity if exploration == "correlated" else "na"
    return f"{env}_{mixer}_{exploration}_{sim}_s{seed}"


def is_complete(out_path: Path, episodes: int) -> bool:
    if not out_path.exists():
        return False
    try:
        d = np.load(out_path, allow_pickle=True)
        return int(d["episodes_done"]) >= episodes
    except Exception:
        return False


def run_one(args_tuple):
    """Worker: subprocess.run a single long_run.py job."""
    (env, mixer, exploration, similarity, seed, episodes,
     out_dir, log_dir, parameter_sharing, balanced_buffer) = args_tuple
    tag = make_tag(env, mixer, exploration, similarity, seed)
    out_path = Path(out_dir) / f"{tag}.npz"
    log_path = Path(log_dir) / f"{tag}.log"

    if is_complete(out_path, episodes):
        return tag, "skipped", 0.0

    cmd = [
        sys.executable,
        "long_run.py",
        "--exploration", exploration,
        "--mixer", mixer,
        "--similarity", similarity,
        "--env", env,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--out", str(out_path),
        "--save-every", "100",
    ]
    if parameter_sharing:
        cmd.append("--parameter-sharing")
    if balanced_buffer:
        cmd.append("--balanced-buffer")
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
    ap.add_argument("--envs", nargs="+", default=["switches", "lbf"])
    ap.add_argument("--mixers", nargs="+", default=["vdn", "qmix"])
    ap.add_argument("--explorations", nargs="+", default=["independent", "correlated"])
    ap.add_argument("--similarities", nargs="+", default=["obs", "q_values", "hidden"],
                    help="Only used for correlated exploration.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Max parallel jobs. Set to 1 to serialize.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--parameter-sharing", action="store_true",
                    help="Pass --parameter-sharing to every job.")
    ap.add_argument("--balanced-buffer", action="store_true",
                    help="Pass --balanced-buffer to every job.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Build grid. For independent exploration, similarity is a no-op,
    # so we collapse it to a single entry.
    jobs = []
    for env, mixer, expl, seed in itertools.product(
        args.envs, args.mixers, args.explorations, args.seeds
    ):
        sims = args.similarities if expl == "correlated" else ["obs"]
        for sim in sims:
            jobs.append((
                env, mixer, expl, sim, seed, args.episodes,
                args.out_dir, args.log_dir,
                args.parameter_sharing, args.balanced_buffer,
            ))

    print(f"Planned {len(jobs)} jobs (concurrency={args.concurrency}):")
    for j in jobs:
        env, mixer, expl, sim, seed, *_ = j
        tag = make_tag(env, mixer, expl, sim, seed)
        out_path = Path(args.out_dir) / f"{tag}.npz"
        marker = " [done]" if is_complete(out_path, args.episodes) else ""
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
