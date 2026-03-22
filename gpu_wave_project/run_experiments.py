#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path

DEFAULT_LENGTHS = [1, 2, 4, 8]
DEFAULT_METHODS = ["global", "shared", "cusparse"]


def run_cmd(cmd, cwd=None):
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_metrics(path: Path):
    with path.open() as f:
        return next(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Run scaling experiments for the GPU wave solver.")
    parser.add_argument("--binary", default="./wave2d_solver")
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dy", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--block-x", type=int, default=16)
    parser.add_argument("--block-y", type=int, default=16)
    parser.add_argument("--lengths", nargs="*", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--with-snapshots", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for L in args.lengths:
        nx = int(round(L / args.dx)) + 1
        ny = int(round(L / args.dy)) + 1
        for method in args.methods:
            run_dir = out_root / f"{method}_L{L}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.binary,
                "--method", method,
                "--nx", str(nx),
                "--ny", str(ny),
                "--steps", str(args.steps),
                "--dx", str(args.dx),
                "--dy", str(args.dy),
                "--dt", str(args.dt),
                "--c", str(args.c),
                "--block-x", str(args.block_x),
                "--block-y", str(args.block_y),
                "--snapshot-every", str(args.snapshot_every),
                "--output-dir", str(run_dir),
            ]
            if not args.with_snapshots:
                cmd.append("--no-snapshots")
            run_cmd(cmd)
            metrics_path = run_dir / f"{method}_metrics.csv"
            row = read_metrics(metrics_path)
            row["domain_length"] = L
            summary_rows.append(row)

    summary_path = out_root / "summary_metrics.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
