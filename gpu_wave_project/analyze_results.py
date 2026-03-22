#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Analyze scaling results from summary_metrics.csv")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    df["grid_points"] = df["nx"] * df["ny"]

    for ycol, ylabel, fname in [
        ("avg_kernel_ms", "Average kernel time per timestep (ms)", "kernel_time_vs_grid.png"),
        ("effective_bandwidth_gbps", "Effective memory bandwidth (GB/s)", "bandwidth_vs_grid.png"),
        ("total_sim_ms", "Total simulation time (ms)", "total_time_vs_grid.png"),
    ]:
        plt.figure(figsize=(7, 5))
        for method, sub in df.groupby("method"):
            sub = sub.sort_values("grid_points")
            plt.plot(sub["grid_points"], sub[ycol], marker="o", label=method)
        plt.xlabel("Total grid points")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()

    pivot = df[["method", "domain_length", "grid_points", "avg_kernel_ms", "effective_bandwidth_gbps", "total_sim_ms"]]
    pivot.to_csv(out_dir / "analysis_table.csv", index=False)
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
