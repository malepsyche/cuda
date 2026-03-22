#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_manifest(manifest_path: Path):
    df = pd.read_csv(manifest_path)
    if not {"step", "time", "csv_path"}.issubset(df.columns):
        raise ValueError("Manifest must contain step,time,csv_path columns")
    return df


def load_field(csv_path: Path):
    return np.loadtxt(csv_path, delimiter=",")


def make_heatmap(field, title, out_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(field, origin="lower", aspect="auto")
    plt.colorbar(label="u(x, y, t)")
    plt.title(title)
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_surface(field, title, out_path: Path):
    ny, nx = field.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, field, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_zlabel("u")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def make_animation(frames, title_prefix, out_path: Path, fps: int = 6):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0][1], origin="lower", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("u(x, y, t)")

    def update(frame_tuple):
        step, field = frame_tuple
        im.set_data(field)
        im.set_clim(vmin=field.min(), vmax=field.max())
        ax.set_title(f"{title_prefix} step={step}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 // max(fps, 1), blit=False)
    try:
        ani.save(out_path, writer="ffmpeg", fps=fps)
    except Exception:
        gif_path = out_path.with_suffix(".gif")
        ani.save(gif_path, writer="pillow", fps=fps)
        print(f"ffmpeg unavailable, wrote {gif_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate wave field visualizations from exported snapshots.")
    parser.add_argument("--manifest", required=True, help="Path to *_manifest.csv")
    parser.add_argument("--output-dir", default=None, help="Defaults to a visualizations/ subdir beside the manifest")
    parser.add_argument("--make-animation", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    frames = []
    for _, row in manifest.iterrows():
        csv_path = base_dir / row["csv_path"]
        field = load_field(csv_path)
        step = int(row["step"])
        time_value = float(row["time"])
        make_heatmap(field, f"Wave heatmap (step={step}, t={time_value:.4f})", out_dir / f"heatmap_step_{step}.png")
        make_surface(field, f"Wave surface (step={step}, t={time_value:.4f})", out_dir / f"surface_step_{step}.png")
        frames.append((step, field))

    if args.make_animation and frames:
        make_animation(frames, "Wave field", out_dir / "wave_animation.mp4")

    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
