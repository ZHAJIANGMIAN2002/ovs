#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np

# Ensure project root is on sys.path so imports work regardless of CWD
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.label_constants import SCANNET_LABELS_20
from util.util import get_palette
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate ScanNet-20 palette image (2 rows x 10).")
    parser.add_argument("--out", type=str, default="out/scannet_palette.jpg", help="Output image path")
    parser.add_argument("--ncol", type=int, default=10, help="Number of columns per row")
    parser.add_argument("--fontsize", type=int, default=12, help="Font size for labels")
    parser.add_argument("--square", type=float, default=0.3, help="Square size relative to axes (0-1)")
    parser.add_argument("--style", type=str, default="legend", choices=["legend", "grid"],
                        help="legend: exactly match eval legend style; grid: custom 2x10 grid")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    labels = list(SCANNET_LABELS_20)
    assert len(labels) == 20, "Expect 20 ScanNet classes"

    if args.style == "legend":
        # Use the exact evaluation legend drawing for identical look
        from util.util import visualize_labels
        palette = get_palette(num_cls=20, colormap='scannet')
        tmp_png = args.out.rsplit(".", 1)[0] + "_tmp.png"
        visualize_labels(list(range(20)), labels, palette, tmp_png, ncol=args.ncol)
        # Load PNG, composite onto pure white and save as JPG/target format
        im = Image.open(tmp_png).convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        bg.save(args.out, quality=95)
        try:
            os.remove(tmp_png)
        except Exception:
            pass
        print(f"Saved palette to {args.out}")
        return

    # grid style (custom 2x10, pure white background, square patches)
    palette = get_palette(num_cls=20, colormap='scannet')  # flat array [r,g,b,...]
    colors = [np.array(palette[i * 3:(i + 1) * 3]) / 255.0 for i in range(20)]

    # Create a clean white canvas, 2 rows x ncol columns
    n_rows, n_cols = 2, args.ncol
    fig_w, fig_h = 18, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), facecolor="white")
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(20):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")

        # Draw color square (no border)
        s = max(0.05, min(0.9, args.square))
        x0, y0 = 0.5 - s / 2, 0.5 - s / 2  # center the square
        rect = plt.Rectangle((x0, y0), s, s, facecolor=colors[idx], edgecolor="none")
        ax.add_patch(rect)

        # Text aligned to the right of square
        ax.text(
            min(0.95, x0 + s + 0.06), y0 + s / 2,
            labels[idx],
            va="center", ha="left",
            fontsize=args.fontsize, color="black"
        )

    plt.tight_layout(pad=0.2)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved palette to {args.out}")


if __name__ == "__main__":
    main()


