"""Plot one shot from raw bin volumes: input, denoised, clean label, residual."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np


_SHAPE_RE = re.compile(r"_ns(?P<ns>\d+)ng(?P<ng>\d+)nt(?P<nt>\d+)\.bin$")


def _infer_shape(path: Path) -> tuple[int, int, int]:
    match = _SHAPE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer shape from {path.name}; pass --shape ns ng nt.")
    return int(match["ns"]), int(match["ng"]), int(match["nt"])


def _read_bin(path: str, shape: Optional[list[int]], dtype: str) -> np.ndarray:
    p = Path(path)
    shp = tuple(shape) if shape is not None else _infer_shape(p)
    arr = np.fromfile(p, dtype=np.dtype(dtype))
    expected = int(np.prod(shp))
    if arr.size != expected:
        raise ValueError(f"{p}: expected {expected} samples for shape {shp}, got {arr.size}.")
    return arr.reshape(shp).astype(np.float32, copy=False)


def _clip_value(*arrays: np.ndarray, percentile: float) -> float:
    vals = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrays])
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(vals, percentile))
    return v if v > 0.0 else float(vals.max() or 1.0)


def _slugify_title(title: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", title.strip()).strip("_").lower()
    return slug or "panel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one shot from input/prediction/label bin files.")
    parser.add_argument("--input-bin", default="/data/bhy/multiple/data/test/free_surface_ns88ng481nt3300.bin", help="Noisy/input free-surface bin.")
    parser.add_argument("--pred-bin", default="/data/bhy/multiple/outputs/260626v1/diffraction_multiples_atten_unet/inference_test/pred_test_ns88ng481nt3300.bin", help="Denoised/predicted bin.")
    parser.add_argument("--label-bin", default="/data/bhy/multiple/data/test/sim_abs_ghost_ns88ng481nt3300.bin", help="Clean label bin.")
    parser.add_argument("--shot", type=int, default=0, help="0-based shot index to plot.")
    parser.add_argument("--shape", type=int, nargs=3, default=None, metavar=("NS", "NG", "NT"))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to <pred-bin-dir>/plots/shot_<shot>_comparison.png.",
    )
    parser.add_argument(
        "--save-subplots",
        action="store_true",
        help="Also save each panel as a separate PNG using the same vmin/vmax as the combined figure.",
    )
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    parser.add_argument("--vmin", type=float, default=None, help="Manual color scale minimum.")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual color scale maximum. If only --vmax is set, vmin=-vmax.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    noisy = _read_bin(args.input_bin, args.shape, args.dtype)
    pred = _read_bin(args.pred_bin, args.shape, args.dtype)
    label = _read_bin(args.label_bin, args.shape, args.dtype)
    if noisy.shape != pred.shape or noisy.shape != label.shape:
        raise ValueError(f"Shape mismatch: input={noisy.shape}, pred={pred.shape}, label={label.shape}.")

    if not 0 <= args.shot < noisy.shape[0]:
        raise IndexError(f"--shot must be in [0, {noisy.shape[0] - 1}], got {args.shot}.")

    input_shot = noisy[args.shot]
    pred_shot = pred[args.shot]
    label_shot = label[args.shot]
    residual = pred_shot - label_shot
    if args.vmin is not None or args.vmax is not None:
        if args.vmax is None:
            raise ValueError("--vmax is required when --vmin is set.")
        vmin = -float(args.vmax) if args.vmin is None else float(args.vmin)
        vmax = float(args.vmax)
        if vmin >= vmax:
            raise ValueError(f"Expected vmin < vmax, got vmin={vmin}, vmax={vmax}.")
    else:
        v = _clip_value(input_shot, pred_shot, label_shot, residual, percentile=args.clip_percentile)
        vmin, vmax = -v, v

    import matplotlib.pyplot as plt

    panels = [
        ("before denoise", input_shot),
        ("after denoise", pred_shot),
        ("clean label", label_shot),
        ("residual", residual),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(arr.T, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("trace")
        ax.set_ylabel("time sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"shot {args.shot}")
    fig.tight_layout()

    if args.out is None:
        out = Path(args.pred_bin).resolve().parent / "plots" / f"shot_{args.shot:04d}_comparison.png"
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    if args.save_subplots:
        subplots_dir = out.parent / f"{out.stem}_subplots"
        subplots_dir.mkdir(parents=True, exist_ok=True)
        for idx, (title, arr) in enumerate(panels, start=1):
            panel_fig, panel_ax = plt.subplots(1, 1, figsize=(5, 5))
            im = panel_ax.imshow(arr.T, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
            panel_ax.set_title(title)
            panel_ax.set_xlabel("trace")
            panel_ax.set_ylabel("time sample")
            panel_fig.colorbar(im, ax=panel_ax, fraction=0.046, pad=0.04)
            panel_fig.tight_layout()
            panel_path = subplots_dir / f"{idx:02d}_{_slugify_title(title)}.png"
            panel_fig.savefig(panel_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(panel_fig)
            print(f"Saved subplot: {panel_path}")


if __name__ == "__main__":
    main()
