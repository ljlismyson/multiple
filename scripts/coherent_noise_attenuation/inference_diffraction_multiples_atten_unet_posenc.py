"""Attention U-Net inference with linear patch-position channels.

Example:
    python scripts/coherent_noise_attenuation/inference_diffraction_multiples_atten_unet_posenc.py \
        --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet_posenc.yaml \
        --checkpoint results/diffraction_multiples_atten_unet_posenc/checkpoints/epoch_0199.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root (a directory containing both ``model/`` and ``utils/``).")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.coherent_noise_attenuation import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import patchify_uniform, unpatchify_uniform  # noqa: E402
from tools.position_encoding import append_linear_position_channels  # noqa: E402
from tools.preprocessing import denormalize, normalize  # noqa: E402
from utils import count_parameters, load_checkpoint, load_config  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    compute_shot_metrics,
    save_shot_visualizations,
    select_random_shots,
)
from utils.metrics import format_metric_value  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Attention U-Net inference on a pre-split paired volume. "
            "CLI arguments override config.inference."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/coherent_noise_attenuation/diffraction_multiples_atten_unet_posenc.yaml",
        help="Path to the training/inference config.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt.")
    parser.add_argument("--split", type=str, default=None, choices=("train", "val", "test"), help="Split to infer.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--max-shots", type=int, default=None, help="Optional number of shots to infer.")
    parser.add_argument("--n-viz-shots", type=int, default=None, help="Number of random shots to visualize.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for visualization selection.")
    parser.add_argument("--save-npy", action="store_true", default=None, help="Save input/pred/target .npy files.")
    parser.add_argument("--save-bin", action="store_true", default=None, help="Save predicted volume as raw float32 .bin.")
    return parser.parse_args()


def _resolve_arg(args_value: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return args_value if args_value is not None else cfg.get(key, default)


def _pair_for_split(data_cfg: Dict[str, Any], split: str) -> Dict[str, Any]:
    key = f"{split}_pair"
    if key not in data_cfg:
        raise ValueError(f"Config does not define data.{key}.")
    return data_cfg[key]


def _load_pair(pair_cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    input_cfg = dict(pair_cfg)
    input_cfg["path"] = pair_cfg["input_path"]
    target_cfg = dict(pair_cfg)
    target_cfg["path"] = pair_cfg["target_path"]
    input_shots = load_volume(input_cfg)
    target_shots = load_volume(target_cfg)
    if input_shots.shape != target_shots.shape:
        raise ValueError(f"Shape mismatch: input {input_shots.shape} vs target {target_shots.shape}.")
    return input_shots, target_shots


def _inference_on_shots_posenc(
    model: torch.nn.Module,
    input_shots: np.ndarray,
    *,
    patch_size: tuple[int, int],
    overlap: float,
    device: torch.device,
    batch_size: int,
    position_encoding_range: str,
) -> np.ndarray:
    patches, info = patchify_uniform(
        input_shots, patch_size=patch_size, overlap=overlap, output_ndim=4
    )
    patches = append_linear_position_channels(
        patches,
        info,
        value_range=position_encoding_range,  # type: ignore[arg-type]
    )
    ds = TensorDataset(torch.from_numpy(patches.astype(np.float32, copy=False)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    was_training = model.training
    model.eval()
    preds: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device, non_blocking=True)
                preds.append(model(batch).cpu())
    finally:
        if was_training:
            model.train()

    pred_patches = torch.cat(preds, dim=0).numpy()
    return unpatchify_uniform(pred_patches, info)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    infer_cfg = cfg.get("inference", {})

    checkpoint = _resolve_arg(args.checkpoint, infer_cfg, "checkpoint", None)
    if checkpoint is None:
        raise ValueError("--checkpoint is required, or set inference.checkpoint in the config.")
    split = str(_resolve_arg(args.split, infer_cfg, "split", "test"))
    output_dir = _resolve_arg(args.output_dir, infer_cfg, "output_dir", None)
    device = torch.device(_resolve_arg(args.device, infer_cfg, "device", cfg["experiment"].get("device", "cpu")))
    batch_size = int(_resolve_arg(args.batch_size, infer_cfg, "batch_size", cfg["data"].get("loader", {}).get("batch_size", 8)))
    max_shots = _resolve_arg(args.max_shots, infer_cfg, "max_shots", None)
    n_viz_shots = int(_resolve_arg(args.n_viz_shots, infer_cfg, "n_viz_shots", 5))
    seed = int(_resolve_arg(args.seed, infer_cfg, "seed", cfg["experiment"].get("seed", 42)))
    save_npy = bool(_resolve_arg(args.save_npy, infer_cfg, "save_npy", False))
    save_bin = bool(_resolve_arg(args.save_bin, infer_cfg, "save_bin", True))

    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    print(f"Model parameters: {count_parameters(model)}")

    pair_cfg = _pair_for_split(cfg["data"], split)
    input_shots, target_shots = _load_pair(pair_cfg)
    if max_shots is not None:
        m = int(max_shots)
        if m <= 0:
            raise ValueError(f"max_shots must be positive, got {m}.")
        input_shots = input_shots[:m]
        target_shots = target_shots[:m]

    prep = cfg["preprocess"]
    skip = set(prep.get("skip", []))
    norm_mode = str(prep.get("normalize_mode", "max_abs"))
    norm_scope = str(prep.get("normalize_scope", "global"))
    clip_raw = prep.get("clip_percentile")
    clip_p = float(clip_raw) if clip_raw is not None else None

    stats: Optional[Dict[str, Any]] = None
    if "normalize" not in skip:
        input_norm, stats = normalize(input_shots, mode=norm_mode, per=norm_scope, clip_percentile=clip_p)
        target_norm, _ = normalize(target_shots, mode=norm_mode, per=norm_scope, override_stats=stats)
    else:
        input_norm = input_shots.astype(np.float32, copy=False)
        target_norm = target_shots.astype(np.float32, copy=False)

    patch_trace = int(prep.get("patch_trace", 128))
    patch_time = int(prep.get("patch_time", 256))
    overlap = float(prep.get("patch_overlap", 0.5))

    start = time.time()
    pred_norm = _inference_on_shots_posenc(
        model=model,
        input_shots=input_norm,
        patch_size=(patch_trace, patch_time),
        overlap=overlap,
        device=device,
        batch_size=batch_size,
        position_encoding_range=str(prep.get("position_encoding_range", "minus_one_to_one")),
    )
    elapsed = time.time() - start
    print(f"Inference time: {elapsed:.2f}s")

    metric_names = [m["name"] for m in cfg.get("metrics", [])]
    per_shot, mean = compute_shot_metrics(
        pred_norm,
        target_norm,
        metric_names=metric_names,
        psnr_peak=1.0 if norm_mode in ("max_abs", "minmax") else float(np.max(np.abs(target_norm)) or 1.0),
        ssim_data_range=2.0 if norm_mode == "max_abs" else 1.0,
    )

    if "normalize" not in skip and stats is not None:
        pred_shots = denormalize(pred_norm, stats, mode=norm_mode, per=norm_scope)
        input_out = denormalize(input_norm, stats, mode=norm_mode, per=norm_scope)
        target_out = denormalize(target_norm, stats, mode=norm_mode, per=norm_scope)
    else:
        pred_shots = pred_norm
        input_out = input_norm
        target_out = target_norm

    if output_dir is None:
        exp = cfg.get("experiment", {})
        output_dir = str(Path(exp.get("output_dir", "results")) / exp.get("name", "exp") / f"inference_{split}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metrics_per_shot.csv").open("w", encoding="utf-8") as f:
        names = list(per_shot.keys())
        f.write(",".join(["shot_idx", *names]) + "\n")
        for i in range(pred_norm.shape[0]):
            row = [str(i), *[format_metric_value(k, float(per_shot[k][i])) for k in names]]
            f.write(",".join(row) + "\n")

    summary = dict(mean)
    summary["split"] = split
    summary["n_shots"] = int(pred_norm.shape[0])
    summary["inference_time_seconds"] = round(elapsed, 3)
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_bin:
        bin_path = out_dir / f"pred_{split}_ns{pred_shots.shape[0]}ng{pred_shots.shape[1]}nt{pred_shots.shape[2]}.bin"
        pred_shots.astype(np.float32, copy=False).tofile(bin_path)
        print(f"Saved prediction bin: {bin_path}")

    if save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "input_shots.npy", input_out.astype(np.float32, copy=False))
        np.save(npy_dir / "pred_shots.npy", pred_shots.astype(np.float32, copy=False))
        np.save(npy_dir / "target_shots.npy", target_out.astype(np.float32, copy=False))
        print(f"Saved .npy files to: {npy_dir}")

    viz_dir = out_dir / "visualizations"
    indices = select_random_shots(pred_norm.shape[0], n_viz_shots, seed=seed)
    save_shot_visualizations(
        input_shots=input_out,
        pred_shots=pred_shots,
        target_shots=target_out,
        indices=indices,
        save_dir=viz_dir,
        title_prefix=f"atten_unet_{split}",
    )

    print(f"Inference complete. Outputs saved to: {out_dir}")
    print(f"Visualized shots: {list(indices)}")
    print("Mean metrics (normalized domain):")
    for k, v in mean.items():
        print(f"  {k}: {format_metric_value(k, v)}")


if __name__ == "__main__":
    main()
