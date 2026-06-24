"""Diffraction-multiple attenuation with linear patch-position channels.

Windows:
    conda run -n segy python scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet_posenc.py \
        --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet_posenc.yaml
Linux/DDP:
    torchrun --nproc_per_node=2 scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet_posenc.py \
        --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet_posenc.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Windows conda environments can load duplicate OpenMP runtimes when torch and
# matplotlib/scipy coexist. Keep the original training logic unchanged and only
# apply the known runtime workaround when the user has not set it explicitly.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# Bootstrap repo root into sys.path BEFORE importing utils/model. Walks up from
# this file looking for a directory that contains both ``model/`` and ``utils/``.
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(
        "Cannot find repo root (a directory containing both ``model/`` and ``utils/``)."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.coherent_noise_attenuation import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import patchify_uniform  # noqa: E402
from tools.position_encoding import append_linear_position_channels  # noqa: E402
from tools.preprocessing import normalize  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    apply_denoise_experiment_name_from_model,
    build_loss,
    build_metrics,
    build_optimizer,
    build_scheduler,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    init_distributed,
    load_config,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    train_one_epoch,
    training_device,
    visualize_random_sample,
)


def _load_pair_volume(pair_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    input_cfg = dict(pair_cfg)
    input_cfg["path"] = pair_cfg["input_path"]
    if "input_member" in pair_cfg:
        input_cfg["member"] = pair_cfg["input_member"]
    target_cfg = dict(pair_cfg)
    target_cfg["path"] = pair_cfg["target_path"]
    if "target_member" in pair_cfg:
        target_cfg["member"] = pair_cfg["target_member"]

    input_shots = load_volume(input_cfg)
    target_shots = load_volume(target_cfg)
    if input_shots.shape != target_shots.shape:
        raise ValueError(
            "Paired volume shape mismatch: "
            f"input {input_shots.shape} vs target {target_shots.shape}."
        )
    return input_shots, target_shots


def _build_denoise_patch_pairs_from_pair(
    pair_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    *,
    max_shots: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack paired shots into input + label patch batches.

    Parameters
    ----------
    pair_cfg :
        ``{segy,npy,mat,bin}_pair`` supplies ``input_path``, ``target_path``,
        and format-specific options. ``preprocess`` uses patch sizes,
        ``normalize_*``, ``max_shots``, and an optional ``skip`` list.
        Statistics are always computed on the **noisy input**; the noise label volume is
        scaled with ``normalize(..., override_stats=...)`` so both share the
        noisy-derived scale (including ``clip_threshold`` when percentile clipping is
        used). Requires ``normalize_scope: global``. Spherical-divergence compensation
        is not applied in this pipeline.

    Returns
    -------
    tuple
        ``(input_patches, target_patches)`` float32 arrays shaped
        ``(P, 3, H, W)`` and ``(P, 1, H, W)``. Input channels are seismic,
        absolute trace coordinate, and absolute time coordinate.
    """
    input_shots, target_shots = _load_pair_volume(pair_cfg)
    if max_shots is None and prep.get("max_shots") is not None:
        max_shots = int(prep["max_shots"])
    if max_shots is not None:
        m = int(max_shots)
        if m <= 0:
            raise ValueError(f"max_shots must be positive, got {m}.")
        input_shots = input_shots[:m]
        target_shots = target_shots[:m]

    skip = set(prep.get("skip", []))

    # Denoise pipeline: no spherical-divergence compensation (raw paired volumes).

    if "normalize" not in skip:
        mode = str(prep.get("normalize_mode", "max_abs"))
        per = str(prep.get("normalize_scope", "global"))
        clip_raw = prep.get("clip_percentile")
        clip_p = float(clip_raw) if clip_raw is not None else None

        mode_keys = {
            "minmax": ("min", "max"),
            "max_abs": ("max_abs",),
            "mean_std": ("mean", "std"),
        }
        if mode not in mode_keys:
            raise ValueError(
                f"Unknown normalize_mode {mode!r} for paired denoise pipeline."
            )

        input_shots, in_stats = normalize(
            input_shots, mode=mode, per=per, clip_percentile=clip_p
        )
        target_shots, _ = normalize(
            target_shots,
            mode=mode,
            per=per,
            override_stats=in_stats,
        )

    patch_t = int(prep.get("patch_time", 256))
    patch_x = int(prep.get("patch_trace", 128))
    overlap = float(prep.get("patch_overlap", 0.5))

    target_patches, _ = patchify_uniform(
        target_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    input_patches, input_info = patchify_uniform(
        input_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    pos_range = str(prep.get("position_encoding_range", "minus_one_to_one"))
    input_patches = append_linear_position_channels(
        input_patches,
        input_info,
        value_range=pos_range,  # type: ignore[arg-type]
    )
    return input_patches.astype(np.float32), target_patches.astype(np.float32)


def _configured_ns(pair_cfg: Dict[str, Any]) -> Optional[int]:
    shape = pair_cfg.get("shape")
    if shape is None:
        if "n_shots" in pair_cfg:
            return int(pair_cfg["n_shots"])
        return None
    return int(shape[0])


def _resolve_train_val_shot_limits(data_cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Return train/val shot limits; val follows the configured train:val ratio."""
    raw_train_shots = data_cfg.get("train_shots")
    if raw_train_shots is None:
        return None, None

    train_total = _configured_ns(data_cfg["train_pair"])
    val_total = _configured_ns(data_cfg["val_pair"])
    requested_train = int(raw_train_shots)
    if requested_train <= 0:
        raise ValueError(f"data.train_shots must be positive, got {requested_train}.")
    if train_total is not None:
        requested_train = min(requested_train, train_total)

    if train_total is None or val_total is None:
        val_limit = max(1, int(math.floor(requested_train / 8.0 + 0.5)))
    else:
        val_limit = int(math.floor(requested_train * (val_total / train_total) + 0.5))
        val_limit = min(max(1, val_limit), val_total)
    return requested_train, val_limit


def _build_train_val_loaders(
    cfg: Dict[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DataLoader]]:
    """Build loaders from pre-split train/val volumes; test_pair is never loaded here."""
    data_cfg = cfg["data"]
    if "train_pair" not in data_cfg or "val_pair" not in data_cfg:
        raise ValueError("Config must define data.train_pair and data.val_pair.")

    prep = cfg["preprocess"]
    train_shots, val_shots = _resolve_train_val_shot_limits(data_cfg)
    x_train_np, y_train_np = _build_denoise_patch_pairs_from_pair(
        data_cfg["train_pair"],
        prep,
        max_shots=train_shots,
    )
    x_val_np, y_val_np = _build_denoise_patch_pairs_from_pair(
        data_cfg["val_pair"],
        prep,
        max_shots=val_shots,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np))
    val_ds = TensorDataset(torch.from_numpy(x_val_np), torch.from_numpy(y_val_np))

    loader_cfg = data_cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg["experiment"]["seed"]),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if distributed and rank != 0:
        eval_train_loader: Optional[DataLoader] = None
    else:
        eval_train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, train_sampler, eval_train_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train denoising from paired volumes (sgy/npy/mat). "
            "Default config path matches this script name (configs/<name>.yaml). "
            "Multi-GPU: torchrun --nproc_per_node=N ..."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_relpath_for_train_script(__file__),
        help="Path to denoise config (expects data.*_pair).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_denoise_experiment_name_from_model(cfg)
    cfg["metrics"] = resolve_denoise_metrics(cfg)

    distributed, rank, local_rank, world_size = init_distributed()

    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    train_loader, val_loader, train_sampler, eval_train_loader = _build_train_val_loaders(
        cfg,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )
    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(
        model,
        distributed=distributed,
        device=device,
        local_rank=local_rank,
    )
    model_type = str(cfg["model"]["type"])
    loss_fn = build_loss(cfg["loss"]).to(device)
    metrics = build_metrics(cfg["metrics"])
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
    if logger is not None:
        logger.info(
            f"Model {model_type} | train/val patches: {len(train_loader.dataset)} / {len(val_loader.dataset)}"
        )

    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))

    start_time = time.time()
    for epoch in range(total_epochs):
        sampler_set_epoch(train_sampler, epoch)
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            grad_clip=cfg["train"].get("grad_clip"),
            log_interval=int(cfg["train"].get("log_interval", 20)),
            logger=logger if log_step else None,
        )
        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate(
                model=model,
                loader=eval_train_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                metrics_on_denoised_signal=False,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device,
                    metrics_on_denoised_signal=False,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train": train_stats["train"],
                    "val": val_losses.get("val", float("nan")),
                },
                metrics=metric_row,
                extras={"lr": optimizer.param_groups[0]["lr"]},
            )

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extras={"config": cfg},
            )

        if rank == 0 and (epoch + 1) % vis_interval == 0:
            visualize_random_sample(
                model=model,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"Denoise {model_type} epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(f"Denoise {model_type} training finished in {elapsed:.2f}s ({elapsed/60:.2f} min).")
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
