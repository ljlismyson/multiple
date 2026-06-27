"""Linear-position Attention U-Net training with residual/signal/detail loss.

Windows:
    conda run -n segy python scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet_posenc_detail_loss.py
Linux/DDP:
    torchrun --nproc_per_node=1 scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet_posenc_detail_loss.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root (a directory containing both `model/` and `utils/`).")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.coherent_noise_attenuation import build_model  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    apply_denoise_experiment_name_from_model,
    build_loss,
    build_metrics,
    build_optimizer,
    build_scheduler,
    compute_metrics,
    destroy_distributed,
    init_distributed,
    load_config,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    visualize_random_sample,
)

_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "coherent_noise_attenuation"
    / "diffraction_multiples_atten_unet_posenc_detail_loss.yaml"
)
_BASE_SCRIPT = Path(__file__).with_name("train_diffraction_multiples_atten_unet_posenc.py")


def _load_base_training_module():
    spec = importlib.util.spec_from_file_location("_posenc_detail_loss_base_train", _BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base training script: {_BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train_one_epoch_detail(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[_LRScheduler] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for step, batch in enumerate(loader):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("_train_one_epoch_detail expects loader batch as (input, target).")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        # Detail loss needs the seismic input channel to form the denoised signal.
        loss = loss_fn(pred, y, input=x)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
        if logger is not None and (step + 1) % max(1, int(log_interval)) == 0:
            logger.info(
                f"[epoch={epoch} step={step + 1}/{len(loader)}] "
                f"train_step_loss={loss.detach().item():.6g}"
            )

    if scheduler is not None:
        scheduler.step()
    mean_loss = total_loss / max(n_batches, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            stat = torch.tensor([float(total_loss), float(n_batches)], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            mean_loss = float(stat[0].item() / max(stat[1].item(), 1.0))
    return {"train": float(mean_loss)}


@torch.no_grad()
def _evaluate_detail(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    metrics: Optional[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("_evaluate_detail expects loader batch as (input, target).")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y, input=x)
        total_loss += float(loss.detach().item())
        n_batches += 1

        if metrics:
            x_signal = x[:, : pred.shape[1], ...]
            pred_m = x_signal - pred
            targ_m = x_signal - y
            batch_metrics = compute_metrics(metrics, pred_m, targ_m)
            for k, v in batch_metrics.items():
                metric_sums[k] += float(v)

    denom = max(n_batches, 1)
    return {"val": float(total_loss / denom)}, {k: float(v / denom) for k, v in metric_sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear-position Attention U-Net with residual/signal/detail loss."
    )
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG), help="Path to detail-loss config.")
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

    base_train = _load_base_training_module()
    train_loader, val_loader, train_sampler, eval_train_loader = base_train._build_train_val_loaders(
        cfg,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )

    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(model, distributed=distributed, device=device, local_rank=local_rank)
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
        train_stats = _train_one_epoch_detail(
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
            _, train_metrics = _evaluate_detail(
                model=model,
                loader=eval_train_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = _evaluate_detail(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={"train": train_stats["train"], "val": val_losses.get("val", float("nan"))},
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
                title=f"Denoise {model_type} detail-loss epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(f"Denoise {model_type} detail-loss training finished in {elapsed:.2f}s ({elapsed/60:.2f} min).")
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
