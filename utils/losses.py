"""Loss registry + factory with basic supervised reconstruction losses.

See ``utils/README.md`` for the registry workflow (how to add a new loss).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY: Dict[str, Type["BaseLoss"]] = {}


def register_loss(name: str) -> Callable[[Type["BaseLoss"]], Type["BaseLoss"]]:
    """Class decorator that registers a loss under ``name``."""

    def _decorator(cls: Type["BaseLoss"]) -> Type["BaseLoss"]:
        if name in LOSS_REGISTRY:
            raise KeyError(f"Loss '{name}' already registered.")
        LOSS_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseLoss(nn.Module):
    """Common loss interface: ``forward(pred, target=None, **extras) -> Tensor``; ``extras`` passes optional mask / weight."""

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        raise NotImplementedError


@register_loss("mse")
class MSELoss(BaseLoss):
    """Mean squared error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MSELoss requires `target`.")
        return nn.functional.mse_loss(pred, target, reduction=self.reduction)


@register_loss("l1")
class L1Loss(BaseLoss):
    """Mean absolute error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("L1Loss requires `target`.")
        return nn.functional.l1_loss(pred, target, reduction=self.reduction)


@register_loss("mse_l1")
class MSEL1Loss(BaseLoss):
    """Weighted MSE + L1 loss for residual prediction."""

    def __init__(
        self,
        mse_weight: float = 0.3,
        l1_weight: float = 0.7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MSEL1Loss requires `target`.")
        mse = nn.functional.mse_loss(pred, target, reduction=self.reduction)
        l1 = nn.functional.l1_loss(pred, target, reduction=self.reduction)
        return self.mse_weight * mse + self.l1_weight * l1


@register_loss("residual_signal_detail")
class ResidualSignalDetailLoss(BaseLoss):
    """Residual supervision plus denoised-signal detail preservation."""

    def __init__(
        self,
        residual_l1_weight: float = 0.35,
        signal_l1_weight: float = 0.35,
        gradient_weight: float = 0.20,
        highfreq_weight: float = 0.10,
        highfreq_kernel_size: int = 5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction != "mean":
            raise ValueError("ResidualSignalDetailLoss currently supports reduction='mean' only.")
        if highfreq_kernel_size < 3 or highfreq_kernel_size % 2 == 0:
            raise ValueError("highfreq_kernel_size must be an odd integer >= 3.")
        self.residual_l1_weight = float(residual_l1_weight)
        self.signal_l1_weight = float(signal_l1_weight)
        self.gradient_weight = float(gradient_weight)
        self.highfreq_weight = float(highfreq_weight)
        self.highfreq_kernel_size = int(highfreq_kernel_size)
        self.reduction = reduction

    @staticmethod
    def _gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
        target_dx = target[..., 1:, :] - target[..., :-1, :]
        pred_dt = pred[..., :, 1:] - pred[..., :, :-1]
        target_dt = target[..., :, 1:] - target[..., :, :-1]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dt, target_dt)

    def _highfreq(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.highfreq_kernel_size // 2
        smooth = F.avg_pool2d(
            x,
            kernel_size=self.highfreq_kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        return x - smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("ResidualSignalDetailLoss requires `target`.")
        input_tensor = extras.get("input")
        if input_tensor is None:
            raise ValueError("ResidualSignalDetailLoss requires extras['input'].")
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.as_tensor(input_tensor, device=pred.device, dtype=pred.dtype)
        input_tensor = input_tensor.to(device=pred.device, dtype=pred.dtype)
        seismic_input = input_tensor[:, : pred.shape[1], ...]

        denoised_pred = seismic_input - pred
        clean_target = seismic_input - target

        residual_l1 = F.l1_loss(pred, target)
        signal_l1 = F.l1_loss(denoised_pred, clean_target)
        gradient_l1 = self._gradient_l1(denoised_pred, clean_target)
        highfreq_l1 = F.l1_loss(self._highfreq(denoised_pred), self._highfreq(clean_target))

        return (
            self.residual_l1_weight * residual_l1
            + self.signal_l1_weight * signal_l1
            + self.gradient_weight * gradient_l1
            + self.highfreq_weight * highfreq_l1
        )


@register_loss("weighted_mse")
class WeightedMSELoss(BaseLoss):
    """MSE weighted by ``extras["weight"]``."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("WeightedMSELoss requires `target`.")
        weight = extras.get("weight")
        if weight is None:
            raise ValueError("WeightedMSELoss requires extras['weight'].")
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, device=pred.device, dtype=pred.dtype)
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        err2 = (pred - target).pow(2)
        weighted = weight * err2
        denom = weight.sum().clamp_min(self.eps)
        return weighted.sum() / denom


def build_loss(cfg: Dict[str, Any]) -> BaseLoss:
    """Instantiate a loss from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**cfg.get("params", {}))
