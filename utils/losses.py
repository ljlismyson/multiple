"""Loss registry + factory with basic supervised reconstruction losses.

See ``utils/README.md`` for the registry workflow (how to add a new loss).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn

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
