"""Training-diagnostic plots: per-sample 4-panel + loss / metrics curves.

Every helper saves to disk and returns the Matplotlib ``Figure``. Matplotlib
is imported lazily inside each function to keep import cost low.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_to_2d(x: ArrayLike, name: str) -> np.ndarray:
    """Squeeze ``(B, C, H, W)`` / ``(B, H, W)`` / ``(C, H, W)`` / ``(H, W)`` to ``(H, W)``.

    Always picks batch index 0 and channel index 0 when those leading axes are
    present. Raises ``ValueError`` for any other shape.
    """
    arr = np.asarray(_to_numpy(x))
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        # Heuristic: small leading dim => channel; otherwise batch of 2D maps.
        if arr.shape[0] <= 4:
            arr = arr[0]
        else:
            arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(
            f"{name}: expected ndim in (2, 3, 4), got shape {arr.shape}."
        )
    return arr


def _symmetric_clip(arr: np.ndarray, q: float = 0.99) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    v = float(np.quantile(np.abs(finite), q))
    return v if v > 0 else float(np.max(np.abs(finite))) or 1.0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_sample(
    input_data: ArrayLike,
    prediction: ArrayLike,
    target: Optional[ArrayLike],
    save_path: Union[str, Path],
    title: Optional[str] = None,
    cmap: str = "gray",
) -> Any:
    """Side-by-side ``input | prediction | target | residual`` panel for one sample.

    Parameters
    ----------
    input_data, prediction : 2D / ``(C, H, W)`` / ``(B, C, H, W)``; auto-squeezed
                             to a 2D map (batch idx 0 + channel idx 0).
    target                 : ground truth; ``None`` -> only ``input | prediction``
                             is plotted (no residual).
    save_path              : output PNG path (parents auto-created).
    title                  : optional global figure title.
    cmap                   : Matplotlib colormap name; default ``"gray"``.

    Display convention
    ------------------
    Inputs are interpreted as ``(trace, time)``. Plots are shown as
    ``x=trace`` (horizontal), ``y=time`` (vertical), so each panel is rendered
    with a transpose before ``imshow``.

    Returns
    -------
    matplotlib.figure.Figure (closed before return; reopen with ``plt.figure(...)``).
    """
    import matplotlib.pyplot as plt  # lazy

    inp_2d = _squeeze_to_2d(input_data, "plot_sample.input_data")
    pred_2d = _squeeze_to_2d(prediction, "plot_sample.prediction")
    if target is not None:
        tgt_2d = _squeeze_to_2d(target, "plot_sample.target")
        res_2d = pred_2d - tgt_2d
        panels = [
            ("input", inp_2d),
            ("prediction", pred_2d),
            ("target", tgt_2d),
            ("residual (pred - target)", res_2d),
        ]
    else:
        panels = [("input", inp_2d), ("prediction", pred_2d)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, arr) in zip(axes, panels):
        # Seismic display convention: horizontal=trace, vertical=time.
        arr_show = arr.T
        v = _symmetric_clip(arr_show)
        im = ax.imshow(arr_show, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("trace")
        ax.set_ylabel("time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    out = Path(save_path)
    _ensure_parent(out)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


def visualize_random_sample(
    model: torch.nn.Module,
    loader: Any,
    save_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    title: Optional[str] = None,
    seed: Optional[int] = None,
    cmap: str = "gray",
) -> Any:
    """Pick a random sample from ``loader.dataset``, run the model, save a 4-panel plot.

    Parameters
    ----------
    model     : ``nn.Module``; switched to ``eval()`` + ``no_grad`` for the
                forward pass and restored to its original training state on exit.
    loader    : a ``DataLoader`` (only ``loader.dataset`` is used so shuffling
                does not matter); ``dataset[idx]`` must return
                ``(input_tensor, target_tensor_or_none)``.
    save_path : output PNG path (parents auto-created).
    device    : device the model lives on; the picked sample is moved here.
    title     : optional title prefix; the resolved ``"sample idx=<i>"`` is
                always appended so the chosen sample is reproducible.
    seed      : ``None`` -> draw a fresh sample each call; ``int`` -> deterministic.
    cmap      : forwarded to :func:`plot_sample`.

    Returns
    -------
    matplotlib.figure.Figure produced by :func:`plot_sample`.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__") or len(dataset) == 0:
        raise ValueError(
            "visualize_random_sample: loader.dataset must support __len__ and be non-empty."
        )

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(dataset)))
    sample = dataset[idx]
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        input_tensor, target_tensor = sample
    else:
        input_tensor, target_tensor = sample, None

    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.as_tensor(input_tensor)
    input_batch = input_tensor.unsqueeze(0).to(device)
    target_batch: Optional[torch.Tensor] = None
    if target_tensor is not None:
        if not isinstance(target_tensor, torch.Tensor):
            target_tensor = torch.as_tensor(target_tensor)
        target_batch = target_tensor.unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(input_batch)
    finally:
        if was_training:
            model.train()

    suffix = f"sample idx={idx}"
    full_title = f"{title} | {suffix}" if title else suffix
    return plot_sample(
        input_data=input_batch,
        prediction=prediction,
        target=target_batch,
        save_path=save_path,
        title=full_title,
        cmap=cmap,
    )


def _is_all_nan(values: Sequence[float]) -> bool:
    arr = np.asarray(list(values), dtype=float)
    return arr.size == 0 or bool(np.all(~np.isfinite(arr)))


def plot_loss_curve(
    history: Dict[str, Sequence[float]],
    save_path: Union[str, Path],
    title: str = "Loss",
    log_y: bool = False,
) -> Any:
    """Plot ``{split: [...]}`` loss curves.

    Parameters
    ----------
    history   : e.g. ``{"train": [...], "val": [...]}``; all-NaN series are skipped.
    save_path : output PNG path.
    title     : figure title; default ``"Loss"``.
    log_y     : if ``True``, use symmetric-log Y axis (handles zero / small negatives).
    """
    import matplotlib.pyplot as plt  # lazy

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for name, values in history.items():
        if _is_all_nan(values):
            continue
        ax.plot(range(len(values)), values, label=name, linewidth=1.5)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    if log_y:
        ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")

    out = Path(save_path)
    _ensure_parent(out)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_single_metric_curve(
    history: Dict[str, Sequence[float]],
    save_path: Union[str, Path],
    title: str = "Metric",
) -> Any:
    """Plot a single metric curve (one line per series in ``history``).

    Parameters
    ----------
    history   : e.g. ``{"train_snr": [...], "test_snr": [...]}``; all-NaN series skipped.
    save_path : output PNG path.
    title     : figure title; default ``"Metric"``.
    """
    import matplotlib.pyplot as plt  # lazy

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for name, values in history.items():
        if _is_all_nan(values):
            continue
        ax.plot(range(len(values)), values, label=name, linewidth=1.5)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("epoch")
    ax.set_ylabel(title.lower())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")

    out = Path(save_path)
    _ensure_parent(out)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
