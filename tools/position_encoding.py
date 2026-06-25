"""Position-channel helpers for patch-based shot-gather training."""

from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np


PositionRange = Literal["minus_one_to_one", "zero_to_one"]
PositionChannels = Literal["trace", "trace_time"]


def append_linear_position_channels(
    patches: np.ndarray,
    patch_info: Dict[str, Any],
    *,
    value_range: PositionRange = "minus_one_to_one",
) -> np.ndarray:
    """Append absolute trace/time coordinate channels to uniform patches.

    ``patch_info`` must be the metadata returned by ``patchify_uniform`` for
    the same ``patches``. The output channel order is:
    ``[seismic, trace_position, time_position]``.
    """
    if patches.ndim != 4:
        raise ValueError(f"patches must have shape (P, C, H, W), got {patches.shape}.")
    if patch_info.get("mode") != "uniform":
        raise ValueError("linear position channels require patchify_uniform metadata.")

    n_shots, n_traces, n_time = [int(v) for v in patch_info["shape"]]
    patch_trace, patch_time = [int(v) for v in patch_info["patch_size"]]
    trace_starts = np.asarray(patch_info["trace_starts"], dtype=np.int64)
    time_starts = np.asarray(patch_info["time_starts"], dtype=np.int64)

    n_per_shot = int(trace_starts.size * time_starts.size)
    expected = int(n_shots * n_per_shot)
    if patches.shape[0] != expected or patches.shape[-2:] != (patch_trace, patch_time):
        raise ValueError(
            "patches do not match patch_info: "
            f"expected ({expected}, C, {patch_trace}, {patch_time}), got {patches.shape}."
        )

    trace_grid = trace_starts[:, None] + np.arange(patch_trace, dtype=np.float32)[None, :]
    time_grid = time_starts[:, None] + np.arange(patch_time, dtype=np.float32)[None, :]
    trace_grid = trace_grid / max(float(n_traces - 1), 1.0)
    time_grid = time_grid / max(float(n_time - 1), 1.0)

    if value_range == "minus_one_to_one":
        trace_grid = trace_grid * 2.0 - 1.0
        time_grid = time_grid * 2.0 - 1.0
    elif value_range != "zero_to_one":
        raise ValueError(
            "value_range must be 'minus_one_to_one' or 'zero_to_one', "
            f"got {value_range!r}."
        )

    dtype = patches.dtype
    trace_maps = np.broadcast_to(
        trace_grid[:, None, :, None],
        (trace_starts.size, time_starts.size, patch_trace, patch_time),
    )
    time_maps = np.broadcast_to(
        time_grid[None, :, None, :],
        (trace_starts.size, time_starts.size, patch_trace, patch_time),
    )
    pos_one_shot = np.stack([trace_maps, time_maps], axis=2).reshape(
        n_per_shot, 2, patch_trace, patch_time
    )
    pos = np.broadcast_to(
        pos_one_shot[None, ...],
        (n_shots, n_per_shot, 2, patch_trace, patch_time),
    ).reshape(expected, 2, patch_trace, patch_time)

    return np.concatenate([patches, pos.astype(dtype, copy=False)], axis=1)


def append_trace_position_channel(
    patches: np.ndarray,
    patch_info: Dict[str, Any],
    *,
    value_range: PositionRange = "minus_one_to_one",
) -> np.ndarray:
    """Append only the absolute trace coordinate channel to uniform patches.

    The output channel order is ``[seismic, trace_position]``. This is useful
    for per-trace models where the time axis is processed directly by a shared
    1D network, while inter-trace attention still needs absolute lateral
    position.
    """
    if patches.ndim != 4:
        raise ValueError(f"patches must have shape (P, C, H, W), got {patches.shape}.")
    if patch_info.get("mode") != "uniform":
        raise ValueError("trace position channels require patchify_uniform metadata.")

    n_shots, n_traces, _ = [int(v) for v in patch_info["shape"]]
    patch_trace, patch_time = [int(v) for v in patch_info["patch_size"]]
    trace_starts = np.asarray(patch_info["trace_starts"], dtype=np.int64)
    time_starts = np.asarray(patch_info["time_starts"], dtype=np.int64)

    n_per_shot = int(trace_starts.size * time_starts.size)
    expected = int(n_shots * n_per_shot)
    if patches.shape[0] != expected or patches.shape[-2:] != (patch_trace, patch_time):
        raise ValueError(
            "patches do not match patch_info: "
            f"expected ({expected}, C, {patch_trace}, {patch_time}), got {patches.shape}."
        )

    trace_grid = trace_starts[:, None] + np.arange(patch_trace, dtype=np.float32)[None, :]
    trace_grid = trace_grid / max(float(n_traces - 1), 1.0)

    if value_range == "minus_one_to_one":
        trace_grid = trace_grid * 2.0 - 1.0
    elif value_range != "zero_to_one":
        raise ValueError(
            "value_range must be 'minus_one_to_one' or 'zero_to_one', "
            f"got {value_range!r}."
        )

    trace_maps = np.broadcast_to(
        trace_grid[:, None, :, None],
        (trace_starts.size, time_starts.size, patch_trace, patch_time),
    )
    pos_one_shot = trace_maps.reshape(n_per_shot, 1, patch_trace, patch_time)
    pos = np.broadcast_to(
        pos_one_shot[None, ...],
        (n_shots, n_per_shot, 1, patch_trace, patch_time),
    ).reshape(expected, 1, patch_trace, patch_time)

    return np.concatenate([patches, pos.astype(patches.dtype, copy=False)], axis=1)


def append_configured_position_channels(
    patches: np.ndarray,
    patch_info: Dict[str, Any],
    *,
    channels: PositionChannels = "trace_time",
    value_range: PositionRange = "minus_one_to_one",
) -> np.ndarray:
    """Append position channels selected by config."""
    if channels == "trace":
        return append_trace_position_channel(patches, patch_info, value_range=value_range)
    if channels == "trace_time":
        return append_linear_position_channels(patches, patch_info, value_range=value_range)
    raise ValueError("channels must be 'trace' or 'trace_time', got " f"{channels!r}.")
