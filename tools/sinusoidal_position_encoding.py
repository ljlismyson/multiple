"""Sinusoidal position-channel helpers for patch-based shot-gather training."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def append_sinusoidal_position_channels(
    patches: np.ndarray,
    patch_info: Dict[str, Any],
    *,
    frequencies: Iterable[float] = (1.0,),
) -> np.ndarray:
    """Append sinusoidal absolute trace/time coordinate channels.

    For every frequency ``f``, four channels are appended in this order:
    ``sin(2*pi*f*trace)``, ``cos(2*pi*f*trace)``,
    ``sin(2*pi*f*time)``, and ``cos(2*pi*f*time)``. Coordinates are absolute
    in-shot positions normalized to ``[0, 1]`` before applying sin/cos.
    """
    if patches.ndim != 4:
        raise ValueError(f"patches must have shape (P, C, H, W), got {patches.shape}.")
    if patch_info.get("mode") != "uniform":
        raise ValueError("sinusoidal position channels require patchify_uniform metadata.")

    freqs = tuple(float(v) for v in frequencies)
    if not freqs or any(v <= 0.0 for v in freqs):
        raise ValueError(f"frequencies must be positive, got {freqs!r}.")

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

    pos_channels = []
    for freq in freqs:
        trace_angle = np.float32(2.0 * np.pi * freq) * trace_grid
        time_angle = np.float32(2.0 * np.pi * freq) * time_grid
        trace_sin = np.broadcast_to(
            np.sin(trace_angle)[:, None, :, None],
            (trace_starts.size, time_starts.size, patch_trace, patch_time),
        )
        trace_cos = np.broadcast_to(
            np.cos(trace_angle)[:, None, :, None],
            (trace_starts.size, time_starts.size, patch_trace, patch_time),
        )
        time_sin = np.broadcast_to(
            np.sin(time_angle)[None, :, None, :],
            (trace_starts.size, time_starts.size, patch_trace, patch_time),
        )
        time_cos = np.broadcast_to(
            np.cos(time_angle)[None, :, None, :],
            (trace_starts.size, time_starts.size, patch_trace, patch_time),
        )
        pos_channels.extend([trace_sin, trace_cos, time_sin, time_cos])

    pos_one_shot = np.stack(pos_channels, axis=2).reshape(
        n_per_shot, 4 * len(freqs), patch_trace, patch_time
    )
    pos = np.broadcast_to(
        pos_one_shot[None, ...],
        (n_shots, n_per_shot, 4 * len(freqs), patch_trace, patch_time),
    ).reshape(expected, 4 * len(freqs), patch_trace, patch_time)

    return np.concatenate([patches, pos.astype(patches.dtype, copy=False)], axis=1)
