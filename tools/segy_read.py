"""SEG-Y reader: flat trace table -> shot gathers, with optional time downsampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import segyio
except ImportError as exc:  # pragma: no cover - surfaced to the user at runtime
    raise ImportError(
        "segyio is required. Install it with `pip install segyio`."
    ) from exc


# Default per-shot headers kept alongside the trace tensor.
_DEFAULT_HEADER_KEYS: Tuple[str, ...] = (
    "FieldRecord",  # FFID, shot identifier
    "SourceX",
    "SourceY",
    "GroupX",
    "GroupY",
)


# ----------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------

def inspect_segy(path: Union[str, Path]) -> Dict[str, Any]:
    """Probe SEG-Y metadata for shot-gather planning.

    Returns
    -------
    dict with keys ``n_traces`` / ``n_samples`` / ``sample_interval_us``
    / ``unique_ffid_count`` / ``traces_per_ffid_min_max``.
    """
    path = Path(path)
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = int(f.tracecount)
        n_samples = int(len(f.samples))
        sample_interval_us = int(f.bin[segyio.BinField.Interval])
        ffid = np.asarray(f.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64)

    unique, counts = np.unique(ffid, return_counts=True)
    return {
        "n_traces": n_traces,
        "n_samples": n_samples,
        "sample_interval_us": sample_interval_us,
        "unique_ffid_count": int(unique.size),
        "traces_per_ffid_min_max": (int(counts.min()), int(counts.max())),
    }


# ----------------------------------------------------------------------
# Regular data (default path)
# ----------------------------------------------------------------------

def read_regular_shots(
    path: Union[str, Path],
    traces_per_shot: int,
    time_downsample: int = 1,
    dtype: np.dtype = np.float32,
    return_headers: bool = True,
    verify_ffid: bool = True,
    header_keys: Tuple[str, ...] = _DEFAULT_HEADER_KEYS,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Read a regular SEG-Y (every shot has the same trace count).

    Parameters
    ----------
    traces_per_shot : run :func:`inspect_segy` if unknown.
    time_downsample : 1 = off; > 1 applies anti-aliased ``scipy.signal.decimate``.
    verify_ffid     : raise if any shot does not share one ``FieldRecord`` value.

    Returns
    -------
    traces  : ``(n_shots, traces_per_shot, time_length)``,
              ``n_shots = n_traces // traces_per_shot``.
    headers : ``dict[str, (n_shots, traces_per_shot)]`` or ``None``.
    """
    path = Path(path)
    if traces_per_shot <= 0:
        raise ValueError(f"traces_per_shot must be positive, got {traces_per_shot}.")
    if time_downsample < 1:
        raise ValueError(f"time_downsample must be >= 1, got {time_downsample}.")

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = int(f.tracecount)
        n_samples = int(len(f.samples))

        if n_traces % traces_per_shot != 0:
            raise ValueError(
                f"File {path.name} has {n_traces} traces, not divisible by "
                f"traces_per_shot={traces_per_shot}. The file is likely "
                f"irregular; use read_irregular_shots_by_header() instead."
            )
        n_shots = n_traces // traces_per_shot

        traces_flat = segyio.tools.collect(f.trace[:]).astype(dtype, copy=False)
        traces = traces_flat.reshape(n_shots, traces_per_shot, n_samples)

        if time_downsample > 1:
            from scipy.signal import decimate  # lazy: only needed here
            traces = decimate(
                traces, q=time_downsample, axis=-1, zero_phase=True,
            ).astype(dtype, copy=False)

        headers: Optional[Dict[str, np.ndarray]] = None
        if return_headers or verify_ffid:
            headers_flat: Dict[str, np.ndarray] = {}
            keys_to_read = set(header_keys) | ({"FieldRecord"} if verify_ffid else set())
            for name in keys_to_read:
                field = getattr(segyio.TraceField, name)
                headers_flat[name] = np.asarray(f.attributes(field)[:], dtype=np.int64)

        if verify_ffid:
            ffid_2d = headers_flat["FieldRecord"].reshape(n_shots, traces_per_shot)
            per_shot_unique = np.array([np.unique(row).size for row in ffid_2d])
            if np.any(per_shot_unique != 1):
                raise ValueError(
                    "Regularity check failed: some shots contain more than one "
                    "FieldRecord value. Use read_irregular_shots_by_header()."
                )

        if return_headers:
            headers = {
                name: headers_flat[name].reshape(n_shots, traces_per_shot)
                for name in header_keys
            }

    return traces, headers


# ----------------------------------------------------------------------
# Irregular data (placeholder)
# ----------------------------------------------------------------------

def read_irregular_shots_by_header(
    path: Union[str, Path],
    header_key: str = "FieldRecord",
    time_downsample: int = 1,
    dtype: np.dtype = np.float32,
    return_headers: bool = True,
    header_keys: Tuple[str, ...] = _DEFAULT_HEADER_KEYS,
) -> Tuple[List[np.ndarray], Optional[List[Dict[str, np.ndarray]]]]:
    """Read a SEG-Y with variable trace count per shot. NOT IMPLEMENTED YET.

    Returns
    -------
    traces  : ``List[(n_traces_i, time_length)]``, length = ``n_shots``.
    headers : ``List[dict[str, (n_traces_i,)]]`` or ``None``.
    """
    raise NotImplementedError(
        "read_irregular_shots_by_header is a placeholder; implement when the "
        "first SEG_C3NA_ffid_*.sgy file is consumed."
    )


# ----------------------------------------------------------------------
# CLI smoke test (read-only; safe to run manually)
# ----------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover - manual inspection helper
    """Read-only smoke test against the regular demo file. Run with ``python -m tools.segy_read``."""
    path = Path("/data/liuqi/code/MAE/5d-transformer/data/SEGC3-45/SEG_45Shot_shots1-9.sgy")
    if not path.exists():
        print(f"[demo] File not found: {path}")
        return

    info = inspect_segy(path)
    print(f"[demo] inspect_segy -> {info}")

    traces_min, traces_max = info["traces_per_ffid_min_max"]
    if traces_min != traces_max:
        print(
            "[demo] File appears irregular (per-FFID trace count varies); "
            "read_regular_shots would fail. Use read_irregular_shots_by_header."
        )
        return
    traces_per_shot = traces_min
    print(f"[demo] inferred traces_per_shot = {traces_per_shot}")

    traces, headers = read_regular_shots(
        path, traces_per_shot=201, time_downsample=1, return_headers=True,
    )
    print(f"[demo] no downsample  -> shape = {traces.shape}, dtype = {traces.dtype}")

    traces_d2, _ = read_regular_shots(
        path, traces_per_shot=traces_per_shot, time_downsample=2, return_headers=False,
    )
    print(f"[demo] downsample = 2 -> shape = {traces_d2.shape}, dtype = {traces_d2.dtype}")

    if headers is not None:
        for name, arr in headers.items():
            print(f"[demo] headers['{name}'].shape = {arr.shape}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
