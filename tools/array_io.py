"""Unified volume I/O: .npy, .mat, .bin, and .sgy -> (n_shots, n_traces, n_time) ndarray."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import re
import zipfile

import numpy as np


_BIN_SHAPE_RE = re.compile(r"_ns(?P<ns>\d+)ng(?P<ng>\d+)nt(?P<nt>\d+)\.bin$")


def read_npy_volume(path: Union[str, Path]) -> np.ndarray:
    """Load a .npy file; ensure 3-D output."""
    arr = np.load(str(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"npy volume must be 2-D or 3-D, got {arr.ndim}D.")
    return arr.astype(np.float32, copy=False)


def read_mat_volume(path: Union[str, Path], key: Optional[str] = None) -> np.ndarray:
    """Load a .mat file using scipy.io.loadmat."""
    from scipy.io import loadmat

    data = loadmat(str(path))
    if key is None:
        candidates = [k for k in data.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"No usable variable found in {path}.")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple variables found in {path}: {candidates}. "
                f"Please specify the 'key' to load."
            )
        key = candidates[0]
    arr = data[key]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"mat volume must be 2-D or 3-D, got {arr.ndim}D.")
    return arr.astype(np.float32, copy=False)


def read_sgy_volume(
    path: Union[str, Path],
    traces_per_shot: int,
    time_downsample: int = 1,
) -> np.ndarray:
    """Wrapper around tools.segy_read.read_regular_shots; returns shots only."""
    from tools.segy_read import read_regular_shots

    shots, _ = read_regular_shots(
        path=path,
        traces_per_shot=traces_per_shot,
        time_downsample=time_downsample,
        return_headers=False,
    )
    return shots


def read_bin_volume(
    path: Union[str, Path],
    shape: tuple[int, int, int],
    dtype: str = "float32",
    order: str = "C",
    mmap: bool = False,
) -> np.ndarray:
    """Read raw binary seismic volume and reshape as ``(n_shots, n_traces, n_time)``."""
    p = Path(path)
    np_dtype = np.dtype(dtype)
    expected_items = int(shape[0]) * int(shape[1]) * int(shape[2])
    expected_bytes = expected_items * np_dtype.itemsize
    actual_bytes = p.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Binary size mismatch for {p}: expected {expected_bytes} bytes "
            f"from shape={shape}, dtype={np_dtype}, got {actual_bytes} bytes."
        )
    arr = np.memmap(p, dtype=np_dtype, mode="r", shape=shape, order=order) if mmap else np.fromfile(p, dtype=np_dtype).reshape(shape, order=order)
    return arr.astype(np.float32, copy=False)


def _infer_bin_shape(path: Union[str, Path]) -> tuple[int, int, int]:
    match = _BIN_SHAPE_RE.search(Path(path).name)
    if match is None:
        raise ValueError(
            f"Cannot infer raw .bin shape from {path!r}. "
            "Set shape: [n_shots, n_traces, n_time] in the config."
        )
    return int(match["ns"]), int(match["ng"]), int(match["nt"])


def _resolve_archived_member(data_cfg: Dict[str, Any]) -> Optional[Path]:
    """Extract a configured zip member when ``path`` is missing."""
    path = Path(str(data_cfg["path"]))
    if path.exists():
        return path
    zip_path_raw = data_cfg.get("zip_path")
    member = data_cfg.get("member")
    if not zip_path_raw or not member:
        return None

    zip_path = Path(str(zip_path_raw))
    if not zip_path.exists():
        raise FileNotFoundError(f"zip_path not found: {zip_path}")
    extract_dir = Path(str(data_cfg.get("extract_dir", zip_path.with_suffix(""))))
    out_path = extract_dir / Path(str(member)).name
    if out_path.exists():
        return out_path

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if member not in names:
            matches = [name for name in names if name.endswith(str(member))]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Cannot resolve member {member!r} in {zip_path}; matches={matches}."
                )
            member = matches[0]
        with zf.open(member) as src, out_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    return out_path


def load_volume(data_cfg: Dict[str, Any]) -> np.ndarray:
    """Dispatch to the correct reader based on ``path`` suffix.

    Parameters
    ----------
    data_cfg :
        Dict with ``path`` (required) and format-specific keys.
        For ``.sgy`` / ``.segy``: ``traces_per_shot`` (default 201),
        ``time_downsample`` (default 1).
        For ``.mat``: optional ``key`` (variable name).

    Returns
    -------
    shots :
        ``(n_shots, n_traces, n_time)`` float32 ndarray.
    """
    resolved = _resolve_archived_member(data_cfg)
    path = str(resolved if resolved is not None else data_cfg["path"])
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        return read_npy_volume(path)
    elif suffix == ".mat":
        return read_mat_volume(path, key=data_cfg.get("key"))
    elif suffix in (".sgy", ".segy"):
        return read_sgy_volume(
            path,
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
        )
    elif suffix == ".bin":
        shape = data_cfg.get("shape")
        if shape is None:
            if all(k in data_cfg for k in ("n_shots", "n_traces", "n_time")):
                shape = (
                    int(data_cfg["n_shots"]),
                    int(data_cfg["n_traces"]),
                    int(data_cfg["n_time"]),
                )
            else:
                shape = _infer_bin_shape(path)
        return read_bin_volume(
            path,
            shape=tuple(int(v) for v in shape),
            dtype=str(data_cfg.get("dtype", "float32")),
            order=str(data_cfg.get("order", "C")),
            mmap=bool(data_cfg.get("mmap", False)),
        )
    else:
        raise ValueError(f"Unsupported volume format: {suffix!r} ({path}).")
