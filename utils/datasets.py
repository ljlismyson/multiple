"""Dataset registry + DataLoader factory.

See ``utils/README.md`` for the registry workflow (how to add a new dataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATASET_REGISTRY: Dict[str, Type["BaseArrayDataset"]] = {}


def register_dataset(name: str) -> Callable[[Type["BaseArrayDataset"]], Type["BaseArrayDataset"]]:
    """Class decorator that registers a dataset under ``name``."""

    def _decorator(cls: Type["BaseArrayDataset"]) -> Type["BaseArrayDataset"]:
        if name in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{name}' already registered.")
        DATASET_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseArrayDataset(Dataset):
    """Abstract base class for array-style datasets (npy / mat / npz / ...).

    Parameters
    ----------
    root       : directory or single file path containing the samples.
    input_key  : key used to look up the input inside ``.npz`` / ``.mat`` containers.
    target_key : key for supervised targets; ``None`` for self-supervised tasks.
    transforms : list of callables applied sequentially to each loaded input.

    ``__getitem__`` returns ``(input_tensor, target_tensor_or_none)``, both CPU tensors.
    """

    def __init__(
        self,
        root: str,
        input_key: str = "data",
        target_key: Optional[str] = None,
        transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.input_key = input_key
        self.target_key = target_key
        self.transforms = transforms or []
        self._index: List[Path] = []  # TODO: populate in _build_index()
        self._build_index()

    def _build_index(self) -> None:
        """Scan ``self.root`` and populate ``self._index`` with sample paths."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return one ``(input, target)`` sample; subclasses implement ``_load_sample``."""
        x, y = self._load_sample(self._index[idx])
        for t in self.transforms:
            x = t(x)
        return x, y

    def _load_sample(
        self, path: Path
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


@register_dataset("npy")
class NpyDataset(BaseArrayDataset):
    """Dataset backed by ``.npy`` / ``.npz`` files."""

    def _build_index(self) -> None:
        if self.root.is_file():
            if self.root.suffix.lower() not in (".npy", ".npz"):
                raise ValueError(f"Unsupported file type: {self.root}")
            self._index = [self.root]
            return
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        files = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in (".npy", ".npz")
        )
        if not files:
            raise FileNotFoundError(f"No .npy/.npz files found under {self.root}")
        self._index = files

    def _load_sample(
        self, path: Path
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            x_np = np.asarray(arr)
            y_np = None
        else:
            with np.load(path, allow_pickle=False) as data:
                if self.input_key not in data:
                    raise KeyError(
                        f"{path.name}: missing input_key={self.input_key!r}. "
                        f"Available keys: {list(data.keys())}."
                    )
                x_np = np.asarray(data[self.input_key])
                y_np = (
                    np.asarray(data[self.target_key])
                    if self.target_key is not None and self.target_key in data
                    else None
                )
        x = torch.from_numpy(np.asarray(x_np)).float()
        y = torch.from_numpy(np.asarray(y_np)).float() if y_np is not None else None
        return x, y


@register_dataset("mat")
class MatDataset(BaseArrayDataset):
    """Dataset backed by MATLAB ``.mat`` files."""

    def _build_index(self) -> None:
        if self.root.is_file():
            if self.root.suffix.lower() != ".mat":
                raise ValueError(f"Unsupported file type: {self.root}")
            self._index = [self.root]
            return
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        files = sorted(self.root.rglob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No .mat files found under {self.root}")
        self._index = files

    def _load_sample(
        self, path: Path
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required for MatDataset (`pip install scipy`).") from exc
        data = loadmat(path)
        if self.input_key not in data:
            raise KeyError(
                f"{path.name}: missing input_key={self.input_key!r}. "
                f"Available keys: {list(data.keys())}."
            )
        x_np = np.asarray(data[self.input_key])
        y_np = (
            np.asarray(data[self.target_key])
            if self.target_key is not None and self.target_key in data
            else None
        )
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float() if y_np is not None else None
        return x, y


def build_dataset(cfg: Dict[str, Any]) -> BaseArrayDataset:
    """Instantiate a dataset from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](**cfg.get("params", {}))


def build_dataloader(split_cfg: Dict[str, Any]) -> DataLoader:
    """Build a DataLoader from a ``{type, params, loader}`` split config."""
    dataset = build_dataset(split_cfg)
    loader_cfg = split_cfg.get("loader", {})
    return DataLoader(dataset, **loader_cfg)
