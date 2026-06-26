"""Attention U-Net training with weighted MSE+L1 residual loss.

Windows:
    conda run -n segy python scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet_mse_l1.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root (a directory containing both `model/` and `utils/`).")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "coherent_noise_attenuation"
    / "diffraction_multiples_atten_unet_mse_l1.yaml"
)
_BASE_SCRIPT = Path(__file__).with_name("train_diffraction_multiples_atten_unet.py")


def _has_config_arg() -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:])


def _load_base_main():
    spec = importlib.util.spec_from_file_location("_atten_unet_mse_l1_base_train", _BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base training script: {_BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    if not _has_config_arg():
        sys.argv.extend(["--config", str(_DEFAULT_CONFIG)])
    _load_base_main()()


if __name__ == "__main__":
    main()
