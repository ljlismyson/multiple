"""Train per-trace U-Net with inter-trace attention.

Windows:
    conda run -n segy python scripts/coherent_noise_attenuation/train_diffraction_multiples_trace_attention_unet.py \
        --config configs/coherent_noise_attenuation/diffraction_multiples_trace_attention_unet.yaml
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_training_module():
    base_path = Path(__file__).with_name("train_diffraction_multiples_atten_unet_posenc.py")
    spec = importlib.util.spec_from_file_location("_trace_attention_base_train", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base training script: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = __file__
    return module


if __name__ == "__main__":
    _load_base_training_module().main()
