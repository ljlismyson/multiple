"""Run inference for per-trace U-Net with inter-trace attention.

Example:
    conda run -n segy python scripts/coherent_noise_attenuation/inference_diffraction_multiples_trace_attention_unet.py \
        --checkpoint outputs/260624v2/diffraction_multiples_trace_attention_unet/checkpoints/epoch_0199.pt
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_inference_module():
    base_path = Path(__file__).with_name("inference_diffraction_multiples_atten_unet_posenc.py")
    spec = importlib.util.spec_from_file_location("_trace_attention_base_inference", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base inference script: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._DEFAULT_CONFIG = (
        "configs/coherent_noise_attenuation/diffraction_multiples_trace_attention_unet.yaml"
    )
    return module


if __name__ == "__main__":
    _load_base_inference_module().main()
