# Diffraction Multiple Attenuation with Attention U-Net

This project follows the network architecture and training pipeline from
`D:/edge_download/home/code/v2.3`:

- model: `model/coherent_noise_attenuation/atten_unet.py`
- training loop/helpers: `utils/train_utils.py`
- preprocessing: `tools/preprocessing.py` and `tools/patching.py`

Only the data I/O layer is extended to read the raw `.bin` volumes and the
project-specific 8:1:1 shot split.

## Data Shape

The paired files are:

- `data_free_surface_ns880ng481nt3300.bin`
- `data_sim_abs_ghost_ns880ng481nt3300.bin`

Their shape is:

```text
(ns, ng, nt) = (880, 481, 3300)
```

That is 880 shots, 481 traces per shot gather, and 3300 time samples.

You can verify this without extracting the full zip:

```powershell
conda run -n segy python scripts/check_data_shape.py --zip D:/edge_download/0623_multiples.zip
```

Linux example:

```bash
python scripts/check_data_shape.py --zip /data/bhy/0623_multiples.zip
```

## Create Train/Val/Test Splits

Create the three split folders directly under `data/`:

```powershell
conda run -n segy python scripts/prepare_data_splits.py
```

Linux example:

```bash
python scripts/prepare_data_splits.py
```

This writes:

- `data/train/free_surface_ns704ng481nt3300.bin`
- `data/train/sim_abs_ghost_ns704ng481nt3300.bin`
- `data/val/free_surface_ns88ng481nt3300.bin`
- `data/val/sim_abs_ghost_ns88ng481nt3300.bin`
- `data/test/free_surface_ns88ng481nt3300.bin`
- `data/test/sim_abs_ghost_ns88ng481nt3300.bin`

The split is along the shot axis:

```text
train/val/test = 704/88/88 shots = 8:1:1
```

## Training

The default training config reads `data/train` and `data/val`. `data/test` is
kept held out and is not loaded by the training script.

To train with fewer shots, set `data.train_shots` in the YAML config. The
validation split uses the corresponding 8:1 ratio and rounds to an integer:

```yaml
data:
  train_shots: 200  # uses 200 train shots and 25 val shots
```

`train_shots: null` uses all 704 training shots and all 88 validation shots.

Windows:

```powershell
conda run -n segy python scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet.py --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet.yaml
```

Linux:

```bash
python scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet.py --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet.yaml
```

Multi-GPU Linux:

```bash
torchrun --nproc_per_node=2 scripts/coherent_noise_attenuation/train_diffraction_multiples_atten_unet.py --config configs/coherent_noise_attenuation/diffraction_multiples_atten_unet.yaml
```

## Train/Val/Test Split

The split is a shot-level split saved as `.bin` files:

- `data/train`: 704 shots
- `data/val`: 88 shots
- `data/test`: 88 shots

With the default patch settings:

- `patch_trace: 128`
- `patch_time: 256`
- `patch_overlap: 0.5`

each shot produces 7 trace windows by 25 time windows, so the full dataset yields
175 patches per shot. Therefore:

- training patches: `704 * 175 = 123200`
- validation patches: `88 * 175 = 15400`
- test patches: `88 * 175 = 15400`

During training, only the validation split is evaluated at `eval_interval`.
The test split is reserved for final evaluation and never participates in the
training loop.
