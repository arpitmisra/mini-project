# Fine-tune DINOv2 on OCT, then save dense representations

**Repository:** https://github.com/arpitmisra/mini-project

The default workflow is now **self-supervised fine-tuning of the DINOv2 backbone on OCT training images → freeze the adapted EMA teacher → extract and cache dense features**. There is no binary classifier, disease-label loss, prediction stage, accuracy report or classifier evaluation in the current pipeline.

Start with the [ordered Mac runbook](docs/MAC_RUNBOOK.md). The [repository guide](docs/REPOSITORY_GUIDE.md) explains every component and saved output. The [fine-tuning method](docs/FINETUNING.md) specifies the objective and its limitations.

## One command after installation

On an Apple Silicon Mac, with the environment activated:

```bash
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path "/path/to/OCT2017"
```

This runs a **bounded smoke experiment**: up to 4 images per original class per split (48 selected images total), one fine-tuning epoch capped at two optimizer steps using training images only, then dense extraction for all selected splits. It does not download the dataset. The pretrained weights may download once (~89 MB). The small run verifies software behavior; it is not sufficient domain adaptation for a research result.

The full experiment, after Mac tests and the smoke check pass:

```bash
python run_pipeline.py --config configs/default.yaml --dataset-path "/path/to/OCT2017"
```

The full default uses five epochs over the selected training set (unlimited selection), batch size 2 and learning rate 1e-5. These are starting hyperparameters, not validated optimal settings. Fine-tuning is substantially more expensive than frozen inference. Keep the smoke configuration for initial checks.

## Install on the Mac

Use native arm64 Python, preferably Python 3.12, on Apple Silicon. The project requires macOS 14 or newer; installed MLX wheels can impose a newer minimum.

```bash
git clone --recurse-submodules https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform; print(platform.machine())"
# Must print arm64.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_pipeline.py --stage system
```

For an existing clone, use `git pull --ff-only`, activate its environment, and reinstall requirements. Older YAML files with a `training:` classifier section must be replaced by the new `finetuning:` configuration.

## What fine-tuning means here

The verified pretrained DINOv2 ViT-S/14 initializes a student and teacher. All student backbone parameters are unfrozen. Two crop geometries are generated per training image. Each teacher/student crop pair shares its geometry but uses different mild brightness/contrast augmentation; student views may also be blurred and have patch tokens masked. No vertical flips or rotations are used.

The loss combines **cross-view global self-distillation** with **same-view masked-patch self-distillation**. Teacher targets are centered, sharpened and stop-gradient; teacher parameters are updated by exponential moving average (EMA) of the student. Temporary nonlinear projection heads produce self-supervised prototype distributions. They are not disease classifiers and are discarded from the exported backbone.

This is a DINO/iBOT-style adaptation of pretrained DINOv2, **not an exact reproduction of Meta's full DINOv2 training recipe**. It does not include local multicrop, KoLeo or distributed training. Diagnosis labels are retained as provenance only and never passed to the loss. Only original TRAIN records enter optimization; validation/test images are used only for later extraction. The final epoch's EMA teacher is exported, without test-based or validation-based checkpoint selection. See [the full method and references](docs/FINETUNING.md).

## Dense output and how to inspect it

For each 224 × 224 preprocessed image, the adapted backbone returns:

| Array | Shape per image | Meaning |
|---|---|---|
| CLS | `[384]` | Auxiliary global representation |
| Patches | `[256,384]` | Final normalized patch descriptors |
| Spatial grid | `[16,16,384]` | The same patch descriptors in row-major spatial order |

The patch grid covers the resized **center crop**, not every original image pixel. It is a feature tensor, not a lesion segmentation or CAM map. No representation quality or medical interpretation is guaranteed by successful optimization.

After the smoke command finishes:

```bash
python scripts/export_dense.py --feature-manifest outputs/smoke-adapted/features_manifest.json --split test --output outputs/smoke-adapted/dense_sample.npz
python - <<'PY_ARRAYS'
import numpy as np
with np.load("outputs/smoke-adapted/dense_sample.npz", allow_pickle=False) as scan:
    print("Image:", scan["sample_id"])
    print("Patch matrix:", scan["patches"].shape)
    print(scan["patches"])
    print("Dense grid:", scan["spatial_grid"].shape)
    print(scan["spatial_grid"])
    print("Descriptor at row 7, column 11:", scan["spatial_grid"][7, 11])
PY_ARRAYS
```

The exporter writes a new `.npz` and `.json` pair; use a different output filename on repeat exports. NumPy abbreviates large printed arrays, but every value is saved. Add `--sample-id` with an exact ID from the split index to select a different scan. Export and inspection work on Windows with files copied from the Mac. For a full run, use `outputs/oct-adapted` instead of `outputs/smoke-adapted`.

## Stages

| Stage | Behavior |
|---|---|
| `all` (default) | Fine-tune or reuse the matching completed adapted model, then extract dense representations |
| `finetune` | Fine-tune only; save the adapted EMA backbone and resumable epoch states |
| `extract` | Extract using an existing adapted checkpoint; never silently use the original pretrained model |
| `inspect` | Bounded offline image/preprocessing check, without MLX |
| `system` | Report platform and package versions |

```bash
# Run separately when needed.
python run_pipeline.py --config configs/smoke_test.yaml --stage finetune --dataset-path "/path/to/OCT2017"
python run_pipeline.py --config configs/smoke_test.yaml --stage extract --dataset-path "/path/to/OCT2017" --offline
```

By default extraction reads `<output-dir>/finetuning/backbone.safetensors`. Use `--checkpoint "/path/to/backbone.safetensors"` with `--stage extract` to use another completed adapted run. Its matching `.json` sidecar is required. `--checkpoint` is not a pretrained HF folder and is not accepted for `all`/`finetune`.

The former `train`, `evaluate` and `predict` stages have been removed. Old classifier files in an existing local output directory are not modified or used. New default output directories avoid mixing old classification outputs with adaptation runs.

## Local data and KaggleHub fallback

Path precedence: `--dataset-path` → `OCT_DATASET_PATH` → YAML `dataset.local_path`. Configured relative dataset/model/cache/output paths resolve against the repository root. Command-line config/checkpoint paths resolve from the shell's working directory.

```bash
export OCT_DATASET_PATH="/path/to/OCT2017"
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline
```

The loader expects train/val/test with NORMAL, CNV, DME and DRUSEN folders; nested roots are supported. Existing malformed local data causes an error instead of an automatic download. Missing local data falls back to `kagglehub.dataset_download("paultimothymooney/kermany2018")` only when enabled by the full configuration. KaggleHub downloads/extracts files locally and reuses its cache; this is not streaming. If authentication is requested, use `python -c "import kagglehub; kagglehub.login()"`. Never commit credentials.

If no OCT data is available, an explicitly artificial plumbing check is available:

```bash
python scripts/create_synthetic_smoke.py
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path outputs/synthetic-smoke-data --output-dir outputs/artificial-adaptation
```

These are not OCT scans. The synthetic source marker is retained in run/export metadata. Do not describe the resulting features or losses as OCT research results.

## Checkpoints, resume and cache reuse

- `outputs/smoke-adapted/finetuning/backbone.safetensors` is the **fine-tuned EMA backbone**, without the self-supervised heads. Its JSON sidecar records the base checkpoint, method, train-data fingerprint, settings and adapted weights' SHA-256.
- `finetuning/epoch-XXXX/state.safetensors` retains student, teacher, SSL heads, optimizer and centers. The matching JSON stores epoch, steps, history and NumPy RNG state. `latest.json` points to the last completed epoch. Interrupted work resumes from that epoch; an incomplete epoch is replayed.
- Rerunning `all` with identical data/settings reuses a completed adapted model without fine-tuning again. Changing settings, seed or selected training records requires a new `--output-dir`; the pipeline refuses to overwrite an incompatible run.
- Dense cache identities include the **adapted weights' checksum**, adaptation contract, base model, preprocessing/Pillow version, storage settings and sample identity. Features from different adapted backbones cannot share an identity. Old pretrained-only feature caches are not reused.
- Valid feature shards skip inference/image decoding. Corrupt shards are regenerated during extraction. Cache checks and dataset hashing still read files; cache reuse is not zero I/O.
- Copy the complete `outputs/` layout to preserve relative manifest/index paths. A standalone dense export needs no model or original images. Extraction requires the adapted checkpoint; it does not need the original HF weights when the adapted checkpoint is present.

Default float16 patch persistence rounds float32 inference values and uses about **15.5 GiB** for 84,484 images, plus metadata and CLS. Set `features.patch_dtype: float32` before extraction to retain inference precision (about 30.9 GiB). Fine-tuning epoch states and optimizer data require additional disk space, and all completed epoch states are retained. Dataset download/extraction needs additional space. Dense mode requires `features.save_patches: true`.

## Verification

```bash
python -m pytest -q
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
```

Ordinary Mac tests include native gradients, EMA updates, stop-gradient targets, masked loss, optimizer/RNG resume and checkpoint checks. Opt-in tests load real pretrained weights and exercise tiny fine-tuning → adapted extraction → export → reuse. No full dataset is downloaded by tests.

The optional base-model parity tool compares the **original pretrained** MLX backbone to the identical HF/PyTorch checkpoint; it does not compare fine-tuned weights to the original model:

```bash
python -m pip install -e '.[parity]'
python scripts/verify_parity.py --image "/path/to/an/OCT/scan.jpeg"
```

The adapted backbone is separately checked by save/reload agreement for CLS and patches before its completion metadata is written. See [validation records](docs/VALIDATION.md). **Native fine-tuning and representation quality have not been verified on the Windows development host.** Portable tests and source/API review do not substitute for the Mac tests. No real fine-tuning loss or embedding values are claimed from Windows.

On Windows, use `python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline` and the portable tests. No initial smoke test processes the entire dataset.

## Troubleshooting and scientific limits

| Problem | Action |
|---|---|
| Missing adapted checkpoint | Run `all`/`finetune` first, or provide the adapted `.safetensors` and `.json` pair with `--checkpoint`. |
| Different fine-tuning data/settings | Use a new `--output-dir`; changing epochs also changes the run contract. |
| Old YAML has `training:` | Use the new supplied config with `finetuning:`. |
| Apple Silicon/Metal unavailable | Use native arm64 Python and a compatible Mac/MLX installation. |
| Memory exhaustion | Reduce `finetuning.batch_size` (fine-tuning) or `features.batch_size` (extraction); they are separate settings. |
| Stale writer lock | Verify no writer is active before removing only the reported `.writer.lock`. |
| Nonfinite loss or zero backbone gradient | Stop and investigate data, learning rate and native dependencies; do not treat the run as successful adaptation. |

Train/val/test overlap checks reject exact duplicate bytes in the selected population. They do not establish patient independence. Optional `dataset.validation_fraction` reserves an image-level part of train; it is not used for loss or checkpoint selection. Dense extraction does not establish segmentation/localization quality. Monitor the self-supervised history for instability or collapse, and evaluate representations independently before making scientific claims.
