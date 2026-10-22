# Mac runbook: installation, tests and programs in order

Run commands in Terminal from the cloned repository root, in the order below. Stop and resolve a failed check before starting the full experiment. This runbook is for an Apple Silicon Mac with native arm64 Python (Python 3.12 recommended). The project requires macOS 14 or newer; an installed MLX wheel can impose a newer minimum. Intel Macs and Rosetta/x86_64 Python do not support this pipeline's execution path.

## 1. Clone and install

Ensure Git and native Python are installed, then:

```bash
git clone --recurse-submodules https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform, sys; print(sys.version); print(platform.machine())"
```

The architecture must print `arm64`. If a native Python 3.12 installation is available as `python3.12` instead, use that command to create the environment below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_pipeline.py --stage system
```

Expected: no broken requirements, Darwin/arm64 reported, MLX and MLX-VLM versions shown. The system command reports the environment; the next step actually exercises Metal. Installation downloads Python packages, but does not download OCT data or pretrained model weights.

Already cloned? Start from that directory, run `git pull --ff-only`, activate the environment, and reinstall requirements if dependencies changed.

## 2. Run ordinary tests, including native MLX tests

```bash
python -m pytest -q
```

Expected: no failures. On a correctly configured Mac the native gradient, frozen-model shape, weight-conversion and checkpoint tests execute. Two opt-in pretrained tests are skipped at this step. Do not expect the Windows result of `49 passed, 1 skipped`: Windows skips the entire native module. These ordinary tests use artificial fixtures and require no dataset or pretrained weight downloads.

## 3. Run the real pretrained integration tests

```bash
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
```

Expected: the two pretrained tests pass. They check actual model loading, CLS/patch output shapes and batch consistency, a tiny artificial-image training/evaluation run, dense export and cache reuse. The pinned weights may download once (~89 MB). These tests do not download the OCT dataset; artificial-image metrics are not research results.

## 4. Compare MLX outputs against the reference

```bash
python -m pip install -e '.[parity]'
python -m pip check
python scripts/verify_parity.py
```

Without `--image`, this uses deterministic artificial pixels. With a real OCT image available, also run:

```bash
python scripts/verify_parity.py --image "/absolute/path/to/an/OCT/scan.jpeg"
```

Expected: both `cls` and `patches` report `passed: true`. The report is `outputs/parity.json`; later runs replace it unless a different `--output` is supplied. The comparison requires `atol=5e-4, rtol=1e-3`. Investigate a failure rather than ignoring or loosening the tolerance. PyTorch is used only for this optional reference comparison, not normal MLX extraction/training.

## 5. Select real local OCT data and inspect 48 images

Replace the example with the existing extracted dataset directory on this Mac. Keep quotes around paths containing spaces.

```bash
export OCT_DATASET_PATH="/absolute/path/to/OCT2017"
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline
```

The dataset should have train/val/test directories containing NORMAL, CNV, DME and DRUSEN. Nested OCT2017 roots are supported. Expected: local source reported; 16 images decoded per split, 48 total; no network or model execution. Filename indexing can count the larger directory, but only selected images are hashed/decoded. The report is `outputs/smoke/portable_smoke.json`.

**If no OCT dataset is available yet**, steps 2–4 still work. For an additional artificial end-to-end check, run:

```bash
python scripts/create_synthetic_smoke.py
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path outputs/synthetic-smoke-data --output-dir outputs/artificial-smoke
python scripts/export_dense.py --feature-manifest outputs/artificial-smoke/features_manifest.json --output outputs/artificial-smoke/dense_sample.npz
```

This is a separate artificial run, not an OCT result. The fixture generator and exporter refuse to overwrite their existing outputs. Reuse an existing fixture or choose a new export filename on repeat runs. To continue steps 6–9 on real data, supply a real dataset in `OCT_DATASET_PATH`. Alternatively, after the initial software checks, step 10 provides the explicit full KaggleHub download route.

## 6. Extract dense features for the small real-data selection

```bash
python run_pipeline.py --config configs/smoke_test.yaml --stage extract
```

Expected: up to 48 scans extracted, with both CLS and spatial features saved. The first run may download the model if it is not already cached. No dataset download is permitted. This stage does not train a classifier.

## 7. Export and inspect one scan's complete dense representation

```bash
python scripts/export_dense.py --feature-manifest outputs/smoke/features_manifest.json --split test --output outputs/smoke/dense_sample.npz
python - <<'PY'
import numpy as np
with np.load("outputs/smoke/dense_sample.npz", allow_pickle=False) as scan:
    print("Sample:", scan["sample_id"])
    print("CLS:", scan["cls"].shape)
    print("Patches:", scan["patches"].shape)
    print("Grid:", scan["spatial_grid"].shape)
    print("First eight values at row 7, column 11:", scan["spatial_grid"][7, 11, :8])
PY
```

Expected shapes: `(384,)`, `(256, 384)`, `(16, 16, 384)`. Outputs are `dense_sample.npz` and `dense_sample.json`; the latter records provenance and coordinate conventions. Choose a new output filename if rerunning the export. Add `--sample-id` with an exact ID from the split index to choose a different image. The grid describes the preprocessed center crop, not a lesion mask.

## 8. Confirm cache reuse

```bash
python run_pipeline.py --config configs/smoke_test.yaml --stage extract --offline
```

For an unchanged 48-image selection/configuration and intact caches, expected output is `extracted 0, reused 16` for each split. The pipeline rechecks sample identity and cache integrity, but does not run the backbone or decode the images on valid cache hits.

## 9. Optionally train, evaluate and predict

These commands reuse the smoke features and do not re-extract them:

```bash
python run_pipeline.py --config configs/smoke_test.yaml --stage train --feature-manifest outputs/smoke/features_manifest.json
python run_pipeline.py --config configs/smoke_test.yaml --stage evaluate --feature-manifest outputs/smoke/features_manifest.json --checkpoint outputs/smoke/checkpoints/best_classifier.safetensors
python run_pipeline.py --config configs/smoke_test.yaml --stage predict --checkpoint outputs/smoke/checkpoints/best_classifier.safetensors --image "/absolute/path/to/an/OCT/scan.jpeg"
```

Training saves the best classifier and history; evaluation saves metrics, predictions and plots in `outputs/smoke/metrics/`. Prediction prints the class and probability and runs the backbone on the supplied image. A 48-image smoke result is a software check, not a meaningful performance estimate.

As an alternative to running extraction/training/evaluation separately, `python run_pipeline.py --config configs/smoke_test.yaml` executes all three stages. Do not run it again merely to evaluate an existing classifier: use `--stage evaluate` for that.

## 10. Start the full real-data experiment only after the small checks pass

For an existing real dataset, ensure the environment variable points to it:

```bash
export OCT_DATASET_PATH="/absolute/path/to/real/OCT2017"
python run_pipeline.py --config configs/default.yaml --stage extract
```

For automatic KaggleHub retrieval instead, unset a previous path override:

```bash
unset OCT_DATASET_PATH
python run_pipeline.py --config configs/default.yaml --stage extract
```

The second route uses local `data/OCT2017` if available; otherwise it downloads/extracts Kermany data through KaggleHub. This is intentionally a full-data operation. If authentication is requested, follow the README's KaggleHub login instructions.

Allow about 15.5 GiB for float16 patch arrays at the indexed dataset size, plus other outputs and the dataset itself. A download also needs archive/extraction space. Float32 patch persistence doubles patch storage; configure it before extraction if required.

The full dense representation is already the main deliverable. If classification is also wanted:

```bash
python run_pipeline.py --config configs/default.yaml --stage train --feature-manifest outputs/default/features_manifest.json --output-dir outputs/probe-01
python run_pipeline.py --config configs/default.yaml --stage evaluate --feature-manifest outputs/default/features_manifest.json --checkpoint outputs/probe-01/checkpoints/best_classifier.safetensors --output-dir outputs/probe-01
python scripts/export_dense.py --feature-manifest outputs/default/features_manifest.json --output outputs/default/dense_sample.npz
python -m pip freeze > outputs/requirements-used.txt
```

Preserve the parity report, resolved configurations, manifests, feature shards, checkpoint metadata, dependency snapshot and evaluation results. Select models using validation, then perform the final test evaluation; do not tune experiments against test scores.

## 11. Return to the project later

```bash
cd /path/to/mini-project
source .venv/bin/activate
export OCT_DATASET_PATH="/absolute/path/to/OCT2017"
```

Use the appropriate stage above. Completed extraction shards are reusable after interruption with unchanged settings. If a stale writer lock is reported, first verify no extraction process is still running before removing only that lock. See the README's troubleshooting table for platform, memory, dependency and cache-identity errors.
