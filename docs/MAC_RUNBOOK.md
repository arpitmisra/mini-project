# Mac runbook: fine-tuning followed by dense representations only

Run Terminal commands from the repository root. Use an Apple Silicon Mac with native arm64 Python, preferably 3.12. The project requires macOS 14 or newer; installed MLX wheels can impose a newer minimum. Stop on failed tests before starting a full experiment.

## 1. Clone and install once

```bash
git clone --recurse-submodules https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform; print(platform.machine())"
# Confirm arm64.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_pipeline.py --stage system
```

For an existing clone, use `git pull --ff-only`, activate the environment, and reinstall requirements. Native model stages check the platform before dataset/model retrieval.

## 2. Check native execution

```bash
python -m pytest -q
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
```

Ordinary tests execute native synthetic gradient/loss/EMA/checkpoint/resume checks on the Mac, with two pretrained tests skipped. The opt-in command runs those two tests using actual pinned weights (~89 MB first download), including tiny backbone fine-tuning, dense export and reuse. No OCT dataset download is performed by these tests.

Optionally verify the original pretrained adapter against the reference:

```bash
python -m pip install -e '.[parity]'
python scripts/verify_parity.py
```

Both CLS and patches should report `passed: true`; results go to `outputs/parity.json`. This compares the original initialization, not an adapted model to different original weights. Add `--image "/path/to/scan.jpeg"` for an actual OCT-image comparison.

## 3. Inspect local data without training or downloading

```bash
export OCT_DATASET_PATH="/absolute/path/to/OCT2017"
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline
```

Expected: 16 selected images per split, 48 total. This checks only the selected images' hashes/pixels. The full directory can be counted without decoding all images. Dataset folders must include train/val/test and the four original diagnoses.

## 4. Run the requested workflow with one command

```bash
python run_pipeline.py --config configs/smoke_test.yaml
```

This fine-tunes on original TRAIN images only (one epoch capped at two steps), freezes the final EMA teacher, then extracts dense features for the 48 selected images. No classifier or classification metrics are produced. No dataset download is allowed. Model weights can be fetched once if needed.

Outputs:

- `outputs/smoke-adapted/finetuning/backbone.safetensors` and `.json`: adapted EMA backbone, with no SSL projection heads.
- `outputs/smoke-adapted/finetuning/epoch-0001/`: resumable training state.
- `outputs/smoke-adapted/finetuning/history.json`: self-supervised training diagnostics.
- `outputs/smoke-adapted/features_manifest.json`: links to the split feature indices.
- `outputs/features/<adapted-identity>/`: CLS/patch shards and sample metadata.

If no OCT data is available, test with explicitly artificial images instead:

```bash
python scripts/create_synthetic_smoke.py
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path outputs/synthetic-smoke-data --output-dir outputs/artificial-adaptation
```

Do not treat this artificial run as OCT research. The generator refuses to overwrite an existing fixture; reuse it if already created. Subsequent real-data commands must use the real dataset, not this fixture.

## 5. Export and display the dense arrays

```bash
python scripts/export_dense.py --feature-manifest outputs/smoke-adapted/features_manifest.json --split test --output outputs/smoke-adapted/dense_sample.npz
python - <<'PY_ARRAYS'
import numpy as np
with np.load("outputs/smoke-adapted/dense_sample.npz", allow_pickle=False) as scan:
    print("Image:", scan["sample_id"])
    print("Patches:", scan["patches"].shape)
    print(scan["patches"])
    print("Spatial grid:", scan["spatial_grid"].shape)
    print(scan["spatial_grid"])
    print("One full descriptor:", scan["spatial_grid"][7, 11])
PY_ARRAYS
```

Expected shapes: patches `(256,384)`, grid `(16,16,384)`; CLS `(384,)` is also saved. Large printed arrays are abbreviated by NumPy; the file retains every value. The spatial coordinates describe the preprocessed center crop. Use a new export filename when repeating the command. Add `--sample-id` with an exact ID from the split index for another image. For the artificial run, use `outputs/artificial-adaptation` in these paths.

## 6. Verify reuse or run stages separately

Rerun the same full command offline:

```bash
python run_pipeline.py --config configs/smoke_test.yaml --offline
```

Expected: a message that the completed adapted backbone is reused, followed by `extracted 0, reused 16` for each split. No repeated optimization or feature inference is needed. File integrity checks still run.

To run only one part:

```bash
python run_pipeline.py --config configs/smoke_test.yaml --stage finetune
python run_pipeline.py --config configs/smoke_test.yaml --stage extract --offline
```

Extraction requires the adapted checkpoint from a completed fine-tuning run. For a different completed model, use `--stage extract --checkpoint "/path/to/backbone.safetensors"`; keep the matching JSON beside it. The pipeline does not fall back silently to pretrained-only extraction.

## 7. Run the full real dataset after the checks pass

```bash
export OCT_DATASET_PATH="/absolute/path/to/real/OCT2017"
python run_pipeline.py --config configs/default.yaml
```

This performs five fine-tuning epochs over training images and then extracts all splits. Full outputs use `outputs/oct-adapted/`. Configure epochs, batch size and learning rate before starting; five epochs is an unvalidated starting setting.

If the dataset is not present and automatic KaggleHub retrieval is desired:

```bash
unset OCT_DATASET_PATH
python run_pipeline.py --config configs/default.yaml
```

The command uses local `data/OCT2017` if available, otherwise downloads/extracts the full Kermany dataset. It is intentionally not a smoke command. Follow the README's authentication instructions if requested. Allow dataset/archive space, about 15.5 GiB for float16 patch arrays at the indexed size, and additional space for epoch/optimizer checkpoints. Float32 patches double patch storage.

After the full run, replace `outputs/smoke-adapted` with `outputs/oct-adapted` in step 5 to inspect its features. Save the environment:

```bash
python -m pip freeze > outputs/requirements-used.txt
```

## 8. Resume or change experiments

After interruption, rerun the same command/config/output directory. Fine-tuning resumes after the last fully saved epoch; an incomplete epoch is repeated. Once adaptation completes, feature extraction resumes from valid completed shards. If a stale lock is reported, verify no writer is active before removing only that lock.

Changing training settings, seed, epochs or selected training records requires a new output directory:

```bash
python run_pipeline.py --config configs/default.yaml --output-dir outputs/oct-adapted-experiment-02
```

In a new Terminal session, return to the repository, run `source .venv/bin/activate`, and set the real `OCT_DATASET_PATH` again. The former classifier commands (`train`, `evaluate`, `predict`) no longer exist in the main pipeline.

Native fine-tuning and actual representation quality still require verification on the Mac. A low self-supervised training loss does not establish good OCT representations or clinical performance.
