# Detailed use (the README is the short starting point)

## What changed

The default `oct.py`/`oct.sh` flow now uses frozen pretrained DINOv2 through PyTorch/Transformers.
The optional MLX adaptation now supports full and query/value LoRA training through `oct.py finetune`.
The default baseline does not train. No segmentation decoder or weak-label training is implemented yet.
See [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) for attention dashboards and normalization experiments.

Pinned baseline engines: torch 2.9.1, transformers 4.57.6. The original model revision and weight SHA-256
are checked before strict loading. Arbitrary remote code and random-weight fallback are not used.
Other dependencies have version bounds; capture `pip freeze` for the environment used in a paper.
The Mac wrapper creates a fresh dedicated environment separate from the earlier MLX setup.

## Everyday commands on Mac

```bash
bash oct.sh setup
bash oct.sh test
bash oct.sh run "/absolute/path/to/OCT2017"
bash oct.sh show
```

Only setup is installation. The test uses two artificial images and real pretrained weights.
The run selects at most 12 real scans without downloading data. Full mode selects all images:

```bash
bash oct.sh run "/absolute/path/to/OCT2017" --full
bash oct.sh show --full
```

The CPU baseline is intentionally the default so a Metal setup is not required.
`--device mps` explicitly selects Mac acceleration; `--device cuda` requires a CUDA-capable torch installation.
Test each device first. There is no silent switch between devices. We do not claim identical bits across devices.

Repeat a run with `--offline` after the first download to prohibit both dataset and model network access.
Cache reuse skips model construction entirely when all shards are valid; it still verifies data/cache hashes.
Changes to model, device, preprocessing, package identity or storage affect feature identity.
A changed identity in an existing run folder requires a new `--output-dir`.

Both run/show support `--output-dir outputs/my-experiment`. Use the same directory with both.
Paths resolve against the repository root. The shell wrapper changes into that root, even when launched elsewhere.
The input path overrides `OCT_DATASET_PATH`, which overrides YAML.
Full mode falls back to KaggleHub when the chosen local path is missing; a malformed existing directory raises an error.

## Windows

```powershell
python -m venv .venv-dense
.venv-dense\Scripts\python.exe -m pip install -r requirements.txt
.venv-dense\Scripts\python.exe oct.py test
.venv-dense\Scripts\python.exe oct.py run "C:\path\to\OCT2017"
.venv-dense\Scripts\python.exe oct.py show
```

Use the same `--full`, `--offline` and `--output-dir` options as on Mac.

## Understanding and selecting arrays

```bash
bash oct.sh show --all-values
bash oct.sh show --split train
bash oct.sh show --sample-id "EXACT_ID_FROM_THE_SPLIT_INDEX"
```

The viewer prints the split-index path. Copy a sample ID from its records list.
The default test scan is `dense_sample.npz/.json`. Other split defaults are `dense_sample_train` and `dense_sample_val`.
Explicit IDs produce safely hashed filenames under the run folder's `samples/`.
Existing exports are reused only when their image metadata, feature identity and checksum match.
Move a corrupt/stale export pair aside before repeating show; do not delete the feature cache.

```python
import numpy as np
with np.load("outputs/pretrained-smoke/dense_sample.npz", allow_pickle=False) as scan:
    features = scan["patches"]          # 256 x 384
    grid = scan["spatial_grid"]         # 16 x 16 x 384
    descriptor = grid[7, 11]            # same as features[123]
    print(features.shape, descriptor)
```

Other export arrays: `cls`, `patch_row_col`, `patch_bounds_xyxy`, `sample_id`, `original_class`, `binary_label`.
Labels are provenance only: NORMAL=0 and other diagnoses=1 for the retained binary metadata.
This does not define pixel-level lesion classes.

RGB conversion → shortest-edge resize 256 → center crop 224 → ImageNet normalization → DINOv2.
Each patch descriptor has an input anchor of 14×14 crop pixels but incorporates global context through attention.
Token order is row-major. The coordinate bounds are crop-relative, upper bound exclusive.
Center cropping may remove relevant anatomy; the baseline is not a full-image segmentation method.
For future segmentation, benchmark larger inputs, tiling or geometry-preserving preprocessing with explicit coordinate transforms.
Do not compare methods with different crop coverage as though only the adapter changed.

## Files written by a baseline run

Run folder: `outputs/pretrained-smoke` or `outputs/pretrained` (or your chosen directory).

| File | Meaning |
|---|---|
| `features_manifest.json` | Links to each split's cache index and model/preprocessing identity |
| `config.resolved.json` | Effective configuration; retained finetuning section is unused in the baseline |
| `dataset_summary.json` | Local/Kaggle source, original splits, selected counts and integrity scope |
| `experiment_metadata.json` | Model identity, software/environment, configuration and protocol |
| `extraction_benchmark.json` | Extracted/reused counts, elapsed extraction/I/O time and cache size |
| `dense_sample.npz/.json` | Single-scan export created by show |
| `samples/<key>.npz/.json` | Explicitly chosen scans created by show |

Shared cache: `outputs/features/<identity>/<split>/`.
Each `index-<selection>.json` lists source paths/IDs, class metadata, checksums and shard mappings.
Each shard `.npz` contains `cls [N,384]`, `patches [N,256,384]`, labels, IDs and original classes.
A matching shard JSON holds its checksum and provenance. Default patches are float16 (rounded from float32 inference).
The spatial grid is a reshape; it is not separately duplicated in the main cache.

Copy the **entire outputs layout** to preserve relative links, or share a single export pair for NumPy-only access.
Datasets, model weights, caches, virtual environments and error logs are excluded from Git.

## Testing and troubleshooting

`bash oct.sh test` checks real pretrained inference, batch-versus-single agreement and array save/reload.
It requires the first model download but no OCT dataset.
For all portable tests and the real extraction/export/reuse tests:

```bash
OCT_TEST_BASELINE=1 .venv-dense/bin/python -m pytest -q
.venv-dense/bin/python -m pip freeze > outputs/requirements-used.txt
```

The default tests skip optional MLX and download-dependent baseline tests.
GitHub's real-baseline workflow installs the documented setup on macOS, Windows and Linux,
runs real pretrained inference and the end-to-end tests, and uploads the self-test report.
Read the workflow conclusion rather than assuming a runner has passed.

Error logs contain paths and tracebacks; inspect them before sharing publicly.
For low memory, reduce `features.batch_size` in the selected configuration.
For a model SHA mismatch, restore the exact pinned checkpoint rather than disabling integrity checks.
For an interrupted extraction, rerun the same command; valid shards are reused.
Investigate a writer lock before removing it, and never run two writers into the same feature directory.

## Optional adaptation research

The previous MLX work is documented in [FINETUNING.md](FINETUNING.md).
Use a **separate** environment with `requirements-mlx.txt`.
Set `OCT_TEST_MLX=1` to enable its native tests and `OCT_TEST_PRETRAINED=1` for its pretrained tests.
Its command remains `python run_pipeline.py --config configs/smoke_test.yaml --dataset-path "/path/to/OCT2017"`.
Its results are in `outputs/smoke-adapted/` (full: `outputs/oct-adapted/`).
To view an existing adapted run from the baseline environment, use
`bash oct.sh show --output-dir outputs/smoke-adapted`.
This earlier experiment has not been established to improve OCT segmentation.
