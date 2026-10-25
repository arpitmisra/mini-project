# OCT dense representations with fine-tuned DINOv2

**For the professor: install once, run the small check, then view the arrays.**

This project adapts pretrained **DINOv2 ViT-S/14** to OCT training images using self-supervised teacher–student learning, then saves dense patch representations. The current pipeline has **no binary classifier, disease-label training loss, classification predictions, or accuracy/AUC evaluation**.

After the one-time installation below, the two everyday commands are:

```bash
python oct.py run "/absolute/path/to/OCT2017"
python oct.py show
```

Replace the quoted path with the actual extracted dataset folder. The first command performs a **small smoke check**; the second displays one scan's saved arrays. To run the full experiment later, add `--full` to **both** commands. All commands below are run in Terminal from the cloned `mini-project` directory.

## 1. Install once on the Mac

Use an **Apple Silicon Mac (M1/M2/M3/M4 or later), native arm64 Python, and macOS compatible with MLX**. The project requires macOS 14 or newer; the installed MLX wheel may require a newer version. Python 3.12 is the documented setup target. Intel Macs and Python running under Rosetta cannot run this native MLX pipeline.

First, clone and enter the repository:

```bash
git clone https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform; print(platform.machine())"
```

The last command must print `arm64`. If it prints `x86_64`, switch to native arm64 Python before continuing. If `git` or `python3` is missing, install those tools first; on macOS, `xcode-select --install` installs Apple's command-line developer tools, including Git.

Create an isolated Python environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Expected: installation finishes without errors and `pip check` reports no broken requirements. The environment activation makes `python` refer to this project's Python. Installation downloads software packages; it does not download the OCT dataset.

For an **existing clone**, activate its environment, run `git pull --ff-only`, and rerun `python -m pip install -r requirements.txt`.

## 2. Run the software tests before the first experiment

```bash
python -m pytest -q
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
```

- The first command checks data handling, cache integrity, configuration, exports and the simple commands. On a compatible Mac it also checks native model/gradient, EMA, loss and resume behavior. Tests requiring downloaded pretrained weights are skipped in this first command.
- The second command enables the two actual pretrained integration tests, including tiny self-supervised adaptation, adapted feature extraction, export and reuse. It may download the pinned pretrained model once (approximately 89 MB).
- Both commands use small artificial fixtures; **neither downloads or processes the full OCT dataset**. Artificial test arrays are not OCT research results.
- Continue only if there are no failures. The native MLX tests are intentionally skipped on Windows/Linux. See [the validation record](docs/VALIDATION.md) for what has actually been executed.

Native fine-tuning cannot be verified on the Windows development machine. Passing portable tests does not establish that the Mac run or representation quality has been verified.

## 3. Point to the existing OCT dataset and run the small check

Use the directory that contains `train`, `val` and `test`. Nested extraction directories are also supported. Each split must contain the original four class folders:

```text
OCT2017/
  train/  NORMAL/  CNV/  DME/  DRUSEN/
  val/    NORMAL/  CNV/  DME/  DRUSEN/
  test/   NORMAL/  CNV/  DME/  DRUSEN/
```

For example, if the folder is in Downloads:

```bash
python oct.py run "$HOME/Downloads/OCT2017"
```

Use your own location if different; keep quotes around paths with spaces. You can also drag the folder from Finder into Terminal to obtain its path.

This single command:

1. Resolves the local dataset and checks the original splits.
2. Selects at most **4 images per original class per split: 48 images total** when all classes have enough images. It enumerates filenames for counts, but hashes/decodes only the selected images.
3. Loads the pinned pretrained DINOv2 weights, downloading them once if missing.
4. Fine-tunes using **only TRAIN images**, for one epoch capped at **two optimizer updates**.
5. Exports and freezes the adapted EMA teacher backbone.
6. Extracts and caches the dense features for the selected train/val/test images.

**This is a software smoke check, not sufficient training for a research result.** Dataset downloading is disabled for this command: an unavailable local path causes an error. Existing local data is never downloaded again.

At completion, Terminal prints the feature-manifest location and the next command. Results go to:

```text
mini-project/outputs/smoke-adapted/
```

## 4. See the dense arrays

```bash
python oct.py show
```

This selects the first test scan, verifies its cached shard, exports its arrays, and prints the image ID, original path, output paths, shapes and numerical previews. It needs no MLX model, GPU or original image files. It also works on Windows with copied outputs.

The easy-to-share files are:

```text
outputs/smoke-adapted/dense_sample.npz
outputs/smoke-adapted/dense_sample.json
```

The `.npz` contains **all values**, even when Terminal abbreviates the display with `...`. The JSON explains the source image, model identity, array shapes, checksums and coordinate system. Repeating `show` reuses a matching verified export; it does not retrain or extract features again.

To print every value, or save that very long display as text:

```bash
python oct.py show --all-values
python oct.py show --all-values > outputs/smoke-adapted/dense_arrays.txt
```

The text file contains the CLS, patch matrix and spatial grid, so some values occur more than once. Use the `.npz` for numerical analysis.

### What the arrays mean

| Array in the single-scan export | Shape at the default input size | Meaning |
|---|---|---|
| `cls` | `(384,)` | Auxiliary global image descriptor |
| `patches` | `(256, 384)` | 256 spatial patch descriptors, each with 384 learned values |
| `spatial_grid` | `(16, 16, 384)` | The same patch values arranged by image row and column |
| `patch_row_col` | `(256, 2)` | Zero-based row and column of each patch |
| `patch_bounds_xyxy` | `(256, 4)` | Patch input bounds `[x0, y0, x1, y1]`, upper bounds exclusive |
| `sample_id` | scalar string | Image identifier from the split index |
| `original_class` | scalar string | Original dataset diagnosis, retained as provenance |
| `binary_label` | scalar integer | Legacy provenance mapping: NORMAL=0, other diagnoses=1; never used in the adaptation loss |

The image is converted to RGB, resized with shortest edge 256, center-cropped to **224 × 224**, then normalized. A patch covers a **14 × 14 input anchor**, producing a **16 × 16 grid**. The descriptors include context from other patches through self-attention. Coordinates refer to the preprocessed crop, not the original scan dimensions.

These are **98,304 dense feature values per scan**, not lesion masks, class probabilities, anatomical labels or CAM heatmaps. The features are final LayerNorm outputs without an additional L2 normalization. Patches default to float16 storage; CLS is float32.

For later Python analysis, this is sufficient:

```python
import numpy as np

with np.load("outputs/smoke-adapted/dense_sample.npz", allow_pickle=False) as scan:
    patches = scan["patches"]          # (256, 384)
    grid = scan["spatial_grid"]        # (16, 16, 384)
    one_patch = grid[7, 11]            # (384,), same as patches[123]
    print(one_patch)
```

### Choose another scan

`show` prints the split-index JSON path. Open that JSON and copy a `sample_id` from its `records` list. Replace the example placeholder below with that exact ID:

```bash
python oct.py show --split train
python oct.py show --split test --sample-id "COPY_EXACT_SAMPLE_ID_HERE"
```

The first command exports `dense_sample_train.npz/.json`. An explicitly selected sample is exported under `outputs/smoke-adapted/samples/<sample-key>.npz/.json`; Terminal prints the exact path. Existing exports are not silently overwritten.

## 5. Run the full experiment after the checks pass

For the same existing dataset:

```bash
python oct.py run "$HOME/Downloads/OCT2017" --full
python oct.py show --full
```

Again, substitute the real dataset path. Full mode uses `configs/default.yaml`: all selected training images, **5 epochs**, training batch size **2**, learning rate **0.00001**, followed by dense extraction for all three splits. These are starting experimental settings, not validated optimal hyperparameters. Full fine-tuning is much more expensive than the two-update smoke check; no runtime estimate has been measured on the professor's Mac.

Full outputs are separate:

```text
outputs/oct-adapted/
outputs/oct-adapted/dense_sample.npz    # created by show --full
outputs/oct-adapted/dense_sample.json
```

Use `show --full` when inspecting the full run. Plain `show` deliberately continues to show the small smoke run.

The indexed dataset copy contains 84,484 images. Its float16 patch arrays alone need about **15.5 GiB**, plus CLS/metadata, the dataset and training resume states. Float32 patch storage needs about **30.9 GiB** before those extras. All completed epoch states are retained. Actual selected image counts are recorded in `dataset_summary.json`.

### If the Mac does not already have the dataset

For initial testing, use the two test commands in step 2. An optional tiny artificial image run is described below. Neither requires a full OCT download.

When ready to deliberately download and run the full dataset:

```bash
python oct.py run --full
python oct.py show --full
```

The loader first tries `OCT_DATASET_PATH`, if set, otherwise the configured local directory `data/OCT2017`. Only if that path is unavailable does full mode use KaggleHub's `paultimothymooney/kermany2018` download and local cache. This downloads/extracts the dataset; it is not streaming. An existing malformed directory raises an error rather than falling back. Full mode also permits fallback when an explicitly supplied local path is missing, so check your spelling and available disk space.

If Kaggle requests authentication:

```bash
python -c "import kagglehub; kagglehub.login()"
```

Follow the authentication prompt; keep credentials out of the repository.

## 6. Repeat, resume, or change an experiment

To verify reuse after the small run, use the same dataset and settings:

```bash
python oct.py run "$HOME/Downloads/OCT2017" --offline
```

A matching completed backbone is reused without further training. Valid feature shards are reused without another feature-extraction forward pass. With the usual 48-image smoke selection, expect `extracted 0, reused 16` for each split. File hashes and cache checks still involve disk reads.

After interruption, rerun the same command. Training resumes after the last completed epoch; an unfinished epoch is replayed. Extraction resumes from valid completed shards. A reported stale lock must be investigated before removal: ensure no other writer is still running.

To change training settings, edit `configs/default.yaml` before the full run. Use a new output directory for changed epochs, learning rate, seed or training selection:

```bash
python oct.py run "$HOME/Downloads/OCT2017" --full --output-dir outputs/experiment-02
python oct.py show --full --output-dir outputs/experiment-02
```

The pipeline refuses incompatible training contracts in an existing output directory. Cache identities include the **adapted backbone's SHA-256**, preprocessing, sample identity and storage settings, so features from different adapted models are not silently mixed.

In a new Terminal session:

```bash
cd "/absolute/path/to/mini-project"
source .venv/bin/activate
python oct.py show
```

The activation must be repeated in each new Terminal. You do not need to reinstall packages each time.

## 7. Where every result is saved

All default paths below are relative to the cloned repository. Substitute `outputs/oct-adapted` for `outputs/smoke-adapted` after a full run, or use your custom output directory.

| File or directory | Created by | What to use it for |
|---|---|---|
| `outputs/smoke-adapted/features_manifest.json` | `run` | Starting point for all features; links to the train/val/test indices |
| `outputs/features/<adapted-identity>/<split>/index-<selection>.json` | `run` | List of image IDs, original paths, labels, hashes and shard mappings |
| `outputs/features/<adapted-identity>/<split>/<shard-hash>.npz` | `run` | Actual cached arrays for multiple scans: CLS, patches and provenance |
| Same shard name with `.json` | `run` | Shard integrity/metadata record |
| `outputs/smoke-adapted/dense_sample.npz` | `show` | All arrays for the first test scan, easy to load with NumPy |
| `outputs/smoke-adapted/dense_sample.json` | `show` | Export provenance, coordinates, dtypes, shapes and checksum |
| `outputs/smoke-adapted/dense_sample_train.npz/.json` or `dense_sample_val.npz/.json` | `show --split …` | First scan in the requested non-test split |
| `outputs/smoke-adapted/samples/<sample-key>.npz/.json` | `show --sample-id …` | A specifically requested scan |
| `outputs/smoke-adapted/finetuning/backbone.safetensors` | `run` | Final adapted EMA teacher backbone; no SSL projection heads |
| `outputs/smoke-adapted/finetuning/backbone.json` | `run` | Completed adaptation contract, model hash, settings and provenance |
| `outputs/smoke-adapted/finetuning/history.json` | `run` | Self-supervised loss, gradient and teacher/feature diagnostics by epoch |
| `outputs/smoke-adapted/finetuning/latest.json` | `run` | Pointer to the last complete training state |
| `outputs/smoke-adapted/finetuning/epoch-XXXX/state.safetensors` | `run` | Resumable student/teacher, SSL heads, optimizer and centers |
| Same epoch directory's `state.json` | `run` | State checksum, epoch/step, history and random-generator state |
| `outputs/smoke-adapted/config.resolved.json` | `run` | Effective settings and resolved paths |
| `outputs/smoke-adapted/dataset_summary.json` | `run` | Dataset source, split counts, selection and integrity information |
| `outputs/smoke-adapted/experiment_metadata.json` | `run` | Environment, protocol, adaptation and feature identities |
| `outputs/smoke-adapted/extraction_benchmark.json` | `run` | Per-split extraction/reuse statistics |
| `extraction_config.resolved.json`, `extraction_dataset_summary.json`, `extraction_metadata.json` in the chosen run folder | Advanced `--stage extract` | Separate extraction provenance that preserves the training records |
| `inspection_config.resolved.json`, `inspection_dataset_summary.json`, `portable_smoke.json` in the chosen run folder | Advanced `--stage inspect` | Bounded offline data/preprocessing report; no model executed |

Each cache shard stores `cls [N,384]` and `patches [N,256,384]`; the spatial grid is a reshape when exporting/viewing. Full-mode shards hold up to 128 scans; smoke shards hold up to 8. The pipeline does not assemble every dense feature into one giant in-memory tensor.

The feature cache is shared under `outputs/features/`, **outside** the individual run folder. To transfer a complete run, copy the entire `outputs/` layout so relative links still work. To share just one representation, send its `dense_sample.npz` and `.json`; only NumPy is needed to read the arrays. Datasets, weights, caches and results are ignored by Git and are not uploaded to this repository.

## 8. What happens inside the model

1. The pinned pretrained DINOv2 ViT-S/14 checkpoint initializes a student and a teacher.
2. Two crop geometries are sampled per training image. Each student/teacher crop pair shares geometry, with mild independent photometric changes; student views may be blurred and masked. No vertical flips or rotations are used.
3. All student backbone parameters are unfrozen. The loss combines cross-view global self-distillation and same-view masked-patch self-distillation.
4. Teacher targets are centered, sharpened and stop-gradient. Teacher parameters follow an exponential moving average (EMA) of student parameters.
5. Temporary nonlinear projection heads predict self-supervised prototype distributions. They are not disease classifiers. They are kept in resume states but omitted from the exported backbone.
6. The final epoch's EMA backbone is saved, checked by strict save/reload agreement, frozen and used to generate the dense arrays.

Only original TRAIN records enter optimization. Diagnosis labels are retained as metadata, never supplied to the loss. Validation and test images are used only for feature extraction, not optimization or checkpoint selection.

This is a **DINO/iBOT-style adaptation of pretrained DINOv2**, not a reproduction of Meta's entire DINOv2 training recipe. It omits local multicrop, KoLeo and distributed training. The [method document](docs/FINETUNING.md) gives formulas and references. Low self-supervised loss alone does not prove useful OCT representations; patient-independent evaluation, collapse checks and downstream research validation remain necessary.

This matches the requested deliverable: **adapt DINOv2 to OCT, then provide dense representations only**. It does not implement the classification branch in the earlier diagram.

## 9. What each repository file does

These tables cover the project's tracked files. Paths are relative to the repository root. The separate upstream `dino/` Git submodule has its own source tree and documentation; its internals are not part of the active pipeline.

### Entry points, configuration and installation

| File | Purpose |
|---|---|
| `oct.py` | Simple front door: `run` and `show`; delegates to the same tested pipeline |
| `run_pipeline.py` | Advanced entry point for choosing config, stages, offline mode and adapted checkpoints |
| `configs/smoke_test.yaml` | Small selection, two training updates, small shards, local dataset only |
| `configs/default.yaml` | Full experiment settings and permitted KaggleHub fallback |
| `requirements.txt` | Install project/test dependencies and conditional Apple Silicon MLX dependencies |
| `pyproject.toml` | Package/version metadata, Python/dependency requirements, optional extras and pytest settings |
| `requirements-legacy.txt` | Dependencies for the independent earlier experiments |

### Implementation: `src/oct_dinov2_mlx/`

| File | Purpose |
|---|---|
| `__init__.py` | Package initialization and version |
| `cli.py` | Implements simple commands, chooses smoke/full settings, displays and verifies reusable single-scan exports |
| `pipeline.py` | Orchestrates data preparation, adaptation, feature extraction and run metadata |
| `config.py` | Loads YAML, resolves paths/overrides and validates settings |
| `common.py` | Hashing, atomic JSON writes, system information and native MLX checks |
| `data.py` | Local/Kaggle dataset resolution, lazy images, records, original splits and duplicate checks |
| `preprocessing.py` | Resize, center crop, RGB conversion and normalization for extraction |
| `augmentations.py` | Aligned teacher/student views, photometric changes and patch masks for training |
| `backbone.py` | Validates pinned pretrained weights, builds the native model, loads adapted weights and extracts frozen CLS/patch features |
| `adaptation.py` | Defines the train-only contract and verifies adapted checkpoint identity/checksums |
| `finetuning.py` | SSL heads, losses, student updates, teacher EMA, epoch resume and final backbone export |
| `cache.py` | Writes/reads feature shards, validates checksums, reuses or repairs cached work |
| `representations.py` | Exports one scan's arrays and spatial/provenance metadata |
| `localization.py` | Reshapes patch descriptors into a grid and defines a future localization interface; no lesion localization algorithm |

### Utility scripts

| File | Purpose |
|---|---|
| `scripts/export_dense.py` | Advanced standalone exporter with explicit manifest, sample and output paths |
| `scripts/show_features.py` | Older low-level feature preview using a split-index path |
| `scripts/create_synthetic_smoke.py` | Generates explicitly artificial test images; never substitutes for OCT research data |
| `scripts/verify_parity.py` | Optional original-pretrained MLX versus HF/PyTorch numerical comparison |

### Tests: `tests/`

| File | Purpose |
|---|---|
| `conftest.py` | Creates small artificial OCT-shaped folder fixtures |
| `test_cli.py` | Checks simple command routing, bounded defaults, paths, offline behavior and errors |
| `test_data.py` | Dataset resolution, fallback, lazy loading, transforms and split integrity |
| `test_cache.py` | Cache reuse, corruption repair, identity invalidation, interruption and locks |
| `test_dense_representation.py` | Array values, export shape/order/coordinates, integrity and repeated simple viewing |
| `test_portable_pipeline.py` | Bounded inspection, configuration and original checkpoint schema/hash checks |
| `test_handoff.py` | Platform checks, invalid settings, relative manifests and rejection of classifier stages |
| `test_adaptation.py` | Train-only contracts, augmentation/masks, adapted identity and fine-tune-to-extract routing |
| `test_mlx.py` | Native model/weight checks and opt-in real pretrained integration |
| `test_mlx_finetuning.py` | Native loss mathematics, backbone gradients, EMA, stop-gradient and resume correctness |

### Documentation, automation and earlier experiments

| File | Purpose |
|---|---|
| `README.md` | This complete installation/run/results/file guide |
| `docs/MAC_RUNBOOK.md` | Compact ordered Mac checklist |
| `docs/REPOSITORY_GUIDE.md` | Additional architecture, array and cache explanations |
| `docs/FINETUNING.md` | Exact self-supervised objective, protocol, references and limitations |
| `docs/VALIDATION.md` | Executed checks and outstanding native validation, with historical records clearly separated |
| `docs/RESEARCH_NOTES.md` | Current method summary and historical backbone investigation |
| `docs/CHECKPOINT_AUDIT.json` | Original pretrained checkpoint tensor-header audit |
| `docs/DEPENDENCY_AUDIT.json` | Historical Mac dependency-resolution audit; not a lockfile or proof of native execution |
| `.github/workflows/portable-tests.yml` | Windows/Ubuntu Python 3.12 CI; no Apple GPU tests |
| `.gitignore` | Keeps data, outputs, weights, environments and common credential files out of Git |
| `.gitattributes` | Cross-platform text line-ending rules |
| `.gitmodules` | Source location of the upstream DINO submodule |
| `dataset_loader.py` | Earlier independent PyTorch dataset loader; not called by the current pipeline |
| `train_classifier.py` | Earlier independent ResNet-18 classification experiment; not called by the current pipeline |
| `dino_oct_cam.py` | Earlier independent DINO/CAM experiment; not called by the current pipeline |
| `dino/` | Original upstream DINO source reference, at its recorded commit and with its upstream license; not needed for current execution |

The normal clone command is sufficient. If you specifically want to browse the upstream reference checkout, run `git submodule update --init dino`. Its internal files are documented by its upstream README and license.

## 10. Optional advanced commands

These are not needed for the ordinary two-command workflow.

### Portable local data check, including on Windows

```bash
python run_pipeline.py --stage inspect --dataset-path "/absolute/path/to/OCT2017" --offline
```

This stays bounded to at most 48 images and does not load a model.

### Artificial Mac plumbing check when no OCT data is present

```bash
python scripts/create_synthetic_smoke.py
python oct.py run outputs/synthetic-smoke-data --output-dir outputs/artificial-adaptation
python oct.py show --output-dir outputs/artificial-adaptation
```

The generator refuses to overwrite an existing fixture; reuse that directory on subsequent runs. These images and resulting features are artificial, and their source is marked in metadata.

### Separate adaptation and extraction

```bash
python run_pipeline.py --stage finetune --dataset-path "/absolute/path/to/OCT2017"
python run_pipeline.py --stage extract --dataset-path "/absolute/path/to/OCT2017" --offline
```

Both default to the smoke config. Add `--config configs/default.yaml` to both for full mode. Extraction requires a completed adapted checkpoint. To use a different completed adaptation, pass `--checkpoint "/path/to/backbone.safetensors"` with `--stage extract`; keep its matching `backbone.json` alongside it. There is no silent fallback to pretrained-only extraction.

### Optional numerical comparison of the original pretrained backbone

```bash
python -m pip install -e '.[parity]'
python scripts/verify_parity.py
```

This installs the optional PyTorch reference dependencies and writes `outputs/parity.json`. Both CLS and patches should report `passed: true`. It compares the original initialization, not the adapted model against different original weights. Add `--image "/path/to/scan.jpeg"` to use a specific image.

### Save the installed environment after a run

```bash
python -m pip freeze > outputs/requirements-used.txt
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| `python` or a package is missing | From the repository, run `source .venv/bin/activate`; install requirements if this is a new environment |
| Apple Silicon/Metal error | Check native arm64 Python, compatible macOS and MLX installation; Windows cannot run these model stages |
| Local dataset unavailable in smoke mode | Correct the quoted dataset path; smoke mode intentionally never downloads the dataset |
| Missing split/class folders | Point at a complete extracted OCT root with train/val/test and all four diagnoses |
| `show` reports no manifest | Complete `run` first; use matching `--full` and `--output-dir` options |
| Export is incomplete/corrupt or belongs to another run | Move only the reported single-scan `.npz/.json` export pair aside and rerun `show`; preserve the feature cache |
| Changed training settings/data conflict with existing results | Choose a new `--output-dir` and use it with both commands |
| Out of memory | Lower `finetuning.batch_size` for adaptation or `features.batch_size` for extraction; use a new run directory if changing adaptation settings |
| Interrupted run or writer lock | Rerun the same command; if a lock remains, ensure its writer has stopped before removing the reported lock |
| Nonfinite loss or zero backbone gradient | Stop and investigate the input/settings/native installation before treating adaptation as successful |
| Old config contains `training:` | Replace it with the supplied configuration using `finetuning:` |
| Looking for classification accuracy or a lesion map | Those are not outputs of this dense-representation pipeline |

Exact duplicate-byte checks do not establish patient independence or detect all near duplicates. Dense feature quality must be evaluated separately; neither a successful run nor the two-step smoke check validates clinical use.
