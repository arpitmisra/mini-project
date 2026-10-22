# Dense OCT representations and classification with DINOv2 and Apple MLX

A local-first research pipeline for extracting **dense spatial representations of OCT scans** with frozen pretrained DINOv2 ViT-S/14. Each preprocessed scan produces a **16 × 16 × 384 feature grid** and a global CLS vector. An optional downstream MLX linear head learns **NORMAL (0)** versus **ABNORMAL (1)** from CLS vectors. CNV, DME and DRUSEN map to ABNORMAL, and every record retains its original diagnosis.

**Repository:** https://github.com/arpitmisra/mini-project

**First run on a Mac:** follow the [ordered Mac runbook](docs/MAC_RUNBOOK.md). It gives the exact commands, expected results, a no-dataset test route, dense-feature export, cache-reuse checks, optional classification and the full-data run in execution order.

Start with **Apple Silicon Mac: setup and smoke test** below. The GitHub repository contains source, configuration, documentation and tests. The large OCT dataset, local Python environment, downloaded model weights and generated outputs are excluded; the pipeline uses existing local data or downloads through KaggleHub when explicitly enabled by the full-run configuration.

The entry point is **`run_pipeline.py`**. See the [detailed repository walkthrough](docs/REPOSITORY_GUIDE.md) for every component, the exact output files, and how the implementation maps to the requested architecture.

## Primary deliverable: dense representations

Both configurations now set **`features.save_patches: true`**. `--stage extract` saves CLS `[384]` and spatial patch `[256,384]` arrays for every selected scan without training a classifier. Reshaping the ordered patch matrix gives `[16,16,384]`; each grid location has a learned 384-value descriptor. This is the dense representation requested by the architecture diagram.

After installing on the Mac, start with 48 images:

```bash
python run_pipeline.py --config configs/smoke_test.yaml --stage extract --dataset-path /path/to/existing/OCT2017
python scripts/export_dense.py --feature-manifest outputs/smoke/features_manifest.json --split test --output outputs/smoke/dense_sample.npz
```

The export creates `dense_sample.npz` with the complete CLS vector, patch matrix, spatial grid, patch row/column positions, crop-relative patch bounds, sample ID and labels. `dense_sample.json` records provenance, shapes, dtypes and the checksum. Add `--sample-id` with an exact ID from the split index to choose another scan; otherwise the first indexed scan is exported. The command reads and verifies only the selected shard, requires neither MLX nor the original image, and works on Windows with a cache copied from the Mac.

```python
import numpy as np
with np.load("outputs/smoke/dense_sample.npz", allow_pickle=False) as scan:
    print(scan["cls"].shape)           # (384,)
    print(scan["patches"].shape)       # (256, 384)
    print(scan["spatial_grid"].shape)  # (16, 16, 384)
    descriptor = scan["spatial_grid"][7, 11]  # 384 values at row 7, column 11
```

These features cover the **224 × 224 preprocessed center crop**, with a 14 × 14 input patch per grid location. Transformer attention includes context from the other patches. They are numerical descriptors, not pixelwise lesion labels or a CAM heatmap. See [representation semantics and diagram alignment](docs/REPOSITORY_GUIDE.md#what-dense-representation-means).

## Current verification status

Implemented: local/KaggleHub dataset resolution, deterministic preprocessing, strict pretrained loading, MLX inference and training, resumable feature shards, validation checkpoint selection, evaluation, prediction, and Mac integration/parity tests.

Reviewed again on **22 October 2026**, including dense export and grid orientation: **49 tests passed, 1 native MLX module skipped**, plus a repeated local-data preprocessing smoke check of **48 images total**. A Python 3.12/macOS 14 arm64 dependency-resolution dry run, including the matching MLX Metal backend, also succeeded during the earlier handoff review. This checks dependency compatibility, not native execution. Directory indexing found **83,484 train, 32 validation, 968 test images**. Indexing filenames is not a full image-processing run. See [the validation record](docs/VALIDATION.md).

**Apple Silicon model execution, real pretrained numerical parity, classifier training and end-to-end model results have not been verified on this host.** No clinical performance or DINOv2 embedding values are claimed from the Windows checks. Test doubles only validate cache mechanics.

## Windows: use the dataset already on disk

The checked-in configuration uses the repository-relative `data/OCT2017`, which exists in this workspace. No absolute Windows path is embedded in the new code.

```powershell
# Use a Python environment with the portable requirements installed.
python -m pip install -r requirements.txt
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline
python -m pytest -q
```

For a dataset elsewhere:

```powershell
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline --dataset-path "D:\datasets\OCT2017"
# Or set it for this PowerShell session:
$env:OCT_DATASET_PATH = "D:\datasets\OCT2017"
```

Path precedence is CLI `--dataset-path`, then `OCT_DATASET_PATH`, then YAML `dataset.local_path`. Dataset, local model, feature-cache and output-directory settings resolve against the project root when relative. Paths passed to `--config`, `--feature-manifest`, `--checkpoint` and `--image` use the shell's current directory; the examples assume the repository root. Nested roots such as `download/OCT2017/OCT2017/train` are discovered; multiple possible roots require an explicit choice. Missing/empty classes or splits are errors.

`inspect` always disables dataset downloads and caps selection at four per class if the configuration is otherwise unlimited. It reads, hashes, and preprocesses only the selected images. It never imports MLX or PyTorch. Normal model/training execution checks the host before a download.

## Apple Silicon Mac: setup and smoke test

Use native arm64 Python, preferably Python 3.12, on Apple Silicon with macOS 14 or newer; the installed MLX wheel may impose a newer OS requirement. The program reports actual platform, Python architecture, versions, and Mac hardware rather than assuming a particular Ultra chip. [Official MLX installation requirements](https://github.com/ml-explore/mlx/blob/main/docs/src/install.rst).

Clone and install in a fresh environment (the commands below do not download OCT images or model weights):

```bash
git clone --recurse-submodules https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform, sys; print(sys.version); print(platform.machine())"
# Confirm arm64 above. An x86_64 Python on an Apple Silicon Mac is using Rosetta.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python run_pipeline.py --stage system
python -m pytest -q
```

If `python3` is not a native Python 3.10+ installation, install a native Apple Silicon Python first; Python 3.12 is recommended. An Intel Mac cannot run this project's Metal execution path.

When an extracted OCT dataset is available on the Mac, run the real-data smoke test (replace the example path):

```bash
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path /path/to/existing/OCT2017
```

The smoke configuration selects at most **4 images per original class per split (48 total)**, saves CLS plus patch features, and trains for three epochs. **Dataset downloads are disabled in this configuration**, even if the local path is missing. Model weights may be fetched once from Hugging Face (~89 MB); they are separate from the dataset. Add `--offline` after weights are cached to prohibit both network fetches.

If the Mac has no OCT data yet, perform a small, explicitly artificial plumbing test first:

```bash
python scripts/create_synthetic_smoke.py
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline --dataset-path outputs/synthetic-smoke-data --output-dir outputs/artificial-smoke
python run_pipeline.py --config configs/smoke_test.yaml --dataset-path outputs/synthetic-smoke-data --output-dir outputs/artificial-smoke
```

These 48 generated images are **not OCT scans**, and their metrics are meaningless for disease classification. The generated dataset is marked synthetic in run metadata. The real local dataset is preferred whenever available.

After smoke/parity checks, extract the full dataset's dense representations:

```bash
python run_pipeline.py --config configs/default.yaml --stage extract
```

This uses `OCT_DATASET_PATH` or `data/OCT2017` when available. Otherwise it automatically calls `kagglehub.dataset_download("paultimothymooney/kermany2018")`. It downloads nothing when a valid local dataset exists. An existing but malformed local directory fails with an explanation rather than triggering a multi-GB fallback. No packages are installed implicitly by the pipeline.

Omit `--stage extract` to additionally train the binary classifier and evaluate the selected checkpoint. A classifier checkpoint is not needed to obtain dense features.

## Dataset retrieval, authentication and disk use

The source is [Kermany 2018 on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018), not the associated notebook. `kaggle kernels pull` fetches notebook source, not this dataset. KaggleHub downloads/extracts files locally and reuses its cache; this pipeline does **not** claim remote image streaming.

[KaggleHub documents](https://github.com/Kaggle/kagglehub) its default `~/.cache/kagglehub` cache, `KAGGLEHUB_CACHE` override, versioned dataset handles, and single-file `path=` downloads. Selective retrieval requires known dataset-relative filenames. This project does not invent a remote file manifest or depend on selective access for smoke testing.

For reproducibility set `dataset.version` to a specific Kaggle dataset version after confirming it on Kaggle. When null, KaggleHub resolves the currently available version; the returned versioned cache path is recorded. Local datasets record root, split, filename, byte size, modification time, and by default per-image SHA-256. Source metadata never includes credentials.

Many public datasets work without login. If Kaggle requests authentication, configure a current Kaggle API token using `KAGGLE_API_TOKEN` or run:

```bash
python -c "import kagglehub; kagglehub.login()"
```

Use the authentication instructions linked from KaggleHub for token creation and any dataset terms. Never commit credentials. Initial full retrieval requires internet and enough space for both compressed and extracted copies. Budget roughly **15–25 GB of free space for dataset download/extraction**, depending on dataset version and whether archives are retained; this is a planning allowance, not a measured size of this copy. The pipeline does not download the dataset to measure its size.

Feature storage is additional. Float32 CLS vectors require 1,536 bytes/image. Float16 spatial features at 224×224 require 196,608 bytes/image (~192 KiB); for this indexed copy's 84,484 images, that is about **15.5 GiB** before file overhead. Both full and smoke configurations save patches to meet the dense-representation objective. Patches are written in bounded shards, not held for the whole dataset. Set `features.patch_dtype: float32` before extraction if preserving float32 inference values is required; this doubles patch storage to about 30.9 GiB. Default float16 storage rounds the values. For a deliberately CLS-only experiment, explicitly set `save_patches: false`; that cache cannot supply dense features.

To clean caches, stop active writers and remove only the particular experiment/cache directory you intend to discard. `outputs/features/<identity>/` contains this project's derived data. Hugging Face and KaggleHub caches are separate; deleting them causes future retrieval. Never point a cleanup command at the original dataset. A leftover `.writer.lock` can be removed only after verifying its recorded process is no longer writing.

## Extract once; train many classifiers

```bash
# Stage A: frozen backbone, image decoding, cache construction.
python run_pipeline.py --config configs/default.yaml --stage extract

# Stage B: cached CLS -> new classifier; does not load DINOv2 or access images.
python run_pipeline.py --config configs/default.yaml --stage train \
  --feature-manifest outputs/default/features_manifest.json \
  --output-dir outputs/probe-01

# After model selection, evaluate the chosen checkpoint on held-out test data.
python run_pipeline.py --config configs/default.yaml --stage evaluate \
  --feature-manifest outputs/default/features_manifest.json \
  --checkpoint outputs/probe-01/checkpoints/best_classifier.safetensors \
  --output-dir outputs/probe-01
```

Copy a YAML file and vary `training.learning_rate`, `epochs`, `weighted_loss`, etc. while retaining identical model/preprocessing/feature settings. Training settings and random seed are excluded from the feature identity. `train` uses validation only; `all` runs extraction, training and one final test evaluation. Repeatedly selecting experiments by test accuracy invalidates the evaluation protocol.

Each shard's key includes model revision, adapter/MLX-VLM implementation version, preprocessing (including Pillow version), sample identity/content, labels and split, patch persistence and dtype. Stored CLS values remain float32; optional patches can be float16/float32. Every shard has a checksum and metadata. Valid cache hits skip image decoding and model construction. Interrupted extraction preserves completed shards; corrupt shards are regenerated. Changing images, preprocessing, model or feature storage settings produces new appropriate shards. Changing shard grouping or moving from a small selection to the full dataset can change shard membership and require extraction for those groups. Do not edit source images in place during extraction.

Earlier full-run CLS-only caches remain on disk but do not match the new default with patch persistence. Dense extraction must run once to populate a new compatible cache. Subsequent classifier experiments reuse that cache. Cache verification reads the saved shards, including patches, but performs no backbone inference.

New run manifests store index paths relative to the manifest wherever possible; shard paths inside an index are also relative. Copy the complete `outputs/` tree to move a cached run between machines while preserving that layout. Old manifests, or Windows caches on another drive, may contain absolute index paths; update those three entries after moving. Original image paths remain provenance and are not needed by `train`. Identical cache identities, including Pillow version, are checked before training/loading a classifier; preserve the recorded dependency versions when moving a cache.

## Model and representations

```text
OCT image -> RGB -> shortest edge 256 (bicubic) -> center crop 224
          -> ImageNet normalization -> NHWC float32
          -> frozen DINOv2 ViT-S/14 (MLX)
          -> CLS [B,384] + patch tokens [B,256,384]
          -> disk feature shards
          -> MLX Linear(384,2) -> NORMAL / ABNORMAL
```

The implementation uses [`mlx-vlm==0.7.6`'s standalone DINOv2 model](https://github.com/Blaizzy/mlx-vlm/tree/v0.7.6/mlx_vlm/models/dinov2), under MIT, with Meta's [Apache-2.0 `facebook/dinov2-small` checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056). The exact revision is pinned, and the downloaded weights' SHA-256 is saved in shard metadata. MLX-VLM has a broad VLM dependency graph; this project invokes only its native MLX DINOv2 model, not any language model. PyTorch is not used for feature extraction or classifier training. The optional parity tool explicitly uses a separate PyTorch reference.

The checkpoint has 12 blocks, 6 attention heads, 384-dimensional embeddings, 14×14 patches, a 37×37 learned positional grid at its 518 training resolution, a CLS token, and **no register tokens**. MLX-VLM interpolates positional embeddings to the input grid and maps HF weights by fusing Q/K/V and transposing the patch convolution. The adapter verifies the published file SHA-256, exact source key/shape set and complete converted parameter set before a strict load; partial or random backbone loading is never accepted. The actual remote header's 223 keys/shapes were verified without a full checkpoint download. Local model folders must contain the original pinned HF safetensors file. Unverified model/register variants are rejected. See [source investigation](docs/RESEARCH_NOTES.md).

At 224×224, 224/14 = 16, so there are 16×16 = 256 patches plus one CLS token. This differs from original DINO ViT-S/16's 196 patches. Our adapter accepts `[B,H,W,3]`; a PyTorch reference receives the transpose `[B,3,H,W]` of precisely the same preprocessed array.

Pretraining occurred elsewhere. This project trains a supervised downstream classifier; it does not reproduce DINO/DINOv2 self-supervised pretraining.

## Outputs and prediction

`extract` writes `features_manifest.json`, `dataset_summary.json`, resolved config and extraction timings, with cached `.npz` shards and JSON indices under `outputs/features/`. `scripts/export_dense.py` turns one cached scan into a standalone `.npz` and `.json` deliverable. Training additionally writes `checkpoints/best_classifier.safetensors`, compatibility metadata and `metrics/training_history.csv`. Final test evaluation writes `metrics/results.json`, per-image predictions, classification report and confusion/ROC/PR plots. An `all` run writes `experiment_metadata.json`; a later `evaluate` writes separate `evaluation_metadata.json` and `evaluation_config.resolved.json`, preserving training provenance. See the [complete output contract](docs/REPOSITORY_GUIDE.md#exact-output-artifacts).

Evaluation reports accuracy, abnormal precision/recall/F1, normal specificity, per-class results, confusion matrix, ROC-AUC and **average precision** (the documented PR summary). AUC is null when only one ground-truth class is present. Validation loss selects the checkpoint; inverse-frequency loss weights are computed from training labels only. AdamW updates only the head; MLX values/parameters/optimizer state are explicitly evaluated. After saving, the checkpoint is reloaded and its predictions are checked against the values before saving.

Actual extraction wall time includes synchronized inference, shard I/O and initial model loading/downloading when needed; it is not an isolated inference-speed benchmark. Runs also record reused/extracted counts, cache bytes, classifier training time and peak memory where supported. These are measured when the Mac runs, not estimated benchmarks.

```bash
python scripts/show_features.py outputs/features/<identity>/test/index-<selection>.json
python run_pipeline.py --config configs/default.yaml --stage predict \
  --checkpoint outputs/probe-01/checkpoints/best_classifier.safetensors \
  --image /path/to/scan.jpeg
```

Prediction prints class probability, eight CLS values, and patch shape. The config must match the checkpoint's model and preprocessing identity, including storage settings. `FeatureCache.load_sample(sample_id)` loads one sample; `localization.spatial_grid(cache, sample_id)` returns `[16,16,384]` when patches were saved.

## Mac verification before research experiments

```bash
# Native MLX gradient, frozen-weight, shape, and checkpoint tests; no downloads.
python -m pytest -q

# Explicit opt-in: real checkpoint, batch consistency, full small synthetic-image pipeline.
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained

# Separate PyTorch/HF reference, same pinned weights and identical preprocessed pixels.
python -m pip install -e '.[parity]'
python scripts/verify_parity.py --image /path/to/an/OCT/image.jpeg
```

The parity report contains maximum/mean absolute, RMS and relative L2 errors for CLS and patch outputs. It fails outside `atol=5e-4, rtol=1e-3`; inspect any failure before using the model scientifically. Shape agreement alone is insufficient. `--offline` is supported once weights are present. Save `python -m pip freeze` with each study because transitive dependencies can change, and inspect the reported versions.

The repository's [GitHub Actions workflow](https://github.com/arpitmisra/mini-project/actions) runs portable tests on Windows and Ubuntu using synthetic images. It does not download the OCT dataset or pretrained weights, and does not replace Apple Silicon verification. On an arm64 Mac, missing/broken MLX installation fails the native tests rather than silently skipping them.

## Common first-run errors

| Error | Action |
|---|---|
| Local dataset unavailable / downloads disabled | Supply `--dataset-path`, set `OCT_DATASET_PATH`, or use the small synthetic fixture for the initial smoke test. Full `configs/default.yaml` permits KaggleHub fallback. |
| Apple Silicon required / Metal unavailable | Check `platform.machine()` is `arm64`, use a native Python virtual environment, and check macOS/MLX compatibility. Windows supports inspection and portable tests only. |
| Module missing / MLX-VLM version mismatch | Activate `.venv`, rerun `python -m pip install -r requirements.txt`, then `python -m pip check`. |
| Features or classifier metadata mismatch | Use the same model, preprocessing, Pillow version, and feature settings that produced that cache/checkpoint. Change only training settings for cached classifier experiments. |
| Failed checkpoint SHA-256 check | Restore the pinned original HF `model.safetensors`; converted, incomplete or modified checkpoints are intentionally rejected. |
| Memory exhaustion | Reduce `features.batch_size` (first try 1 or 2). Completed feature shards will be reused when the command is rerun. |
| Validation split too small for stratification | Keep the official validation split for a tiny smoke test; use `validation_fraction` only with enough training samples per original class. |

After the Mac smoke, native tests and parity check pass, retain `outputs/parity.json`, the run metadata and `python -m pip freeze` output alongside research results. Code review on Windows cannot guarantee that every Metal/runtime combination is error-free.

## Scientific limits and extension path

The official validation set in this copy contains just 32 images. Optional `dataset.validation_fraction: 0.1` derives a reproducible, diagnosis-stratified validation split **from training only**, excluding the original validation set. This is explicitly image-level: reliable patient identifiers are not supplied and are never guessed from filenames. Implement a reviewed patient-ID manifest and grouped splitting before claiming patient-independent results.

Path and default SHA-256 checks reject exact cross-split duplicates in the selected population. They do not prove patient independence or detect differently encoded copies. A smoke audit covers only its selected images; a full run audits all indexed images. No original test image is reassigned to training/validation. A leakage failure stops the run; resolve it with an explicit documented protocol, not silent deletion or test mixing.

OCT differs substantially from DINOv2 pretraining imagery. Class imbalance and differing abnormal disease subgroups matter. RGB conversion does not add clinical information. Center cropping can remove peripheral abnormalities, while direct square resizing can distort retinal anatomy; preprocessing choices require validation. This project provides experimental image classification, not clinical diagnosis.

Spatial embeddings are dense real-valued features, not lesion masks. The localization module exposes a spatial grid and a method interface; **no heatmaps or localization accuracy are claimed**. Traditional CAM assumes an appropriate spatial classification architecture. Applying final CLS-head weights/gradients to final patch outputs is not valid ViT Grad-CAM. A future method should target an internal layer that affects the class score, verify nonzero patch gradients, and be evaluated against independent localization annotations.

Future work: use preserved original classes to train a four-output MLX head and corresponding multiclass metrics; add verified attention rollout or internal-layer gradient localization; implement end-to-end fine-tuning with revised cache identities (features change when the backbone changes); compare frozen/fine-tuned representations; evaluate patient-level splits, external OCT data, confidence intervals and disease-specific errors. Read the original papers and [research notes](docs/RESEARCH_NOTES.md) before drawing localization or reproduction conclusions.
