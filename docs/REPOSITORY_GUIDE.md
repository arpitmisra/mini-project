# Repository walkthrough and alignment with the requested OCT pipeline

This guide describes the code and output contract. Shapes shown here are specified and tested contracts, not evidence that real pretrained embeddings have already been generated on the Windows development machine. Native model execution and numerical parity must be checked on Apple Silicon.

## The objective

The primary task is to use a pretrained DINO-family backbone to produce a dense representation of each OCT scan. The supplied diagram specifically selects **DINOv2 ViT-S/14**, frozen, running with Apple MLX. The repository implements that choice using Meta's pinned `facebook/dinov2-small` checkpoint and the audited MLX-VLM implementation.

The primary deliverable is a set of spatial feature tensors, accompanied by image IDs, original diagnoses, split information and model/preprocessing metadata. The binary classifier is an additional experiment that measures how useful the global features are for distinguishing NORMAL from ABNORMAL. A successful classifier does not establish spatial localization accuracy.

Use `run_pipeline.py --stage extract` when the task is feature extraction alone. Use `--stage all` (the default stage) to also train and evaluate the linear classifier. With no `--config` argument, the CLI selects the small smoke configuration.

## What dense representation means

For one scan, preprocessing produces an RGB tensor of shape `[224,224,3]`. Dividing each spatial dimension by patch size 14 gives 16 positions along each axis, or 256 input patches.

The transformer processes a sequence of 257 tokens: one CLS token and 256 image patch tokens. Each token has 384 channels. The verified checkpoint has no register tokens. The adapter returns two arrays separately:

| Representation | One scan | Batch of B scans | Meaning |
|---|---|---|---|
| CLS | `[384]` | `[B,384]` | Global learned summary used by the linear classifier |
| Patch matrix | `[256,384]` | `[B,256,384]` | One contextual descriptor for each spatial patch |
| Spatial grid | `[16,16,384]` | Conceptually `[B,16,16,384]` | The same patch descriptors restored to their two-dimensional arrangement |

The spatial grid is the dense representation. It contains **98,304 scalar values per scan**: 16 × 16 × 384. Channel 0 or channel 100 does not have an assigned clinical meaning such as “fluid” or “drusen.” These are learned feature coordinates. The outputs have the backbone's final LayerNorm applied; the pipeline does not additionally normalize each vector to unit L2 length.

Patch order is row-major: `token_index = row * 16 + column`. For example, `spatial_grid[7,11]` equals `patches[123]`. Its input anchor covers columns 154–167 and rows 98–111 in the **preprocessed 224 × 224 crop**. Self-attention means its final descriptor can include information from across the image, not only that 14 × 14 region.

The term dense here means a spatial grid of descriptors at patch resolution. It does not mean a 384-value descriptor at every original image pixel, a lesion segmentation mask, an attention map, or a validated class activation map. Generating these other outputs would require additional methods and validation.

## How the diagram maps to the implementation

| Diagram component | Implementation and actual behavior |
|---|---|
| Local OCT dataset | `data.resolve_dataset()` uses an existing configured directory first. CLI path overrides `OCT_DATASET_PATH`, which overrides YAML. |
| KaggleHub fallback | When the local path is unavailable and downloads are enabled, the pipeline requests `paultimothymooney/kermany2018`. Downloaded files are stored locally through KaggleHub. |
| Original splits and labels | `KermanyOCTDataset` indexes train/val/test; NORMAL=0 and CNV/DME/DRUSEN=1. Original class and original split stay in the metadata. |
| Lazy loader/preprocessing | Images are decoded per batch. RGB conversion, bicubic shortest-edge resize to 256, center crop to 224, 1/255 scaling, ImageNet mean/std, NHWC float32. |
| Frozen pretrained backbone | `DINOv2Extractor` verifies the checkpoint hash and parameter schema, loads the MLX model, freezes it and selects evaluation mode. |
| CLS and spatial features | The final normalized CLS and patch tokens are returned separately, with exact shape and finite/nontrivial-value checks. |
| Feature cache | `.npz` shards store CLS, patches, labels, sample IDs and diagnoses. JSON metadata and SHA-256 checks protect reuse. Both supplied configurations now save patches. |
| Dense output for inspection | `scripts/export_dense.py` exports one scan's full feature arrays and crop coordinates from the cached shards. No trained classifier is needed. |
| Train on TRAIN CLS | `training.py` trains only an MLX linear layer from 384 features to 2 logits. Backbone features stay fixed. |
| Validation-only selection | The best classifier is chosen by validation loss, with early stopping. No test score chooses the checkpoint. |
| Final held-out evaluation | `evaluation.py` produces test predictions, metrics and plots after checkpoint selection. |
| Saved outputs | Manifests, feature shards, classifier weights where requested, configs, timing, histories and provenance are written under `outputs/`. |

Two details refine the diagram's shorthand:

1. “Resize 224 × 224” is implemented as aspect-preserving resize followed by a center crop, matching the pinned checkpoint's [preprocessor configuration](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/preprocessor_config.json). The grid covers that crop, not the entire original scan. Mapping onto the original image needs the resize/crop geometry; simple stretching would misalign it.
2. The diagram calls patch caching optional. Since dense features are the main request, both configurations now enable it. Training still consumes CLS only; the patch grid is retained for spatial experiments.

The diagram's “per-split metrics” box is broader than the current saved reports. The code saves extraction statistics for all three splits, train/validation loss history with validation accuracy/F1, and a comprehensive final test report. It does not save comprehensive train and validation classification reports. “PR-AUC” is reported specifically as scikit-learn average precision, not trapezoidal PR area.

## What every repository component does

### Entry point, configuration and environment

| File | Responsibility |
|---|---|
| `run_pipeline.py` | Makes the source package importable and invokes the CLI. This is the command used in the README. |
| `configs/default.yaml` | Full selection, local dataset first, KaggleHub fallback permitted, dense patch caching enabled, up to 30 classifier epochs. |
| `configs/smoke_test.yaml` | At most four images per original class per split (48 total), no dataset download, smaller batches/shards, patches enabled, three classifier epochs. |
| `requirements.txt` | Installs the project/test dependencies and conditionally installs MLX dependencies for an arm64 Mac. |
| `pyproject.toml` | Package metadata, portable dependencies, optional Mac/parity/test dependency groups and pytest configuration. |
| `.gitignore` | Excludes local data, environments, generated outputs, cached weights and common credentials. |
| `.gitattributes` | Defines consistent text line endings across Windows and macOS. |
| `.gitmodules` | Records the upstream URL for the `dino` submodule. |

### Current implementation: `src/oct_dinov2_mlx/`

| Module | Responsibility |
|---|---|
| `__init__.py` | Package initialization. |
| `config.py` | Reads YAML, merges allowed settings, resolves paths and validates types/ranges. |
| `common.py` | Hashing, atomic JSON writes, system metadata and Apple Silicon/Metal checks. |
| `data.py` | Local/KaggleHub dataset resolution, nested directory discovery, lazy loading, original/binary labels, selected-image hashes, split integrity and optional train-derived validation. |
| `preprocessing.py` | Deterministic RGB/resize/crop/normalization and the preprocessing cache identity. |
| `backbone.py` | Pinned checkpoint contract, strict HF-to-MLX weight conversion, frozen DINOv2 inference and output checks. |
| `cache.py` | Feature shard extraction, checksums, index files, cache hits, interrupted-run recovery, corruption detection/repair and reading cached samples. |
| `localization.py` | Validated patch-to-grid reshape plus a protocol for a future localization method. No implemented lesion localization algorithm. |
| `representations.py` | Portable dense sample export, coordinate arrays and provenance sidecar. Reads only the requested shard and verifies it. |
| `training.py` | MLX binary linear probe, weighted cross-entropy, AdamW, early stopping, checkpoint saving and verified reloading. |
| `evaluation.py` | Binary metrics, prediction table, classification report and confusion/ROC/precision-recall plots. |
| `pipeline.py` | Coordinates stages, writes manifests/configs/metadata and keeps extraction separate from cached classifier experiments. |

### Utilities, tests and documentation

| File or directory | Responsibility |
|---|---|
| `scripts/export_dense.py` | CLI to export a complete single-scan representation as `.npz` plus `.json`. |
| `scripts/show_features.py` | Prints a sample ID, CLS shape/first values and patch/grid shapes from a split index. |
| `scripts/create_synthetic_smoke.py` | Creates 48 artificial images for plumbing checks when real OCT images are unavailable. These are explicitly marked synthetic. |
| `scripts/verify_parity.py` | Optional comparison of native MLX CLS/patch outputs against PyTorch/Hugging Face for the same pinned weights and identical preprocessed pixels. |
| `tests/conftest.py` | Isolated artificial dataset fixture used by tests. |
| `tests/test_data.py` | Dataset discovery, fallback, no-download behavior, preprocessing, labels and split-integrity checks. |
| `tests/test_cache.py` | Cache round trips, reuse without inference/decoding, corruption repair, invalidation, resume and writer locks. |
| `tests/test_portable_pipeline.py` | Inspection, model-schema checks, configuration, metrics and evaluation artifacts. |
| `tests/test_handoff.py` | Early unsupported-platform rejection, input validation, portable manifests and preservation of training provenance during evaluation. |
| `tests/test_dense_representation.py` | Dense default, exact export values, row/column orientation, crop bounds, dtype, selected-shard verification, error cases and CLI execution. Test arrays are synthetic. |
| `tests/test_mlx.py` | Mac-only gradients, frozen parameters, token shapes, weight conversion, checkpoint round trips and opt-in pretrained integration. |
| `.github/workflows/portable-tests.yml` | Runs portable tests and syntax checks on Windows and Ubuntu. Does not certify Metal inference. |
| `README.md` | Installation, extraction/training commands, outputs, testing and troubleshooting. |
| `docs/RESEARCH_NOTES.md` | Upstream sources, checkpoint/model selection, inspected API details and research limitations. |
| `docs/CHECKPOINT_AUDIT.json` | Record of the remote safetensors header/schema audit, without downloading full weights. |
| `docs/DEPENDENCY_AUDIT.json` | Mac-targeted dependency-resolution audit and its limitations. |
| `docs/VALIDATION.md` | What was actually executed, results and what still requires a Mac/full-data run. |
| `docs/REPOSITORY_GUIDE.md` | This component-by-component explanation and architecture alignment. |

For completeness, the repository also contains earlier independent experiments. `dataset_loader.py` is a PyTorch binary dataset loader; `train_classifier.py` trains a **ResNet-18** binary classifier; `dino_oct_cam.py` is an original-DINO ViT-S/16 four-class/CAM experiment with a hard-coded Windows path. These are not invoked by `run_pipeline.py`. The CAM experiment should not be used as evidence of valid localization: a head reading the final CLS token does not provide useful gradients with respect to separately selected final patch outputs. `requirements-legacy.txt` belongs to those experiments. The `dino/` submodule contains the upstream original-DINO source at its recorded commit, including its license. The current backbone comes from the pinned MLX-VLM dependency, not this submodule.

Local `data/`, `.venv/`, `outputs/` and tool caches may appear in the working directory but are not part of the published source. The repository does not include precomputed real OCT embeddings or measured classifier performance.

## Execution flow

1. **Resolve configuration.** CLI path, environment variable and YAML precedence select the dataset. Relative configured paths are interpreted from the repository root. Native model stages check Apple Silicon/Metal before any dataset retrieval.
2. **Resolve and index data.** The loader finds train/val/test and the four original class folders. Existing malformed local data causes an error rather than a large automatic download. Selected records retain their original diagnosis and binary label.
3. **Check split integrity.** Resolved-path overlaps and exact byte duplicates across selected splits are rejected. These checks do not infer patient identity. Inspection/smoke checks only cover selected images.
4. **Preprocess batches.** Pixel decoding is lazy. The source image is converted to RGB (a grayscale scan becomes three equal raw channels), resized/cropped and normalized. RGB conversion adds no new clinical information.
5. **Load the frozen backbone if needed.** The exact model revision and checksum are enforced. All source and converted parameter keys/shapes must match before strict loading. No randomly initialized partial backbone is accepted.
6. **Extract and cache.** A batch produces float32 CLS and patch outputs. CLS is saved as float32; patches are saved as float16 by default, or float32 when configured. A shard is completed and validated before the next shard. Model creation is lazy: valid cache hits do not need the backbone.
7. **Deliver dense features.** The extraction stage can stop here. Exporting a scan reads only its verified shard and writes standalone arrays; no classifier, original images or Mac is needed to inspect an existing export.
8. **Optionally train a linear probe.** Only cached train/validation CLS arrays are loaded for optimization. AdamW changes the head's 770 parameters (384 × 2 weights + 2 biases). Training-label frequencies determine optional loss weights. Validation loss selects the checkpoint; checkpoint reload predictions are checked for equivalence.
9. **Evaluate the selected model.** Held-out test CLS vectors produce two softmax probabilities per image. Argmax chooses the label (a probability tie chooses NORMAL). Metrics and prediction tables are saved. Repeatedly choosing experiments using test scores would compromise this protocol.

Training a new head does not improve or alter the frozen dense descriptors. If the backbone or preprocessing changes, its cached representations must be regenerated. The code does not implement DINO self-supervised pretraining or backbone fine-tuning.

## Exact output artifacts

These paths use the full configuration. The smoke run uses `outputs/smoke/` instead of `outputs/default/`. Hash-based cache directory/file names depend on model, preprocessing and selected samples.

```text
outputs/
  default/
    config.resolved.json
    dataset_summary.json
    features_manifest.json
    extraction_benchmark.json
    checkpoints/                         # after train or all
      best_classifier.safetensors
      best_classifier.json
    metrics/
      training_history.csv               # after train or all
      results.json                       # after evaluate or all
      predictions.csv
      classification_report.txt
      confusion_matrix.png
      roc_curve.png                      # when both labels are present
      precision_recall_curve.png          # when both labels are present
    feature_example.json                 # small summary after evaluation
    experiment_metadata.json             # train or all
    evaluation_metadata.json             # separate evaluate stage
    evaluation_config.resolved.json      # separate evaluate stage
  features/
    <feature-identity>/
      train/  val/  test/
        index-<selection>.json
        <shard-hash>.npz
        <shard-hash>.json
```

`inspect` writes `dataset_summary.json`, `config.resolved.json` and `portable_smoke.json` without feature/model execution. `extract` writes the first four run files and feature caches; it does not create a classifier or classification metrics. `predict` prints a single-image prediction and feature summary; it does not automatically persist that image's complete dense arrays. Use cached extraction and export for the dense deliverable.

### Feature shards and indices

Each feature `.npz` contains `cls [N,384]`, `patches [N,256,384]` when enabled, `labels [N]`, `sample_ids [N]` and `original_classes [N]`, where N is the number of images in that shard. Full configuration uses at most 128 per shard; smoke uses at most 8. The split index preserves image paths, source split, original class, binary label, file size, modification time and image hash. The shard sidecar adds its checksum and model identity.

The image paths are provenance. Cached training and dense export do not read those original images. New manifests use relative index paths wherever possible, so copying the complete `outputs/` layout allows reuse on another machine. Preserve the recorded dependency versions, including Pillow, for matching identities.

The cache identity excludes classifier hyperparameters. It includes model revision/checksum, adapter version, preprocessing, Pillow version, storage settings and sample identity/content. Changing classifier learning rate/epochs does not trigger extraction. Changing patch dtype, preprocessing, model, selected samples or shard membership can require new extraction. A prior CLS-only cache cannot reconstruct the missing spatial features.

### Standalone dense export

```bash
python scripts/export_dense.py --feature-manifest outputs/default/features_manifest.json --split test --output outputs/dense_scan.npz
```

The command exports the first record of the requested split by default. To select another, add `--sample-id 'test/CNV/<actual-filename>.jpeg'` with an exact ID from that split's index. It refuses to overwrite an existing output pair; choose a new path.

| Array in `dense_scan.npz` | Shape at 224 × 224 | Contents |
|---|---|---|
| `cls` | `[384]` | Full global feature vector, float32 |
| `patches` | `[256,384]` | Full patch descriptors in cache dtype |
| `spatial_grid` | `[16,16,384]` | Same descriptors reshaped without transposing spatial axes |
| `patch_row_col` | `[256,2]` | Integer row and column for each patch |
| `patch_bounds_xyxy` | `[256,4]` | Integer `[x0,y0,x1,y1]` crop coordinates, upper bounds exclusive |
| `sample_id`, `original_class`, `binary_label` | Scalars | Identity and ground-truth labels; no classifier prediction is needed |

The matching `dense_scan.json` documents feature identity, original record, dataset-source metadata, shapes/dtypes, coordinate conventions and exported file SHA-256. The spatial grid and flat patches intentionally duplicate the same values for convenient inspection. `np.load(..., allow_pickle=False)` reads the export.

Float16 patch persistence is lossy rounding of the float32 inference outputs. It uses about 192 KiB per scan, approximately 15.5 GiB for this local copy's 84,484 images, before other arrays/metadata. Float32 patches use twice that space. Set `features.patch_dtype: float32` before extraction if the research requires original inference precision. The small smoke cache requires only about 9 MiB of patch arrays.

### Classifier and evaluation outputs

The classifier `.safetensors` contains the linear head, not another copy of DINOv2. Its JSON sidecar records classes, feature identity, training config, seed and checksum. Single-image prediction needs both the head and the pinned backbone. Evaluation from cached CLS features needs only the head and the caches.

`predictions.csv` contains `sample_id`, `source_split`, `original_class`, `binary_label`, `predicted_label`, and `probability_abnormal`. `results.json` contains accuracy, abnormal precision/sensitivity/F1, normal specificity, a 2 × 2 confusion matrix, per-class results, ROC-AUC and average precision. Rows/columns of the matrix follow `[NORMAL, ABNORMAL]`. Undefined AUC values are JSON null.

The history records epoch, train loss, validation loss, validation accuracy and validation abnormal F1. Timing is measured at runtime: extraction includes model loading/download where needed, synchronized inference and disk I/O. It is not an isolated inference-speed benchmark. Memory statistics are recorded by `all`/`evaluate`; extraction-only runs currently save extraction timing/cache statistics rather than a separate peak-memory report.

## Practical handoff sequence

Follow the README to clone, create a native arm64 Python environment and install dependencies. Then:

```bash
# 1. Portable checks and a bounded real-data inspection; no model/data downloads.
python -m pytest -q
python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline --dataset-path /path/to/OCT2017

# 2. Native feature extraction on only 48 real images; weights may download once.
python run_pipeline.py --config configs/smoke_test.yaml --stage extract --dataset-path /path/to/OCT2017
python scripts/export_dense.py --feature-manifest outputs/smoke/features_manifest.json --output outputs/smoke/dense_sample.npz

# 3. Train only if the classification experiment is wanted; use that same cache.
python run_pipeline.py --config configs/smoke_test.yaml --stage train --feature-manifest outputs/smoke/features_manifest.json
python run_pipeline.py --config configs/smoke_test.yaml --stage evaluate --feature-manifest outputs/smoke/features_manifest.json --checkpoint outputs/smoke/checkpoints/best_classifier.safetensors

# 4. Real model integration and a separate numerical reference comparison.
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
python -m pip install -e '.[parity]'
python scripts/verify_parity.py --image /path/to/an/OCT/image.jpeg
```

Replace example paths and the placeholder filename with real ones before running. The parity tool compares both CLS and patch tensors. After smoke and parity succeed, use `configs/default.yaml --stage extract` for the full collection, then train from `outputs/default/features_manifest.json` if required. Re-run extraction with unchanged settings to check `reused` counts; the extractor should not be constructed when all shards are valid.

## What is established and what remains open

The implementation matches the supplied diagram's **frozen DINOv2 feature extraction and binary linear-probe architecture**. The export makes the dense representation a direct, inspectable deliverable. DINOv2 is the backbone specified in that image; reproducing original DINO v1 would be a different experiment. Meta describes global and dense representation learning in its [official DINOv2 repository](https://github.com/facebookresearch/dinov2).

Portable tests establish software behavior using artificial data, and the real local inspection checks preprocessing on 48 OCT scans. They do not establish real MLX output values, spatial usefulness or disease classification performance. See `VALIDATION.md` for the executed results and remaining Mac checks.

The provided validation split has only 32 images. The optional train-derived validation split is image-level; no patient-independent protocol has been established. Exact duplicate checks do not detect all patient overlap or differently encoded copies. Center cropping can discard peripheral OCT anatomy. These are material limits when evaluating representation quality.

The next scientific task after extraction is to evaluate what these descriptors capture on OCT data, using a stated task and suitable annotations or independent evaluation. Classification metrics measure the CLS-based head. Spatial similarity, lesion localization or segmentation require their own evaluation and must not be inferred from classification accuracy alone.
