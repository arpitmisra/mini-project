# Repository guide: self-supervised adaptation and dense features

The current objective is **fine-tune DINOv2 on OCT training images, then generate dense representations only**. It replaces the earlier frozen-backbone plus linear-classifier workflow. The current entry point never trains a diagnostic classifier and has no classification evaluation/prediction stages.

## Execution flow

1. Resolve an existing local Kermany dataset; use KaggleHub fallback only when enabled and the local path is absent.
2. Select and audit train/val/test records. Initial smoke tests select at most 48 images and prohibit dataset downloads.
3. Initialize student and teacher from the verified pretrained DINOv2 ViT-S/14 checkpoint.
4. Fine-tune the entire student backbone on original TRAIN images using cross-view global self-distillation and aligned masked-patch self-distillation. EMA updates the teacher. Diagnosis labels never enter the loss.
5. Save the final EMA teacher backbone and verify a strict save/reload round trip. Temporary SSL projection heads remain only in resume states, not in the exported backbone.
6. Freeze that adapted backbone and extract deterministic features for the selected train/val/test images.
7. Cache arrays under identities containing the adapted weight hash; optionally export one complete scan for inspection.

`all` is the default stage and performs this flow. `finetune` stops after the adapted checkpoint; `extract` requires an existing adapted checkpoint. `inspect` is a bounded offline data/preprocessing check. `system` reports environment details. [Mac commands](MAC_RUNBOOK.md) and [objective details](FINETUNING.md) are documented separately.

## Meaning of the arrays

At input size 224 and patch size 14, the grid has 16 rows and 16 columns. The backbone returns a global CLS vector `[384]` plus patch descriptors `[256,384]`. Reshaping the patch descriptors produces `[16,16,384]`, or 98,304 spatial feature values per scan. For batch size B the shapes are `[B,384]` and `[B,256,384]`.

Tokens are row-major: token index = row × 16 + column. `grid[7,11]` is the same vector as `patches[123]`. Its input anchor is x=[154,168), y=[98,112) in the preprocessed center crop. Self-attention allows the descriptor to include global context. It is not a probability, a named anatomical feature, a segmentation label or a validated activation map.

Extraction uses RGB conversion, bicubic shortest-edge resize to 256, center crop to 224 and ImageNet normalization in NHWC float32. The crop can omit peripheral anatomy. The output is final LayerNorm features; they are not additionally L2-normalized. Default float16 patch storage rounds float32 inference values; `patch_dtype: float32` retains those values at twice the patch-storage cost.

## Source files

| File | Role |
|---|---|
| `run_pipeline.py` | Main CLI entry point |
| `configs/default.yaml` | Full-data adaptation/extraction settings |
| `configs/smoke_test.yaml` | 48-image selection, one epoch capped at two updates, no dataset download |
| `requirements.txt` | Portable dependencies plus conditional native Mac dependencies |
| `pyproject.toml` | Package metadata, dependency groups and test configuration |
| `src/oct_dinov2_mlx/__init__.py` | Package initialization/version |
| `config.py` | Configuration merge, path precedence and strict value checks |
| `common.py` | Hashes, JSON writes, system information and native platform checks |
| `data.py` | Local/KaggleHub resolution, records, lazy images and split integrity |
| `preprocessing.py` | Shared resize/normalization and deterministic extraction transform |
| `augmentations.py` | Aligned teacher/student crop pairs, photometric changes and patch masks |
| `backbone.py` | Pinned original checkpoint validation, native model creation and frozen extraction from original/adapted weights |
| `adaptation.py` | Portable training contract and adapted checkpoint/hash validation |
| `finetuning.py` | SSL projection heads, objective, student optimization, teacher EMA, epoch resume and verified final backbone export |
| `pipeline.py` | Fine-tune → extract orchestration and run metadata; no classifier |
| `cache.py` | Shards, checksums, restart/reuse and reading cached arrays |
| `localization.py` | Patch-to-grid reshape and future localization interface; no implemented lesion method |
| `representations.py` | Standalone dense export and crop-coordinate metadata |
| `scripts/export_dense.py` | Export command accepting a feature manifest and sample/split |
| `scripts/show_features.py` | Print feature shape/value summaries from a split index |
| `scripts/create_synthetic_smoke.py` | Explicitly artificial fixture for plumbing tests |
| `scripts/verify_parity.py` | Original pretrained MLX-versus-HF reference comparison; not a comparison of adapted and original weights |

Module names after `__init__.py` in the table refer to `src/oct_dinov2_mlx/` unless a full relative path is given.

## Tests, documentation and Git files

| File | Role |
|---|---|
| `tests/conftest.py` | Artificial dataset fixture |
| `test_data.py` | Local/fallback behavior, labels, transforms and split integrity |
| `test_cache.py` | Reuse, corruption repair, invalidation, interruption and locks |
| `test_dense_representation.py` | Export values, dtypes, spatial order and coordinates |
| `test_portable_pipeline.py` | Bounded inspection, config and original checkpoint schema/hash checks |
| `test_handoff.py` | Platform checks, invalid settings, relative manifests and rejection of removed classifier stages |
| `test_adaptation.py` | Train-only contract, paired views/masks, adapted identity and default fine-tune/extract routing |
| `test_mlx.py` | Native model shape/weight checks and optional actual pretrained adaptation/extraction integration |
| `test_mlx_finetuning.py` | Native loss math, backbone gradients, teacher EMA, stop-gradient and optimizer/center/RNG resume |
| `.github/workflows/portable-tests.yml` | Windows/Ubuntu offline portable CI; no Apple GPU validation |
| `README.md` | Current setup, single-command workflow, export, reuse and troubleshooting |
| `docs/MAC_RUNBOOK.md` | Exact ordered execution commands |
| `docs/FINETUNING.md` | Mathematical objective, source references, protocol and limitations |
| `docs/RESEARCH_NOTES.md` | Current method reference plus historical backbone investigation |
| `docs/VALIDATION.md` | Current verification record and clearly historical earlier results |
| `docs/CHECKPOINT_AUDIT.json` | Original pretrained tensor-header audit |
| `docs/DEPENDENCY_AUDIT.json` | Historical Mac-targeted dependency-resolution record; not a lockfile or native execution proof |
| `docs/REPOSITORY_GUIDE.md` | This guide |
| `.gitignore` | Excludes data, environments, weights, output caches and common credential files |
| `.gitattributes` | Cross-platform text line endings |
| `.gitmodules` | Upstream original-DINO submodule location |

The root `dataset_loader.py`, `train_classifier.py` (ResNet-18), `dino_oct_cam.py` (original DINO experiment), `requirements-legacy.txt` and `dino/` submodule are earlier independent experiments. They are not called by the current entry point. The submodule retains its upstream license. The former active package modules `training.py` and `evaluation.py` were removed; they are recoverable in Git history. No old classifier outputs are deleted from a user's local workspace.

## Exact output artifacts

Smoke outputs use `outputs/smoke-adapted/`; full default outputs use `outputs/oct-adapted/`.

```text
outputs/smoke-adapted/
  config.resolved.json
  dataset_summary.json
  finetuning/
    backbone.safetensors           # final EMA backbone, no projection heads
    backbone.json                  # completed run contract, hash, architecture, history
    history.json                   # SSL diagnostics per completed epoch
    latest.json                    # pointer to last complete epoch state
    epoch-0001/
      state.safetensors            # student, teacher, heads, optimizer, centers
      state.json                   # hash, epoch/step, history and RNG state
  features_manifest.json
  extraction_benchmark.json
  experiment_metadata.json
  extraction_config.resolved.json  # when extract is run separately
  extraction_dataset_summary.json # preserves the training dataset summary
  extraction_metadata.json        # when extract is run separately
  dense_sample.npz                # when export_dense is invoked
  dense_sample.json
outputs/features/<adapted-identity>/<split>/
  index-<selection>.json
  <shard-hash>.npz
  <shard-hash>.json
```

`inspect` creates `inspection_dataset_summary.json`, `inspection_config.resolved.json` and `portable_smoke.json` without model work. `finetune` creates adaptation artifacts without feature shards. `extract` records its own config/metadata and preserves fine-tuning config provenance. No classification predictions, accuracy/AUC figures, confusion matrices or linear-head checkpoints are generated by the current flow.

Each shard stores `cls [N,384]`, `patches [N,256,384]`, labels, sample IDs and original diagnoses. The labels remain provenance; they have not supervised fine-tuning. Each index records original paths, splits and content hashes. Shards are bounded (128 images full default; 8 smoke). No full-dataset dense tensor is assembled in memory.

A standalone export contains `cls`, `patches`, `spatial_grid`, `patch_row_col`, `patch_bounds_xyxy`, `sample_id`, `original_class` and `binary_label`. Its JSON includes model/adaptation identity, source record, coordinate conventions, shapes, dtypes and checksum. Loading an export requires NumPy only and can be done on Windows. It does not require a classifier, MLX, the original images or the model weights.

## Cache and resume rules

The fine-tuning contract includes the original backbone, selected training samples, preprocessing, settings, seed and implementation versions. The settings are snapshotted. A changed contract requires a new output directory. A completed matching model is reused rather than optimized again. Partial runs restart after the last completed epoch with restored optimizer, teacher, centers and RNG; an incomplete epoch is repeated.

Dense cache identities include the actual adapted weights' SHA-256 and contract, so neither another fine-tuned model nor the original pretrained model can silently reuse the same features. Storage changes can also require new shards. Completed extraction shards are reused after interruption. Verification reads cached files and sample hashes; valid hits avoid backbone construction and image decoding.

A full float16 patch cache for the indexed 84,484-image copy needs about 15.5 GiB before overhead; float32 requires about 30.9 GiB. CLS values add 1,536 bytes/image. Resume checkpoints and the dataset require additional space. All completed epoch states are retained, so their storage increases with the number of epochs.

## Scientific and verification boundaries

This is a compact DINO/iBOT-style self-supervised adaptation, not the full DINOv2 training recipe. There is no test-based model selection or claim that a low SSL loss proves useful medical features. Patient-independent splits have not been established; crop coverage, near duplicates, collapse and external representation quality require evaluation.

Portable tests can establish control flow, metadata and cache behavior. Native gradient/EMA/serialization tests and the actual pretrained integration remain for the Mac. The original checkpoint schema/parity checks validate initialization; they cannot certify fine-tuning quality. See the current validation record for precisely what has been executed.
