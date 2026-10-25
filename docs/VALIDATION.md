# Current validation: self-supervised adaptation — 25 October 2026

The active flow changed to native MLX teacher–student fine-tuning on original training images, then adapted dense extraction only. Classifier stages and active classifier/evaluation modules were removed. The final EMA backbone has its own checksum and complete provenance, and the dense cache includes that adapted identity.

Executed on Windows 11 AMD64 / Python 3.14.5:

- Portable regression suite: **63 passed, 2 skipped in 25.06 seconds**. Both skipped items are whole native MLX test modules. Fixtures and mocked model outputs are explicitly artificial, not real DINOv2 representations.
- Covered new train-only eligibility, immutable training contracts, adapted weight/contract checksum validation, cache identity changes across adapted models, deterministic aligned augmentation, bounded patch masks, missing-adaptation rejection before dataset access, bounded/offline inspection through the Python API, default fine-tune-then-extract routing, reuse without another extractor, and preservation of prior run metadata when an incompatible fine-tuning request is rejected.
- Syntax compilation and CLI help checks passed. The CLI now exposes only `all`, `finetune`, `extract`, `inspect` and `system`.
- Real local offline inspection passed with **48 decoded images total** (16 per split). Record: `outputs/adaptation-review-inspect/portable_smoke.json`. Full directory counts were enumerated, but no full-data image processing or dataset/model download occurred.
- A first test attempt could not create pytest's sandbox temporary directory. Tests were rerun with a fresh temporary directory inside ignored workspace `outputs/`. One regression then exposed a mutable settings reference in the training contract; it was fixed with a deep snapshot before the passing run.

Source/API review used the already downloaded pinned MLX-VLM/MLX wheels, including masked `forward_features`, recursive unfreeze/freeze, native save/load, optimizer state and tree flatten/unflatten. The global and patch distillation design was checked against Meta's primary DINOv2 loss/architecture sources. This is API/source review, not native training execution.

Provided but **not executed on this Windows host**:

- Native masked-loss/cross-view math against a NumPy reference, backbone parameter updates, teacher EMA and stop-gradient assertions.
- Full student/teacher/head/optimizer/center/RNG epoch-state save/restore through the next optimizer step.
- Actual pretrained fine-tuning, final EMA backbone strict save/reload, adapted loading without original HF weights, dense export and repeated-run reuse.
- Base-model numerical parity on the Mac. This checks initialization only; no full fine-tuning-procedure parity with an independent implementation is claimed.

No successful native fine-tuning run, real adapted OCT features, improved representation quality, throughput or clinical performance is claimed. The Mac test instructions are in `MAC_RUNBOOK.md`. Full dataset leakage/patient-independence and representation-quality studies remain separate research work.

---

# Historical validation records (superseded workflow)

The records below describe earlier frozen-backbone/linear-probe versions. Their classifier tests and commands are not the current pipeline.

# Validation record

Development host: Windows 11 AMD64, Python 3.14.5. Date: 19 October 2026 (Asia/Calcutta).

## Dense representation review — 22 October 2026

The requested diagram was checked against the source, including token shapes, pretrained weight loading, normalization, patch ordering, preprocessing geometry, cache persistence and the separate classifier branch. The actual pinned MLX-VLM wheel confirms row-major patch flattening and separate final normalized CLS/patch outputs.

The full configuration previously saved CLS only. Both configurations now persist dense patch features by default. A portable export utility writes a scan's complete CLS vector, patch matrix, spatial grid, crop-relative patch coordinates, labels and provenance. It verifies only the selected shard, avoiding a full dense-cache scan for one export. Existing CLS-only caches require new extraction to obtain spatial features; their data are not overwritten.

Executed for this revision:

- `python -m pytest -q --tb=short`: **49 passed, 1 skipped in 24.72 seconds**. The skipped item remains the native MLX module.
- New tests check exact cached/exported values, float16 persistence, row/column orientation, crop bounds, handling of missing/CLS-only samples, overwrite refusal, selected-shard corruption detection and the export CLI from another working directory. These use explicitly synthetic feature arrays, not real pretrained outputs.
- Syntax compilation and `git diff --check` passed.
- Real local `--stage inspect --offline` passed for 48 selected OCT images, writing its record to `outputs/dense-review-inspect/portable_smoke.json`. No full image-processing or download run was performed.
- Mac pretrained tests were extended to check patch batch consistency and export a dense sample after the integration run. These additional native checks have not been executed here.

The README and `REPOSITORY_GUIDE.md` now distinguish the dense deliverable, optional CLS-based classification, crop coverage, float16 rounding, exact output files and the remaining native/scientific validation requirements. No real OCT feature values or classification performance are claimed from these portable checks.

## Handoff review — 22 October 2026

The source was reviewed again before the first GitHub publication. The final local regression run for this review completed with **42 passed, 1 skipped in 19.82 seconds**. The skipped item is the complete native MLX test module. Syntax compilation and the real local 48-image inspection smoke test also passed again. No full dataset was downloaded or processed.

Issues corrected during review:

- Extract-only runs now check Apple Silicon/Metal support before dataset retrieval, avoiding an unintended large fallback download on unsupported hosts.
- Feature manifest paths are relative where possible, allowing an entire `outputs/` tree to move to another machine without rewriting each index. Absolute legacy paths remain supported.
- Evaluation saves separate metadata/config files and retains the checkpoint's training configuration/seed, rather than overwriting training provenance.
- Invalid YAML types, nonfinite hyperparameters, fractional image dimensions and invalid dataset versions fail before model work.
- An unavailable optional hardware-description query no longer aborts a run; missing MLX-VLM reports an installation instruction. Broken native Mac test setups fail rather than silently skipping.

Additional checks:

- Inspected the actual PyPI `mlx-vlm==0.7.6` wheel's DINOv2 configuration, forward feature API and weight sanitizer, and MLX 0.32.2's save/load/evaluation/memory APIs.
- Resolved all project runtime/test dependencies for Python 3.12, macOS 14 arm64, with a pip `--dry-run` and the matching `mlx-metal` package explicitly included. Resolution succeeded with MLX/MLX-Metal 0.32.4 and MLX-VLM 0.7.6. Cross-platform resolver markers can still depend on the host, so this is not a Mac installation test or a complete lockfile.
- Added an offline native Mac test of all HF-to-MLX parameter mappings, including Q/K/V ordering. This test is provided for the Mac and was not executed on Windows.
- Added GitHub Actions portable tests on Windows and Ubuntu. Hosted-run results are visible in the repository's Actions tab; they do not certify Metal execution.

The README now includes the exact GitHub clone command, native arm64 setup, dependency checks, local/synthetic smoke commands, cached training, evaluation, parity testing and troubleshooting. The repository excludes dataset images, virtual environments, credentials, model weights and generated output caches.

The original 19 October validation record is retained below.

No Apple Silicon device is available in this environment. No full dataset download, full model weight download, full-data image decoding, or real DINOv2 inference was performed during development.

## Executed checks

- Syntax compilation of `src/`, `tests/`, `scripts/`, and `run_pipeline.py`.
- Portable dependency imports: NumPy, Pillow, scikit-learn, pytest, PyYAML.
- `python -m pytest -q --tb=short`: **25 passed, 1 skipped in 5.35 seconds** on the final run. The skipped item is the native MLX test module, which contains the Mac-only tests. The successful run required access to pytest's temporary directory outside the development sandbox.
- `python run_pipeline.py --config configs/smoke_test.yaml --stage inspect --offline`: passed on the existing local `data/OCT2017`. It opened and preprocessed 16 selected images per split, 48 total. No network fallback was invoked.
- The actual remote safetensors header was read with a bounded 64 KiB HTTP range request (206 Partial Content). All 223 tensor names and shapes passed strict validation, totaling 22,056,576 float32 parameters. The published LFS checksum is pinned for verification when the full checkpoint is first loaded. See `CHECKPOINT_AUDIT.json`. This is a schema check, not numerical model verification.

The initial pytest invocation was blocked by the environment's temporary-directory sandbox permissions. Rerunning outside that sandbox succeeded. This was an execution-environment issue, not a dataset/test failure.

## Observed local directory counts

| Split | NORMAL | CNV | DME | DRUSEN | Total |
|---|---:|---:|---:|---:|---:|
| Train | 26,315 | 37,205 | 11,348 | 8,616 | 83,484 |
| Validation | 8 | 8 | 8 | 8 | 32 |
| Test | 242 | 242 | 242 | 242 | 968 |

Counts came from supported image filenames, not fabricated dataset statistics. Only the 48 smoke images were decoded and content-hashed. The smoke split integrity check covers those selected images and found no exact cross-split byte duplicates. It does not establish full-dataset or patient-level independence.

## Test coverage

Portable tests exercise local override without calling KaggleHub, unavailable-path fallback with a stub, download/authentication errors, no-download mode, nested and ambiguous roots, missing classes, lazy image reading, diagnosis mapping, deterministic RGB preprocessing, content duplicate rejection, optional reproducible train-derived validation, feature round trips, no model/image access on cache hits, content/model invalidation, corrupt shard repair, interrupted-run resume, writer exclusion, checkpoint schema rejection, source-weight fingerprint rejection, metrics edge cases, plots and bounded pipeline inspection.

Mac tests exist for real MLX differentiation, weight changes and decreasing synthetic loss, frozen backbone parameters, batch shapes, save/reload equivalence, optional real-pretrained extraction and end-to-end smoke with cache reuse. `scripts/verify_parity.py` compares native MLX output to the HF PyTorch reference for identical pixels and weights, saving numerical errors and failing on tolerance violations.

## Still requiring execution

- Native arm64 dependency installation and MLX tests on the professor's Mac.
- Real checkpoint strict loading, finite/nontrivial feature values and numerical parity.
- Real OCT smoke inference, classifier optimization and checkpoint restoration.
- KaggleHub network download/authentication/cache reuse (stub-tested only; intentionally no multi-GB download here).
- Full dataset leakage audit, trustworthy patient grouping and full research evaluation.

No accuracy, AUC, runtime throughput or memory figure for DINOv2 on real OCT images is reported. Synthetic test metrics are assertions about software behavior, not research results.
