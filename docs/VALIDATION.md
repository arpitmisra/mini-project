# Validation record

Development host: Windows 11 AMD64, Python 3.14.5. Date: 19 October 2026 (Asia/Calcutta).

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
