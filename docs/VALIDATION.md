# MLX CPU validation — 2 November 2026

The earlier Apple-only restriction was incorrect for the current MLX release. Official MLX 0.32.4 and mlx-cpu 0.32.4 Windows wheels were installed in a dedicated environment. Runtime selection now supports CPU explicitly; the Mac Metal path remains selectable.

Executed **natively on this Windows AMD64 CPU / Python 3.14.5**, without WSL and without PyTorch inference in the MLX environment:

- MLX matrix multiplication and automatic differentiation passed.
- The actual MLX-VLM 0.7.6 DINOv2 adapter loaded the verified pinned pretrained checkpoint.
- Two-image real-model self-test passed: finite/nontrivial CLS and patches, batch-versus-single agreement and saved-array reload.
- Frozen MLX extraction completed for **12 real local OCT scans**, with no dataset download.
- Export produced CLS (384,), patches (256,384) and spatial grid (16,16,384).
- A repeated run reused all four cached samples per split without another extraction.
- **7 native MLX tests passed in 62.42 seconds**, including real pretrained inference, strict HF/native conversion, one-update backbone adaptation, final EMA save/reload, dense export/reuse, gradient/EMA/stop-gradient/masked-loss checks and optimizer/center/RNG resume.
- Comparing the same deterministic artificial inputs against the independently run PyTorch reference passed at atol=5e-4 and rtol=1e-3. Maximum absolute differences: CLS **2.635e-5**, patches **9.895e-5**. This is initialization/inference parity, not proof of identical adaptation trajectories.
- Complete suite in the MLX environment: **84 passed, 2 skipped in 103.36 seconds**. The skips are the separate PyTorch-baseline opt-in tests; the native MLX tests executed. The existing PyTorch feature cache was also rechecked and reused unchanged.
- `pip check` passed in the MLX environment. Machine-readable records: [MLX_CPU_VALIDATION.json](MLX_CPU_VALIDATION.json).

An installation attempt stalled after partial package installation; its specific cause was not established. Only that installer was stopped, and rerunning from cached packages completed successfully.

The default pytest file-descriptor capture caused a reproducible Windows DLL initialization failure for MLX, while the normal CLI worked. Using `--capture=sys` made the native suite pass. This setting is now in pyproject.toml; no numerical assertions or native tests were disabled.

Default commands remain the established PyTorch baseline. The explicit `mlx_cpu.py` commands use MLX CPU and separate output/cache identities. See [MLX_CPU.md](MLX_CPU.md) for exact setup/run/view commands. Full-dataset CPU throughput and clinical/segmentation quality are not established by these bounded checks.

---

# Current baseline validation — 2 November 2026

The default now uses **frozen pretrained DINOv2 through PyTorch 2.9.1 / Transformers 4.57.6**, on CPU unless another device is explicitly selected. The earlier professor traceback was unavailable: no specific original failure is diagnosed here. Replacing the default removes its dependency on unexecuted native MLX fine-tuning.

Executed locally on Windows AMD64 / Python 3.14.5:

- Installed the pinned engines in an isolated environment; `pip check` passed.
- Downloaded the exact ~89 MB pretrained model; existing revision, configuration and SHA-256 checks passed.
- `oct.py test` passed with **real weights** and two artificial images: output shapes, nontrivial finite features, batch-versus-single agreement and NPZ save/reload. Machine-readable evidence: [BASELINE_VALIDATION.json](BASELINE_VALIDATION.json).
- `oct.py run data/OCT2017 --offline` extracted **12 real local OCT scans** (4 per split), with no dataset download. Only this bounded selection was hashed/decoded; directory filenames were enumerated for counts.
- `oct.py show` exported real arrays with patches (256,384), grid (16,16,384) and CLS (384,). Local artifact: `outputs/pretrained-smoke/dense_sample.npz`. Generated data is ignored by Git.
- Repeated local extraction reported **extracted 0, reused 4** for every split; no further model construction was needed.
- Tests with `OCT_TEST_BASELINE=1`: **75 passed, 2 optional MLX modules skipped**. Real-checkpoint tests cover bounded synthetic dataset extraction, export, correct patch/grid indexing, repeat cache reuse and batch consistency. An initial test-only float16 standard-deviation warning was corrected by measuring that statistic in float32.
- Companion rank/SVD/LoRA numerical examples executed and assertions passed.


**Published code verification:** commit `f591e7daca75ff80e97ba2a05199efb9098d598b` passed the [real pretrained workflow](https://github.com/arpitmisra/mini-project/actions/runs/41374653197) on **macOS, Windows and Linux CPU**. All three jobs completed successfully. The macOS job installed through `bash oct.sh setup`, ran `bash oct.sh test`, and ran the full suite with the actual baseline integration tests enabled. This is actual Mac-runner execution, not a dependency dry run or an inference from Windows results. MPS acceleration and the professor's particular machine remain unverified. The subsequent documentation-only commit does not change the tested code.


These checks do **not** establish segmentation accuracy, improvement from adaptation, patient independence, full-dataset runtime, or MPS/GPU correctness. CPU execution on a Mac runner also does not establish that the professor's particular installation has passed. The quick-start self-test is the local acceptance check.

The default baseline has no trainable classifier, adapter or segmentation decoder. Optional MLX tests require their own environment and `OCT_TEST_MLX=1`. The older records below refer to previous workflows.

---

# Simple run/show commands — 25 October 2026

The professor-facing workflow now uses `python oct.py run "/path/to/OCT2017"` and `python oct.py show`. The wrapper delegates adaptation/extraction to the existing pipeline. Smoke selection remains bounded; only `--full` selects the full experiment and permits dataset fallback. Viewing exports one scan without MLX or original images and checks the export's identity/checksum before reuse.

Executed on Windows: **73 passed, 2 skipped in 26.15 seconds**. The two skips remain whole native MLX modules. New checks cover smoke/full dispatch, paths with spaces, environment path precedence, offline mode, actionable errors, repeat viewing without rewriting, corrupted/stale exports, choosing a different scan, and running the viewer from another working directory after original fixture images are removed. These tests use synthetic arrays; no real adapted representations are claimed.

The README now provides the ordered install/test/smoke/view/full flow, every project file's purpose and exact output locations. The compact Mac runbook uses the same commands. Native execution still requires the Mac tests below.

---

# Self-supervised adaptation validation — 25 October 2026

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
