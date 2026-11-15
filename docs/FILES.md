# Project file map

The default is pretrained extraction. The historical package name contains `mlx`, but the baseline uses PyTorch. Data, environments and generated outputs are excluded from this source-file inventory; see [USAGE](USAGE.md) for results.

| File | Purpose |
|---|---|
| `src/oct_dinov2_mlx/research_cli.py` | Bounded single-image inspect and LoRA/full fine-tuning commands. |
| `src/oct_dinov2_mlx/inspection.py` | Real attention capture, patch relationships, normalization comparison and checked inspection cache. |
| `src/oct_dinov2_mlx/dashboard.py` | Offline clickable HTML dashboard with embedded crop and arrays. |
| `src/oct_dinov2_mlx/lora.py` | MLX query/value low-rank adapters and native merged-weight export. |
| `tests/test_inspection.py`, `tests/test_lora.py` | Relationship/normalization semantics and adapter update/freeze/merge checks. |
| `tests/test_shell_wrapper.py` | Actual Bash environment selection and argument preservation, including paths with spaces. |
| `docs/RESEARCH_WORKFLOW.md` | End-to-end dashboard and adaptation research runbook. |
| `docs/RESEARCH_VALIDATION.md` | Local test evidence and limits for the research workflow. |
| `oct.sh` | Mac setup/test/run/show wrapper; manages .venv-dense without manual activation. |
| `oct.py` | Python entry point for the default frozen pretrained baseline. |
| `run_pipeline.py` | Earlier optional MLX adaptation/stage entry point; not called by the default baseline. |
| `requirements.txt` | Baseline + tests installation; no MLX dependency. |
| `requirements-mlx.txt` | Optional MLX experiment installation in a separate environment. |
| `requirements-legacy.txt` | Dependencies of independent original scripts. |
| `pyproject.toml` | Package metadata, baseline/MLX/test dependency groups and test markers. |
| `configs/smoke_test.yaml` | Small-run configuration; baseline CLI overrides selection to one image per class/split and ignores finetuning settings. |
| `configs/default.yaml` | Full dataset settings and Kaggle fallback; finetuning fields apply only to the optional MLX experiment. |
| `src/oct_dinov2_mlx/__init__.py` | Package version. The historical package name remains for compatibility. |
| `src/oct_dinov2_mlx/cli.py` | test/run/show commands, bounded baseline defaults, reusable sample export and error logging. |
| `src/oct_dinov2_mlx/pretrained.py` | Strict pretrained HF/PyTorch loading, CPU/MPS/CUDA selection, frozen extraction, caching and real-model self-test. |
| `src/oct_dinov2_mlx/backbone.py` | Pinned model revision/hash/schema and original MLX extractor; shared checkpoint validation is reused by the baseline. |
| `src/oct_dinov2_mlx/mlx_model.py` | Differentiable positional-embedding resizing for MLX CPU and Apple GPU; preserves native checkpoint names. |
| `src/oct_dinov2_mlx/pipeline.py` | Shared data preparation plus earlier MLX adaptation/extraction orchestration. |
| `src/oct_dinov2_mlx/config.py` | YAML validation and path/environment overrides. |
| `src/oct_dinov2_mlx/common.py` | Hashing, atomic JSON, environment records and optional MLX platform check. |
| `src/oct_dinov2_mlx/data.py` | Local/Kaggle resolution, records, lazy image loading and split integrity. |
| `src/oct_dinov2_mlx/preprocessing.py` | RGB, bicubic resize, center crop and normalization. |
| `src/oct_dinov2_mlx/cache.py` | Sharded arrays, integrity checks, locks, resume/reuse and reading samples. |
| `src/oct_dinov2_mlx/representations.py` | Single-scan arrays and crop-coordinate export. |
| `src/oct_dinov2_mlx/localization.py` | Patch-to-grid reshape and a future localization interface; not an implemented lesion method. |
| `src/oct_dinov2_mlx/adaptation.py` | Optional MLX adaptation training contract and checkpoint identity. |
| `src/oct_dinov2_mlx/augmentations.py` | Optional adaptation's paired crops and patch masks. |
| `src/oct_dinov2_mlx/finetuning.py` | Optional MLX teacher/student optimization, SSL heads, EMA and resume. |
| `scripts/export_dense.py` | Advanced export with explicit manifest/sample/output. |
| `scripts/show_features.py` | Low-level split-index feature preview. |
| `scripts/create_synthetic_smoke.py` | Creates explicitly artificial image fixtures. |
| `scripts/verify_parity.py` | Optional original MLX versus HF/PyTorch reference comparison; requires a separately resolved environment. |
| `scripts/learn_linear_algebra.py` | Executable rank/SVD/LoRA arithmetic examples; no data/model download. |
| `tests/conftest.py` | Artificial dataset fixture. |
| `tests/test_cli.py` | Convenience command dispatch, selection bounds, paths and errors. |
| `tests/test_pretrained.py` | Opt-in real pretrained extraction/export/reuse and batch/serialization checks. |
| `tests/test_data.py` | Dataset resolution, transforms and integrity. |
| `tests/test_cache.py` | Reuse, corruption, identities, interruption and locks. |
| `tests/test_dense_representation.py` | Export values/coordinates/dtypes and viewer behavior. |
| `tests/test_portable_pipeline.py` | Portable config, inspect and original checkpoint checks. |
| `tests/test_handoff.py` | Platform/config errors, relative caches and removed classifier stages. |
| `tests/test_adaptation.py` | Portable contracts, augmentations and earlier adaptation routing. |
| `tests/test_mlx.py` | Optional native MLX model and actual adapted integration; opt in with OCT_TEST_MLX. |
| `tests/test_mlx_positions.py` | Independent cubic interpolation/gradient reference and a regression test preventing dispatch to a Metal kernel without gradients. |
| `tests/test_mlx_finetuning.py` | Optional native gradient/loss/EMA/resume checks. |
| `.github/workflows/portable-tests.yml` | Windows/Linux offline portable regression tests. |
| `.github/workflows/baseline-tests.yml` | Real checkpoint inference and extraction tests on Windows/Linux/macOS CPU; Mac uses the user-facing setup wrapper. |
| `README.md` | Short starting guide. |
| `docs/USAGE.md` | Detailed current commands, output schema, platform options and troubleshooting. |
| `docs/FILES.md` | This file map. |
| `docs/STUDY_GUIDE.md` | Worked mathematics and the connection to weakly supervised OCT segmentation. |
| `docs/MEETING.md` | Meeting agenda and proposed controlled experiments. |
| `docs/MAC_RUNBOOK.md` | Pointer to the single current command sequence. |
| `docs/VALIDATION.md` | Actual execution evidence and historical records. |
| `docs/BASELINE_VALIDATION.json` | Machine-readable local real-model self-test evidence; not a segmentation result. |
| `docs/FINETUNING.md` | Earlier optional MLX adaptation objective and limits. |
| `docs/REPOSITORY_GUIDE.md` | Historical adaptation file/output guide, marked superseded for default commands. |
| `docs/RESEARCH_NOTES.md` | Earlier research investigation; use the study guide for the current experimental plan. |
| `docs/CHECKPOINT_AUDIT.json` | Original pretrained tensor audit. |
| `docs/DEPENDENCY_AUDIT.json` | Historical MLX dependency-resolution audit; not the current baseline lockfile. |
| `.gitignore` | Excludes data, results, environments, weights and common secrets. |
| `.gitattributes` | Cross-platform line endings. |
| `.gitmodules` | Upstream original DINO submodule URL. |
| `dataset_loader.py` | Independent earlier PyTorch dataset experiment. |
| `train_classifier.py` | Independent earlier ResNet classifier; not part of the current baseline. |
| `dino_oct_cam.py` | Independent earlier original-DINO/CAM experiment; not part of the baseline. |
| `dino` | Upstream DINO reference submodule, original commit/license; its own source tree is documented upstream. |
| `mlx_cpu.py` | Explicit native MLX CPU test/run/show entry point. |
| `requirements-mlx-cpu.txt` | Installs pinned MLX and the explicit Windows/Linux CPU backend in a separate environment. |
| `src/oct_dinov2_mlx/mlx_cpu_cli.py` | Real MLX self-test and bounded frozen extraction/viewing. |
| `tests/test_mlx_cpu.py` | Device policy, cache separation and bounded entry-point tests. |
| `.github/workflows/mlx-cpu-tests.yml` | Real CPU inference, gradients and adaptation checks on Windows and macOS. |
| `docs/MLX_CPU.md` | CPU installation and exact commands. |
| `docs/MLX_CPU_VALIDATION.json` | Actual Windows MLX CPU self-test and PyTorch numerical comparison. |
