# MLX on a Windows CPU

MLX 0.32.4 publishes native Windows x64/ARM64 wheels and a separate CPU backend. The repository previously rejected Windows and forced Metal; that restriction has been removed. This page documents the explicit CPU route. The PyTorch baseline remains available in its separate environment.

Primary package references: [MLX 0.32.4](https://pypi.org/project/mlx/0.32.4/#files), [MLX CPU 0.32.4](https://pypi.org/project/mlx-cpu/0.32.4/#files).
For this release, the `mlx[cpu]` extra applies to Linux only. Windows needs both `mlx==0.32.4` and `mlx-cpu==0.32.4`; the supplied requirements include them.

## Fresh Windows installation

From PowerShell in the repository:

```powershell
python -m venv .venv-mlx-cpu
.venv-mlx-cpu\Scripts\python.exe -m pip install -r requirements-mlx-cpu.txt
.venv-mlx-cpu\Scripts\python.exe mlx_cpu.py test
```

This environment is separate because MLX-VLM and the PyTorch baseline require different Transformers versions. Do not install both requirement sets into one environment.

The test runs MLX CPU multiplication and gradients, loads the exact pretrained DINOv2 checkpoint, compares batched and single-image features, and checks saved-array reload. It uses two artificial images, downloads no OCT dataset, and can download the model once if its cache is missing.

## Extract a small local OCT sample, then view it

```powershell
.venv-mlx-cpu\Scripts\python.exe mlx_cpu.py run "C:\path\to\OCT2017"
.venv-mlx-cpu\Scripts\python.exe mlx_cpu.py show
```

This uses the **MLX DINOv2 implementation**, not PyTorch inference. It selects at most 12 scans, prohibits dataset downloading, and performs no fine-tuning. Add `--offline` after model weights have been cached to prohibit all fetching.

Results:

- `outputs/mlx-cpu-self-test/self_test.json`: real-model self-test evidence.
- `outputs/mlx-cpu-smoke/features_manifest.json`: links to all selected feature shards.
- `outputs/mlx-cpu-smoke/dense_sample.npz/.json`: single-scan export from show.
- `outputs/features/`: cached arrays, separated by model/runtime/device identity.
- `outputs/mlx-cpu-last-error.log`: diagnostic traceback if the CLI fails.

Use `--full` with run and show only after the small check passes. Full mode uses `outputs/mlx-cpu/` and permits KaggleHub fallback for an unavailable local dataset. `--output-dir` selects a custom run folder. CPU full-dataset throughput has not been benchmarked.

The tested environment created during development is `outputs/mlx-cpu-env`; it can be used directly on this PC:

```powershell
outputs\mlx-cpu-env\Scripts\python.exe mlx_cpu.py test --offline
outputs\mlx-cpu-env\Scripts\python.exe mlx_cpu.py run data/OCT2017 --offline
outputs\mlx-cpu-env\Scripts\python.exe mlx_cpu.py show
```

## CPU execution on the professor's Mac

In a separate environment:

```bash
python3 -m venv .venv-mlx-cpu
.venv-mlx-cpu/bin/python -m pip install -r requirements-mlx-cpu.txt
.venv-mlx-cpu/bin/python mlx_cpu.py test
.venv-mlx-cpu/bin/python mlx_cpu.py run "/path/to/OCT2017"
.venv-mlx-cpu/bin/python mlx_cpu.py show
```

The explicit entry point always selects CPU. The older `run_pipeline.py` defaults to Metal on Apple Silicon and CPU elsewhere; `OCT_MLX_DEVICE=cpu` overrides it. Installing a CPU package alone does not select the device in code.

The Windows run completed: real DINOv2 inference, 12 local OCT scans, export/reuse, and seven native MLX tests passed. The two-image output comparison against PyTorch passed at atol=5e-4 / rtol=1e-3. See [the detailed evidence](VALIDATION.md).

## Optional training checks

The native tests now run on supported MLX CPUs as well as Metal; they no longer skip merely because the host is Windows. To opt in:

```powershell
$env:OCT_MLX_DEVICE = "cpu"
$env:OCT_TEST_MLX = "1"
$env:OCT_TEST_PRETRAINED = "1"
.venv-mlx-cpu\Scripts\python.exe -m pytest -q tests/test_mlx.py tests/test_mlx_finetuning.py
```

These include native gradients, EMA, optimizer/center/RNG resume and a one-update real DINOv2 adaptation/extraction/export/reuse check on artificial fixtures. They do not download or train on the full OCT dataset.

CPU/GPU compatibility does not imply identical floating-point values or equal speed. Keep the model, preprocessing and numerical tolerance fixed when comparing implementations. Read [VALIDATION.md](VALIDATION.md) for actual execution results.

The project selects pytest system-level capture (`--capture=sys`), because the default file-descriptor capture caused a Windows MLX DLL initialization error. All test assertions remain enabled.
