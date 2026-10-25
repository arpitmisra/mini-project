# Mac checklist: adapt DINOv2, then save dense representations

The [README](../README.md) is the complete guide, including every project file, result location and troubleshooting. This page gives the commands in order. Run them from the cloned repository, using native arm64 Python on a compatible Apple Silicon Mac.

## 1. One-time installation

```bash
git clone https://github.com/arpitmisra/mini-project.git
cd mini-project
python3 -c "import platform; print(platform.machine())"
```

Confirm `arm64`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Python 3.12 is the documented target. Use macOS 14 or newer, subject to the installed MLX wheel's minimum. For an existing clone, activate its environment, pull updates with `git pull --ff-only`, and reinstall requirements.

## 2. Test the installation

```bash
python -m pytest -q
OCT_TEST_PRETRAINED=1 python -m pytest -q -m pretrained
```

Stop if tests fail. The first command includes native gradient/loss/EMA/resume tests on a compatible Mac. The second enables actual pretrained integration tests, including tiny adaptation, dense export and reuse. It may download the pinned model (~89 MB), but neither command downloads the OCT dataset. Test images are artificial.

## 3. Run a small check on local OCT images

Replace the path with the actual folder containing train/val/test:

```bash
python oct.py run "$HOME/Downloads/OCT2017"
```

This selects at most 48 scans, trains on TRAIN only for two updates, then extracts dense features. It never downloads the dataset. The two-update result verifies software behavior; it is not a research-quality adaptation experiment.

## 4. See and export the arrays

```bash
python oct.py show
```

Expected shapes: `cls (384,)`, `patches (256,384)`, `spatial_grid (16,16,384)`.

The command prints the source scan and file locations and creates:

```text
outputs/smoke-adapted/dense_sample.npz
outputs/smoke-adapted/dense_sample.json
```

Every value is stored. To see all values in Terminal, use `python oct.py show --all-values`. Repeating `show` reuses the verified export without model execution.

## 5. Check reuse

```bash
python oct.py run "$HOME/Downloads/OCT2017" --offline
```

With unchanged settings/data and a completed first run, the adapted checkpoint is reused. Expect `extracted 0, reused 16` for each split when the smoke selection contains 48 scans. Hash/cache checks still read files.

## 6. Run the full experiment

After the tests and smoke run pass:

```bash
python oct.py run "$HOME/Downloads/OCT2017" --full
python oct.py show --full
```

This uses all training images for five adaptation epochs, then extracts train/val/test features. Results use `outputs/oct-adapted/`; the single-scan export is `outputs/oct-adapted/dense_sample.npz`. Five epochs is a starting setting, not a validated optimum.

If no local dataset exists and you deliberately want the full KaggleHub download, use `python oct.py run --full`. It tries `OCT_DATASET_PATH` or the configured local directory first. Full mode permits KaggleHub fallback if that path is unavailable. Allow dataset/archive space, approximately 15.5 GiB for float16 patches at 84,484 scans, and additional space for training checkpoints.

For a tiny artificial Mac run before downloading OCT data, see the README's optional commands. Never report artificial features as OCT results.

## 7. Find or share all results

- `outputs/<run>/features_manifest.json`: links to the split indices.
- `outputs/features/<adapted-identity>/<split>/`: all cached arrays and image records.
- `outputs/<run>/finetuning/backbone.safetensors` and `backbone.json`: adapted EMA backbone and provenance.
- `outputs/<run>/finetuning/history.json`: self-supervised diagnostics.
- `outputs/<run>/finetuning/epoch-XXXX/`: resumable training states.
- `outputs/<run>/dense_sample.npz/.json`: one scan exported by `show`.

Here `<run>` is `smoke-adapted` or `oct-adapted`. The README lists every additional metadata file. Copy the entire `outputs/` layout to preserve manifest links. To share just one representation, send the single-scan export pair.

## 8. Resume or start a different experiment

After interruption, rerun the same command. Training resumes after the last complete epoch; incomplete epochs are replayed. Feature extraction reuses valid shards. Investigate any stale writer lock before removing it.

Changed training settings/data require a new run directory:

```bash
python oct.py run "$HOME/Downloads/OCT2017" --full --output-dir outputs/experiment-02
python oct.py show --full --output-dir outputs/experiment-02
```

In a new Terminal, return to the repository and reactivate the environment:

```bash
cd "/absolute/path/to/mini-project"
source .venv/bin/activate
```

Native fine-tuning has not been executed on the Windows development host. Successful optimization does not establish representation quality or clinical performance. The output is a learned patch-feature tensor in the preprocessed crop, not a segmentation or class prediction.
