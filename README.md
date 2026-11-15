# OCT scans → DINOv2 features and patch dashboard

Extract spatial features from OCT scans, inspect patch attention, and compare **self-supervised LoRA versus full fine-tuning**. The program saves feature arrays and visualizations. It does not train a disease classifier or produce validated lesion masks.

## 1. Set up on your Mac — once

Use **Python 3.12** and Git. For MLX fine-tuning, use native Python on an **Apple Silicon Mac (M1 or newer)**. Open Terminal and run:

```bash
git clone https://github.com/arpitmisra/mini-project.git
cd mini-project
bash oct.sh setup
bash oct.sh test
```

Wait for **PASS**. The first test downloads the pretrained model (~89 MB), then checks two artificial images. **It does not download the OCT dataset.** No environment activation is needed.

Already cloned? Run `git pull --ff-only` inside `mini-project`, then repeat setup and test. If `python3` is not Python 3.12, run setup as `PYTHON=python3.12 bash oct.sh setup`.

## 2. Open your first dashboard — no dataset needed

```bash
bash oct.sh inspect --output-dir outputs/noise-check
open outputs/noise-check/dashboard.html
```

This uses random noise as a sanity check. Click a patch and try the attention-head, similarity and normalization controls. Structured-looking output on noise is not evidence of disease detection.

## 3. Inspect a real image

Replace `/path/to/scan.jpeg` with an actual image file. Keep the quotes:

```bash
bash oct.sh inspect "/path/to/scan.jpeg" --output-dir outputs/my-scan
open outputs/my-scan/dashboard.html
```

The dashboard shows **attention weights**, **cosine similarity** and **Pearson correlation** separately. Its normalization control compares pretrained mean/std normalization with RGB / 255 only. Use a new output folder for a different image or settings; identical runs reuse saved results.

## 4. Extract features from your local OCT dataset

Replace the example folder with your extracted dataset folder. It should contain `train`, `val` and `test`, with `NORMAL`, `CNV`, `DME` and `DRUSEN` class folders.

```bash
bash oct.sh run "$HOME/Downloads/OCT2017"
bash oct.sh show
```

This processes **at most 12 scans**, uses your local dataset, and reuses valid features on repeat runs. `show` exports one scan and prints its arrays.

Each scan has `patches (256,384)`, `spatial_grid (16,16,384)` and `cls (384,)`. The spatial grid covers the **224×224 preprocessed center crop**. All values are saved even when Terminal abbreviates the display.

## 5. Test LoRA and full fine-tuning — optional

Install the separate MLX environment once, then run both small checks:

```bash
bash oct.sh setup-mlx
bash oct.sh finetune "$HOME/Downloads/OCT2017" --mode lora
bash oct.sh finetune "$HOME/Downloads/OCT2017" --mode full
bash oct.sh show --output-dir outputs/lora-imagenet-smoke
```

Both use CPU by default, the same self-supervised objective, and training images only. Each runs **one training step**, then extracts features for at most 12 scans. These are setup checks, not completed research experiments. No dataset download occurs.

To repeat with mean/std standardization disabled:

```bash
bash oct.sh finetune "$HOME/Downloads/OCT2017" --mode lora --normalization unit
bash oct.sh finetune "$HOME/Downloads/OCT2017" --mode full --normalization unit
```

`unit` still scales pixels to [0,1]. Normalization is applied after augmentation and before both teacher and student. Full experiment commands, adapted-model dashboards and the comparison protocol are in the [research runbook](docs/RESEARCH_WORKFLOW.md).

## Where are my results?

| What you ran | What to open |
|---|---|
| `inspect ... --output-dir outputs/my-scan` | `outputs/my-scan/dashboard.html` |
| Same inspection: exact arrays and comparison | `outputs/my-scan/arrays.npz` and `report.json` |
| `run` followed by `show` | `outputs/pretrained-smoke/dense_sample.npz` and `dense_sample.json` |
| All selected scans' feature locations | `outputs/pretrained-smoke/features_manifest.json` |
| LoRA checkpoint and training history | `outputs/lora-imagenet-smoke/finetuning/` |
| Full fine-tuning checkpoint and history | `outputs/full-imagenet-smoke/finetuning/` |

Normalization-ablation folders replace `imagenet` with `unit`. Actual dataset feature shards are in `outputs/features/`. Original images and generated outputs are not included in GitHub.

## Help and further experiments

- **Windows commands, array loading, attention definitions, normalization code and full experiments:** [Research runbook](docs/RESEARCH_WORKFLOW.md).
- **What every file does:** [File map](docs/FILES.md).
- **What was actually tested:** [Validation evidence](docs/VALIDATION.md).
- **Mathematics and research background:** [Study guide](docs/STUDY_GUIDE.md).
- **A command failed?** Send the command and `outputs/research-error.log` for `inspect`/`finetune`, or `outputs/last-error.log` for `test`/`run`/`show`. For setup errors, copy Terminal's error message.
- **Updating an older MLX experiment?** Add a new folder, for example `--output-dir outputs/lora-v3`. Old checkpoints cannot resume across the Mac fine-tuning correction; your old files are kept.

Start with the small checks above. Full-dataset runs require `--full`; they may download the dataset through KaggleHub if the local path is unavailable. Add `--offline` when both dataset and model are already local. `--mode full` selects all backbone parameters; `--full` selects a full-sized experiment.
