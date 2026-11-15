# OCT → DINOv2 dense features

**Start here.** This version extracts features from **frozen pretrained DINOv2 ViT-S/14**.
The default extraction does not train a classifier or produce lesion segmentation masks.
Optional commands below inspect attention and compare self-supervised LoRA/full adaptation.

## Mac: copy these commands in order

Use Python 3.12 (native arm64 on Apple Silicon). Python and Git must already be installed.
Open Terminal:

```bash
git clone https://github.com/arpitmisra/mini-project.git
cd mini-project
bash oct.sh setup
bash oct.sh test
```

Already cloned? Run `git pull --ff-only` inside the repository first, then the setup/test commands.
Setup uses a separate `.venv-dense` environment; **no manual activation is needed**.
The test downloads the pinned model once (~89 MB) and runs it on **two artificial images**.
It downloads no OCT data. Wait for **PASS** before continuing.

Now replace the path below with the existing extracted OCT folder:

```bash
bash oct.sh run "$HOME/Downloads/OCT2017"
bash oct.sh show
```

The first run processes **at most 12 scans**: one per class per split. It never downloads the dataset.
The folder must contain train/val/test, each with NORMAL, CNV, DME and DRUSEN subfolders.
The second command exports one scan and displays its arrays.

## Your results

| Result | Location |
|---|---|
| One scan's complete arrays | `outputs/pretrained-smoke/dense_sample.npz` |
| That scan's identity and array descriptions | `outputs/pretrained-smoke/dense_sample.json` |
| Links to all selected scans' features | `outputs/pretrained-smoke/features_manifest.json` |
| Actual cached feature shards | `outputs/features/` |
| Real-model setup test report | `outputs/self-test/self_test.json` |
| Error details, if a command fails | `outputs/last-error.log` |

Per scan: `patches (256,384)`, `spatial_grid (16,16,384)`, and `cls (384,)`.
The grid covers the **224×224 preprocessed center crop**, not the whole original scan.
Terminal abbreviates large arrays; the file contains every value.

## Full dataset — only after the small run passes

```bash
bash oct.sh run "$HOME/Downloads/OCT2017" --full
bash oct.sh show --full
```

Full results use `outputs/pretrained/`. Allow approximately **15.5 GiB** for float16 patch arrays at 84,484 scans,
plus the dataset and metadata. No training occurs. Repeating the same command reuses valid cached features.

No local data? `bash oct.sh run --full` uses the configured local path first, then KaggleHub if unavailable.
This can download the **entire dataset**. Use `bash oct.sh test` for the initial check instead.

## If something fails

1. Read the last error line. Check that setup completed and the dataset path is correct.
2. Send `outputs/last-error.log` and the command used. For installation failures, copy Terminal's installation error.
3. The default is **CPU**, so it does not depend on MLX or Metal. Optional Mac GPU testing:
   `bash oct.sh test --device mps`. After that passes, run with `--device mps --output-dir outputs/pretrained-mps`
   and view with `bash oct.sh show --output-dir outputs/pretrained-mps`.
   Use a different run folder for a different device; cache identities remain separate.

The professor's earlier traceback was unavailable, so its exact cause is not established.
The real-model workflow passed on macOS, Windows and Linux CPU. See [evidence](docs/VALIDATION.md).
These tests establish software behavior, not segmentation quality; MPS is a separate optional test.

## Patch dashboard, normalization and fine-tuning

After the baseline setup, inspect any local image (no OCT dataset needed):

```bash
bash oct.sh inspect "/path/to/scan.jpeg" --output-dir outputs/my-scan
open outputs/my-scan/dashboard.html
```

Click a patch; switch among attention heads, cosine similarity and Pearson correlation.
The same dashboard compares pretrained mean/std normalization with RGB / 255 only.
It saves exact arrays and a numerical comparison; these are not lesion masks.
Omit the image path for a deterministic random-noise sanity check:

```bash
bash oct.sh inspect --output-dir outputs/noise-check
```

To test both adaptation methods, install their separate MLX environment once:

```bash
bash oct.sh setup-mlx
bash oct.sh finetune "/path/to/OCT2017" --mode lora
bash oct.sh finetune "/path/to/OCT2017" --mode full
```

Each is a **one-step CPU smoke check**, followed by dense extraction for at most 12 local scans.
No dataset download occurs. Both use self-supervised teacher–student training on TRAIN only.
Use `--normalization unit` for the matched normalization ablation; use `--full` only for an intentional full experiment.
See [the research runbook](docs/RESEARCH_WORKFLOW.md) for Windows commands, checkpoint inspection,
exact normalization code, saved array meanings and the controlled comparison protocol.

## More documentation

- [Full instructions, Windows commands and array loading](docs/USAGE.md)
- [What every project file does](docs/FILES.md)
- [Study guide: rank → SVD → LoRA/PEFT → Swin → weakly supervised OCT segmentation](docs/STUDY_GUIDE.md)
- [Meeting agenda and proposed experiments](docs/MEETING.md)
- [Patch inspection, normalization and LoRA/full fine-tuning](docs/RESEARCH_WORKFLOW.md)
- [Run the actual MLX DINOv2 code on a Windows or Mac CPU](docs/MLX_CPU.md)
- [Optional MLX adaptation experiment](docs/FINETUNING.md)

For publication, compare this baseline and later adapters on the **same patient-independent evaluation protocol**.
Improvement is a hypothesis to test. Dense representations alone do not establish accurate lesion boundaries.
