# Research workflow: inspect, adapt, compare

The project currently saves final LayerNorm CLS and patch embeddings. These are not the outputs of individual attention heads. The new inspector additionally captures actual attention probabilities for one selected transformer block. No disease classifier or segmentation mask is produced.

## 1. Inspect one image

Mac, from the cloned repository (Python 3.12 recommended):

```bash
bash oct.sh setup
bash oct.sh inspect "/path/to/scan.jpeg" --output-dir outputs/my-scan
open outputs/my-scan/dashboard.html
```

Windows, fresh baseline environment:

```powershell
python -m venv .venv-dense
.venv-dense\Scripts\python.exe -m pip install -r requirements.txt
.venv-dense\Scripts\python.exe oct.py inspect "C:\path\scan.jpeg" --output-dir outputs/my-scan
```

Double-click `outputs/my-scan/dashboard.html`. It works offline, without a server. An initial pinned pretrained-weight download may be required; `--offline` forbids it. The inspector never accesses Kaggle or the OCT dataset loader. Repeating an identical inspection reuses verified arrays. Choose a different output folder for a different image, model, layer or device.

Omit the image argument to use deterministic RGB random noise (seed 42):

```bash
bash oct.sh inspect --output-dir outputs/noise-check
```

Any image accepted by Pillow can be inspected: a scan or ordinary photograph. Use a real OCT image and an unrelated photograph as well as noise to assess qualitative behavior. The built-in checks establish finite outputs, expected shapes, full attention row sums, agreement with a repeated ordinary feature forward pass, and lossless NPZ reload. They do not establish meaningful anatomy or out-of-distribution detection. Even noise can produce structured attention.

## 2. Read the dashboard correctly

- Click a patch to choose the query. All-pairs matrix rows are queries and columns are keys. Clicking a matrix cell selects both.
- The default attention block is 11, the last block; `--layer 0` selects the first. Export another folder to inspect another block. Head indices are 0–5; an arithmetic mean is available.
- Attention is `softmax(Q K^T / sqrt(64))`, before multiplication by V. It can be asymmetric. A head output is A V, a different quantity.
- Full attention has shape `(6,257,257)`: CLS is token 0 and spatial patches are 1–256. Full rows sum to one. The displayed 256-by-256 patch matrix excludes CLS without renormalizing; the selected query's CLS mass is printed separately.
- Cosine and Pearson use final-layer `(256,384)` descriptors, regardless of which attention block was selected. They are symmetric. Pearson centers each descriptor over channels; it is not a correlation estimated across patients. Undefined comparisons for constant/zero vectors are shown gray.
- Input preprocessing controls switch between `imagenet` and `unit`. Attention color ranges adapt to each whole matrix; compare printed values/arrays, not darkness alone. Similarity colors have a fixed −1 to +1 range.
- Coordinates belong to the 224-by-224 center crop, not the original scan. Each grid anchor covers 14-by-14 input pixels; attention provides broader context.

Files in the chosen output folder:

| File / array | Meaning |
|---|---|
| `dashboard.html` | Self-contained viewer, including the scan crop and float32 relationship arrays |
| `report.json` | Input checksum, model identity, layer, device, preprocessing, sanity checks and comparisons |
| `arrays.npz` | Exact float32 feature/attention/similarity arrays, plus uint8 crop |
| `crop_rgb` | `(224,224,3)` displayed crop before mean/std standardization |
| `imagenet_cls`, `unit_cls` | `(384,)` final CLS features |
| `imagenet_patches`, `unit_patches` | `(256,384)` final spatial features |
| `imagenet_attention`, `unit_attention` | `(6,257,257)` selected-block attention |
| `*_cosine`, `*_pearson` | `(256,256)` pairwise feature relationships |
| `*_valid_cosine`, `*_valid_pearson` | `(256,)` valid descriptor masks |

Load arrays in the same Python environment:

```python
import numpy as np
with np.load("outputs/my-scan/arrays.npz", allow_pickle=False) as a:
    print(a.files)
    print(a["imagenet_patches"][123])  # row 7, column 11
    print(a["imagenet_attention"][0, 124, 1:])  # head 0, patch 123 -> all patches
    print(a["imagenet_cosine"][123])
```

## 3. Where image normalization occurs

`src/oct_dinov2_mlx/preprocessing.py::normalize_rgb` converts pixels to float32 and divides by 255. In `imagenet` mode it then computes `(x - mean) / std`, using RGB means `(0.485,0.456,0.406)` and standard deviations `(0.229,0.224,0.225)`.

`preprocess` performs RGB conversion, shortest-side bicubic resize and center crop before calling `normalize_rgb`. Both MLX and PyTorch extraction use this function. The usual path is `pipeline.prepare_data` -> lazy dataset transform -> `preprocess` -> backbone.

During adaptation, `augmentations.py::paired_views` creates spatially aligned teacher/student crop pairs, applies photometric augmentation, and calls the same `normalize_rgb` **after augmentation and before the backbone**. Both teacher and student use the selected mode. Extraction after adaptation must match the checkpoint's training preprocessing; mismatched extraction is rejected.

The ablation `unit` still divides by 255, but omits channel mean/std standardization. It does not feed 0–255 values into the model. This is the precise definition of “unnormalized” in this experiment.

Input normalization, transformer LayerNorm, SSL-head vector normalization, and cosine L2 normalization are four separate operations. This ablation changes only input standardization. Internal model normalization remains enabled, and the stored patch descriptors are final LayerNorm outputs.

The inspector compares identical crops and weights under the two inputs. It reports descriptor RMSE, mean corresponding-patch cosine, change in cosine matrices, feature norms and attention entropy. A warmed single forward time is diagnostic only, not a rigorous speed benchmark. None of these measurements determines which input preprocessing gives better segmentation.

## 4. LoRA and full fine-tuning: bounded checks first

Mac, install MLX separately from the PyTorch environment:

```bash
bash oct.sh setup-mlx
bash oct.sh finetune "/path/to/OCT2017" --mode lora
bash oct.sh finetune "/path/to/OCT2017" --mode full
bash oct.sh finetune "/path/to/OCT2017" --mode lora --normalization unit
bash oct.sh finetune "/path/to/OCT2017" --mode full --normalization unit
```

Windows:

```powershell
python -m venv .venv-mlx-cpu
.venv-mlx-cpu\Scripts\python.exe -m pip install -r requirements-mlx-cpu.txt
.venv-mlx-cpu\Scripts\python.exe oct.py finetune "C:\path\OCT2017" --mode lora
.venv-mlx-cpu\Scripts\python.exe oct.py finetune "C:\path\OCT2017" --mode full
```

Add `--normalization unit` for the other two Windows runs. CPU is the default on both platforms; Mac Metal is opt-in with `--device gpu` and a separate `--output-dir`. Use `--offline` to prohibit all downloads.

Default checks index at most one scan per class per split, train for ONE optimizer step on a batch of two TRAIN scans, then extract all 12 selected scans. Validation/test images never enter the training objective. These runs verify code behavior and cannot demonstrate adaptation quality.

Each mode saves under `outputs/{mode}-{normalization}-smoke/`, for example `outputs/lora-imagenet-smoke/`:

| Output | Meaning |
|---|---|
| `finetuning/backbone.safetensors` | Final EMA teacher backbone; LoRA updates merged into ordinary native weights |
| `finetuning/backbone.json` | Training contract, input normalization, checksums, merge/reload checks and parameter counts |
| `finetuning/parameter_counts.json` | Total and trainable backbone counts plus trainable SSL-head counts |
| `finetuning/history.json` | Loss, gradient norms, teacher entropy and feature spread per epoch |
| `finetuning/epoch-*/state.safetensors` | Factorized student/teacher, optimizer and centers for epoch-boundary resume |
| `features_manifest.json` | Dense-cache index for each split |
| `experiment_metadata.json` | Resolved configuration and data provenance |

Show an adapted scan without rerunning inference:

```bash
bash oct.sh show --output-dir outputs/lora-imagenet-smoke
```

Inspect the adapted backbone on the same image used for the pretrained baseline:

```bash
bash oct.sh inspect "/path/to/scan.jpeg" --backend mlx --checkpoint outputs/lora-imagenet-smoke/finetuning/backbone.safetensors --output-dir outputs/my-scan-lora
open outputs/my-scan-lora/dashboard.html
```

On Windows use `.venv-mlx-cpu\Scripts\python.exe oct.py inspect ... --backend mlx`. Use the MLX backend for the pretrained reference too when comparing adapted MLX weights, avoiding an unnecessary framework difference. The inspector intentionally tests both input modes and records the checkpoint's training normalization; the nonmatching mode is an explicit inference ablation.

## 5. What exactly is trained

Both modes use the existing simplified DINO/iBOT-style objective: cross-view CLS distillation and aligned same-view masked-patch distillation, centered teacher targets, EMA teacher, and temporary SSL projection heads. Disease labels are not used in the loss. This is not the complete Meta DINOv2 pretraining recipe.

Full mode unfreezes the backbone. LoRA freezes the backbone and adds rank-r updates to query and value projections in each of the 12 blocks; the fused key slice stays unchanged. Default r=8 and alpha=8, with no adapter dropout. `--rank` sets both r and alpha (scale=1); advanced YAML can specify them independently. At r=8 there are 147,456 trainable backbone adapter parameters, plus the shared SSL heads.

The teacher uses EMA of the trainable parameters: LoRA factors and heads in LoRA mode, backbone and heads in full mode. EMA of factors is not mathematically identical to EMA of their product; this policy is recorded. Frozen base tensors are not EMA-updated. The final teacher factors are merged for dense extraction. The training state retains factors for resume. Merge agreement and strict merged-checkpoint serialization are checked separately.

SSL heads use a separate `seed+1` initialization stream so that drawing LoRA factors cannot change the comparison's head initialization. View sampling/masks share the same seed and settings. Original feature caches cannot train internal adapters; after adaptation, new checkpoint identities cause new dense caches. Repeating a completed matching run reuses the checkpoint and features.

The MLX adapter now uses differentiable bicubic interpolation for positional embeddings on both CPU and Apple GPU. The upstream automatic Metal interpolation kernel had no gradient implementation, breaking full fine-tuning on Macs. Adapter/training-contract version 3 records this correction. Older MLX runs are retained as historical outputs; start a new folder instead of resuming them, for example `bash oct.sh finetune "$HOME/Downloads/OCT2017" --mode lora --output-dir outputs/lora-v3`. Existing PyTorch baseline caches are unaffected.

## 6. Full research comparison, after checks pass

Append `--full` to each fine-tuning command to use the full configuration (five epochs, no step cap). This can be expensive on CPU. Full mode permits Kaggle fallback if the local path is unavailable, so use `--offline` when the local dataset and model are already present. Full outputs use the suffix `-full`; do not confuse `--mode full` (all backbone parameters) with `--full` (full experiment size).

Plan six matched conditions: frozen/LoRA/full crossed with imagenet/unit preprocessing. Frozen inspection compares inference inputs; adapted runs must train and extract consistently within their condition. Use the same patient-independent split, image selection, crop policy, seeds and downstream spatial evaluation. Repeat seeds for reported results; equal steps and equal compute budgets answer different questions. Select settings without using final test outcomes. Lower SSL loss, higher similarity or sharper attention does not prove better localization. Segmentation-quality claims need a defined lesion target and independent spatial reference masks.

## Implementation map

| File | Role |
|---|---|
| `inspection.py` | Actual attention capture in PyTorch/MLX, similarities, numerical comparisons, caching and sanity checks |
| `dashboard.py` | Offline HTML/JavaScript dashboard generator |
| `research_cli.py` | Simple inspect/finetune commands and bounded defaults |
| `lora.py` | Native MLX query/value adapters, freezing and merge export |
| `finetuning.py` | Shared objective, optimization, EMA, merge/reload validation and resume |
| `preprocessing.py`, `augmentations.py` | Shared selectable input standardization in inference/training |
| `tests/test_inspection.py`, `tests/test_lora.py` | Normalization, relationships, attention semantics, safety of rendering, frozen weights and merge checks |
| `tests/test_mlx.py` | Opt-in real-checkpoint integration for full and LoRA adaptation/extraction/reuse |

Existing full file inventory: [FILES.md](FILES.md). Research command exceptions are saved in `outputs/research-error.log`.
