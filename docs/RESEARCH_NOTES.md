# Current method — 25 October 2026

The active pipeline now performs self-supervised teacher–student OCT adaptation followed by dense extraction only. See [FINETUNING.md](FINETUNING.md) for the exact global/masked-patch objective, EMA/centering, train-only data protocol, checkpoint format and differences from full DINOv2 pretraining. No classifier is trained by the current entry point.

The pinned original checkpoint, MLX-VLM implementation, weight-schema conversion and deterministic extraction preprocessing investigated below remain the initialization/extraction foundation. Final features now come from the adapted EMA teacher, with a distinct weight checksum. The earlier classifier/API investigation below is retained as **historical context**, not current execution instructions.

---

# Implementation investigation — 19 October 2026

The technical sources below were inspected during implementation. None of the upstream authors' test claims are treated as locally executed results.

## Choice of backbone

Meta's official [DINOv2 repository](https://github.com/facebookresearch/dinov2) provides ViT-S/14 with and without registers. The [DINOv2 paper](https://arxiv.org/abs/2304.07193) describes transferable self-supervised global and dense features. Original [DINO](https://github.com/facebookresearch/dino) and [its paper](https://arxiv.org/abs/2104.14294) are relevant background, but their ViT-S/16 checkpoint and patch count are not interchangeable with DINOv2.

The maintained [MLX-VLM DINOv2 addition, PR #2184](https://github.com/Blaizzy/mlx-vlm/pull/2184), merged September 2026, provides a standalone native encoder and reports upstream numerical checks. Its [v0.7.6 source](https://github.com/Blaizzy/mlx-vlm/blob/v0.7.6/mlx_vlm/models/dinov2/dinov2.py) and [configuration](https://github.com/Blaizzy/mlx-vlm/blob/v0.7.6/mlx_vlm/models/dinov2/config.py) were inspected directly. The inspected API is `Model(ModelConfig(...))`, `sanitize(weights)`, `forward_features(NHWC)`; it separately returns normalized CLS, register and patch tokens.

Options considered:

| Route | Decision |
|---|---|
| Original PyTorch DINO prototype | Preserved as previous work; fails the native MLX/DINOv2 requirement |
| New custom MLX ViT port | Avoided; would add interpolation, mapping and attention implementation risk |
| MLX-VLM v0.7.6 standalone DINOv2 | Selected; inspectable implementation, HF conversion, proper token separation |
| Unverified community MLX weight conversion | Not required; use pinned Meta HF safetensors directly |

MLX-VLM is MIT-licensed. No source is vendored; its version is pinned as a dependency. The Meta checkpoint is Apache-2.0. The broad VLM dependency graph is a cost of using this maintained implementation; a future minimal dependency extraction would need license preservation and parity checks.

## Exact checkpoint contract

[facebook/dinov2-small config](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/config.json): hidden size 384, depth 12, heads 6, MLP ratio 4, GELU, LayerNorm epsilon 1e-6, patch size 14, image size 518, no registers, qkv biases. Model revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` was confirmed via the model repository metadata. Its safetensors metadata reports 22,056,576 float32 parameters; the file is roughly 89 MB. No full model download was performed in Windows development.

The actual safetensors header was checked using a bounded 64 KiB range request: all 223 keys/shapes matched the adapter schema. The published LFS SHA-256 is `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`; both locally supplied and downloaded checkpoint files must match it before loading. The adapter checks the complete HF schema before conversion, then checks all native parameter names/shapes before `load_weights(strict=True)`. Positional embeddings are `[1,1370,384]` (37×37 patches plus CLS) and are interpolated using MLX-VLM's HF-style size-based branch, without the original repository's +0.1 scale offset. The optional parity script therefore uses the HF reference implementation and the same HF checkpoint, not an incompatible original-repo preprocessing/interpolation combination.

[Checkpoint preprocessing](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/preprocessor_config.json) specifies RGB, shortest edge 256, bicubic resizing, 224 center crop, 1/255 rescaling and ImageNet mean/std. The project implements those steps using Pillow/NumPy; it includes Pillow version in cache identity.

## MLX APIs

The official [MLX repository](https://github.com/ml-explore/mlx), [examples](https://github.com/ml-explore/mlx-examples), [cross entropy API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.cross_entropy.html), and [training state example](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst) informed the head implementation. It uses `nn.value_and_grad`, AdamW's `update(model, gradients)`, integer targets, per-sample cross entropy, and explicit `mx.eval` on loss, parameters and optimizer state. Only cached CLS arrays enter head training.

## Dataset retrieval

[Official KaggleHub](https://github.com/Kaggle/kagglehub) documents dataset downloads, automatic cache reuse, version suffixes, authentication, `KAGGLEHUB_CACHE`, and the `path=` option for one file/directory. That does not establish an efficient streaming data loader for this dataset. Local mode and small synthetic tests avoid that dependency during development. The remote download API is tested through a stub here; a real Kaggle download remains untested intentionally.

## Localization scope

[Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) develops CAM for architectures connecting class scores to spatial features. A final-CLS-only ViT head does not directly depend on the final patch outputs; gradients of that head with respect to those final patch outputs need not provide useful localization. This project preserves spatial embeddings and an extension interface without presenting an unvalidated CAM overlay.
