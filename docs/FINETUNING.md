# Self-supervised OCT adaptation method

The user selected self-supervised teacher–student adaptation. The aim is to adapt a pretrained DINOv2 backbone to OCT appearance, then export dense representations only. There is no downstream classifier or disease-label objective.

## Exact implemented objective

The initialization is the pinned Meta `facebook/dinov2-small` checkpoint, verified by the original SHA-256 and complete HF/native parameter schemas. All student backbone parameters are trainable, including transformer blocks, patch embedding, positions, normalization and mask token. A teacher starts with the same backbone and projection-head parameters and has no gradient updates.

Each training image yields two random 224 × 224 crops after aspect-preserving shortest-edge resize to 256. For a given crop, teacher and student share pixel coordinates. They receive different mild brightness/contrast perturbations; student views may be blurred. Crops are not vertically flipped or rotated. The student replaces a random fraction of patch embeddings with the trainable mask token; the teacher sees the complete crop.

Separate temporary global and patch projection heads map backbone features through a two-layer GELU MLP, L2-normalized bottleneck and row-normalized prototype weight matrix. Their output dimension is a self-supervised vocabulary (`prototypes`), not two diagnostic classes. Neither head is included in the final backbone export.

For teacher logits `z_t` and center `c`, the targets are:

```text
q_t = stop_gradient(softmax((z_t - c) / teacher_temperature))
```

Student log probabilities are `log_softmax(z_s / student_temperature)`. Global cross-entropy matches student crop 0 to teacher crop 1 and student crop 1 to teacher crop 0, preserving image identity. Patch cross-entropy matches **the same spatial location in the same crop pair**, and is averaged only over masked positions per image. No correspondence is assumed between different crop geometries.

```text
loss = global_loss_weight * mean(cross-view CLS cross-entropy)
     + patch_loss_weight  * mean(masked same-view patch cross-entropy)
```

AdamW updates student parameters with global gradient-norm clipping. Teacher parameters are updated after the optimizer step:

```text
teacher = teacher_momentum * teacher + (1 - teacher_momentum) * student
```

Global and patch centers are EMA averages of the preceding teacher logits. Temperatures and momenta are constant in this implementation. Backbone gradients must be finite and nonzero, and both student and EMA backbone parameters must change before an epoch is accepted as fine-tuning. History records loss, gradient norms, teacher target entropy and patch-feature spread. These diagnostics can reveal failures; they do not prove absence of representation collapse or scientific usefulness.

The final epoch's **EMA teacher backbone** is exported. During feature extraction it is frozen and evaluated with deterministic RGB → bicubic shortest-edge resize → center crop → normalization. Outputs are the final LayerNorm CLS and patch tokens, not prototype logits. Patches are not additionally L2-normalized by the pipeline.

## Relationship to DINO and iBOT

The implementation follows the teacher/student, centered target and masked-patch ideas documented in Meta's primary sources:

- [DINOv2 SSL architecture](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py)
- [Global DINO loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py)
- [iBOT patch loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py)
- [MLX module API](https://ml-explore.github.io/mlx/build/html/python/nn/module.html)

This project independently implements a compact adaptation in MLX. It is **not the full DINOv2 pretraining recipe**: no local multicrop branch, KoLeo regularizer, distributed optimization, Sinkhorn target balancing, official giant prototype heads or original large-scale schedules. Its augmentations are deliberately mild and spatially aligned for OCT. The recipe and hyperparameters need evaluation on the actual data; no improvement over frozen pretrained features is claimed from code review.

## Data protocol

Only records whose current split and original source split are both `train` may enter fine-tuning. The function accepts only that dataset, not a mapping containing validation/test. Original diagnoses and binary labels are preserved for later analysis but are never passed into the objective. Per-class smoke selection and optional diagnosis-stratified train/validation reservation can use folder labels for sampling, not supervision.

The official validation/test images are not used for gradients, centers, teacher EMA, early stopping or checkpoint selection. They can be embedded after training. Optional train-derived validation is reserved and also not optimized. Fixed epochs determine the final checkpoint. Exact cross-split duplicate checks run before training; they do not establish patient independence or identify all near duplicates.

## Persistence and reproducibility

The training contract snapshots settings, seed, model identity, preprocessing/Pillow version, MLX/NumPy versions and a hash of selected train-record identities. A different contract requires a new output directory. Completed runs are reused only when the contract and adapted file hash match. The original pretrained weight hash is retained separately from the adapted weight hash.

After each epoch, a state bundle saves student and teacher backbones, both SSL heads, AdamW state, both centers, epoch/step counts, history and NumPy RNG state. A pointer is updated only after the complete epoch bundle has been written. Random augmentation, masking and epoch permutations use that NumPy RNG. There are no stochastic dropout layers in the pinned backbone or these heads. An interrupted epoch restarts from the previous completed epoch; this is not a mid-batch resume guarantee. All completed epoch bundles are retained.

Final export saves the teacher backbone alone and strictly reloads it into a fresh native model. CLS and patch outputs on a training-image probe must match before the completed checkpoint sidecar is written. The sidecar includes the HF architecture configuration, so later adapted extraction does not need the original HF weights or a network request. Feature cache identities include the adapted hash and contract, preventing reuse across different fine-tuned models.

## Validation boundary

Windows tests check augmentation determinism/alignment, train-only eligibility, configuration, provenance, checksums, default-stage routing and dense cache/export behavior using artificial arrays. They cannot execute MLX gradients or native checkpoint serialization.

Mac tests in `test_mlx_finetuning.py` check the loss against a NumPy reference, masked-position selection, cross-view ordering, backbone updates, teacher EMA, stop-gradient targets and complete optimizer/center/RNG restoration through the next step. Opt-in pretrained integration runs a tiny actual DINOv2 adaptation and dense export. These are provided for the Mac and remain unexecuted on the Windows host.

`scripts/verify_parity.py` remains a check of the original pretrained adapter against its HF reference. Comparing an adapted backbone against the original pretrained weights would be meaningless; adapted checkpoints instead have strict schema/hash and save/reload checks. An independent implementation-level parity check of the full fine-tuning procedure is not claimed.
