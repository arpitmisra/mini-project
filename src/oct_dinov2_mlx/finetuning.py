"""Native MLX self-supervised adaptation, not binary classification.

Two aligned teacher/student crop pairs; cross-view CLS distillation and
same-view masked-patch distillation with centered EMA teacher targets.
This small-device adaptation omits the full DINOv2 multicrop/KoLeo recipe.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np
from PIL import Image

from .adaptation import METHOD, read_adapted_checkpoint, training_contract
from .augmentations import paired_views, patch_masks
from .backbone import DINOv2Extractor, native_model, validate_shapes
from .cache import cache_lock
from .common import digest, file_digest, write_json
from .preprocessing import preprocess


class ProjectionHead(nn.Module):
    """Temporary SSL prototypes; no disease classes or binary logits."""
    def __init__(self, dimension, config):
        super().__init__()
        self.fc1 = nn.Linear(dimension, config["head_hidden_dim"])
        self.fc2 = nn.Linear(config["head_hidden_dim"], config["head_bottleneck_dim"])
        self.prototypes = mx.random.normal((config["prototypes"], config["head_bottleneck_dim"])) * 0.02

    def __call__(self, x):
        x = self.fc2(nn.gelu(self.fc1(x)))
        x = x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)
        w = self.prototypes / mx.sqrt(mx.sum(self.prototypes ** 2, axis=-1, keepdims=True) + 1e-6)
        return x @ w.T


class SelfDistillationModel(nn.Module):
    def __init__(self, backbone, config, dimension=384):
        super().__init__()
        self.backbone = backbone
        self.global_head = ProjectionHead(dimension, config)
        self.patch_head = ProjectionHead(dimension, config)

    def __call__(self, images, masks=None):
        f = self.backbone.forward_features(images, masks=masks)
        return self.global_head(f["x_norm_clstoken"]), self.patch_head(f["x_norm_patchtokens"]), f["x_norm_patchtokens"]


def centered_targets(logits, center, temperature):
    return mx.stop_gradient(mx.softmax((logits - center) / temperature, axis=-1))


def distillation_loss(student, images, masks, cls_targets, patch_targets, settings):
    cls_logits, patch_logits, _ = student(images, masks)
    half = cls_logits.shape[0] // 2
    # View-major order: view 0 for all B images, then view 1 for all B images.
    opposite = mx.concatenate([cls_targets[half:], cls_targets[:half]], axis=0)
    def cross_entropy(logits, targets):
        scores = logits / settings["student_temperature"]
        return -mx.sum(targets * (scores - mx.logsumexp(scores, axis=-1, keepdims=True)), axis=-1)
    global_loss = mx.mean(cross_entropy(cls_logits, opposite))
    patch_ce = cross_entropy(patch_logits, patch_targets)
    patch_loss = mx.mean(mx.sum(patch_ce * masks, axis=1) / mx.maximum(mx.sum(masks, axis=1), 1))
    return settings["global_loss_weight"] * global_loss + settings["patch_loss_weight"] * patch_loss


def train_step(student, teacher, optimizer, centers, teacher_images, student_images, masks, settings):
    teacher.eval()
    t_cls, t_patch, patch_features = teacher(teacher_images)
    q_cls = centered_targets(t_cls, centers["cls"], settings["teacher_temperature"])
    q_patch = centered_targets(t_patch, centers["patches"], settings["teacher_temperature"])
    mx.eval(q_cls, q_patch, t_cls, t_patch, patch_features)
    student.train()
    loss, grads = nn.value_and_grad(student, distillation_loss)(student, student_images, masks, q_cls, q_patch, settings)
    norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
    backbone_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads["backbone"])))
    mx.eval(loss, norm, backbone_norm)
    if not np.isfinite([loss.item(), norm.item(), backbone_norm.item()]).all() or backbone_norm.item() == 0:
        raise ValueError("Nonfinite loss/gradients or zero backbone gradient during fine-tuning")
    scale = mx.minimum(1.0, settings["gradient_clip"] / (norm + 1e-6))
    optimizer.update(student, tree_map(lambda g: g * scale, grads))
    mx.eval(student.parameters(), optimizer.state)
    momentum = settings["teacher_momentum"]
    teacher.update(tree_map(lambda t, s: momentum * t + (1 - momentum) * s,
                            teacher.parameters(), student.parameters()))
    c = settings["center_momentum"]
    centers = {"cls": c * centers["cls"] + (1 - c) * mx.mean(t_cls, axis=0, keepdims=True),
               "patches": c * centers["patches"] + (1 - c) * mx.mean(t_patch, axis=(0, 1), keepdims=True)}
    entropy = -mx.mean(mx.sum(q_cls * mx.log(mx.maximum(q_cls, 1e-8)), axis=-1))
    spread = mx.mean(mx.std(patch_features, axis=(0, 1)))
    mx.eval(teacher.parameters(), centers, entropy, spread)
    return centers, {"loss": float(loss.item()), "gradient_norm": float(norm.item()),
                     "backbone_gradient_norm": float(backbone_norm.item()),
                     "teacher_entropy": float(entropy.item()), "patch_feature_std": float(spread.item())}


def save_resume(directory, epoch, student, teacher, optimizer, centers, rng, metadata):
    epoch_dir = Path(directory) / f"epoch-{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, state in (("student", student.parameters()), ("teacher", teacher.parameters()),
                        ("optimizer", optimizer.state), ("centers", centers)):
        arrays.update({f"{name}.{key}": value for key, value in tree_flatten(state)})
    mx.eval(arrays)
    if not bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in arrays.values()])).item()):
        raise ValueError("Cannot save nonfinite fine-tuning state")
    path = epoch_dir / "state.safetensors"
    temporary = epoch_dir / f"state.{os.getpid()}.safetensors"
    mx.save_safetensors(str(temporary), arrays)
    os.replace(temporary, path)
    state_meta = {**metadata, "epoch": epoch, "rng_state": rng.bit_generator.state,
                  "state_sha256": file_digest(path)}
    write_json(epoch_dir / "state.json", state_meta)
    # The pointer advances only once the complete epoch has been safely saved.
    write_json(Path(directory) / "latest.json", {"directory": epoch_dir.name})


def read_resume(directory, contract):
    pointer = Path(directory) / "latest.json"
    if not pointer.exists():
        return None
    name = json.loads(pointer.read_text(encoding="utf-8"))["directory"]
    epoch_dir = (Path(directory) / name).resolve()
    if epoch_dir.parent != Path(directory).resolve():
        raise ValueError("Invalid checkpoint directory")
    meta = json.loads((epoch_dir / "state.json").read_text(encoding="utf-8"))
    if meta["contract_sha256"] != digest(contract):
        raise ValueError("Existing fine-tuning run has different data/settings; choose a new --output-dir")
    path = epoch_dir / "state.safetensors"
    if meta["state_sha256"] != file_digest(path):
        raise ValueError("Resume state checksum mismatch")
    return path, meta


def restore_resume(path, student, teacher, optimizer):
    saved = mx.load(str(path))
    def group(prefix):
        return {k[len(prefix) + 1:]: v for k, v in saved.items() if k.startswith(prefix + ".")}
    for name, model in (("student", student), ("teacher", teacher)):
        values = group(name)
        validate_shapes(values, {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())})
        model.load_weights(list(values.items()), strict=True)
    optimizer.state = tree_unflatten(list(group("optimizer").items()))
    centers = tree_unflatten(list(group("centers").items()))
    mx.eval(student.parameters(), teacher.parameters(), optimizer.state, centers)
    return centers


def finetune(train_dataset, config, output_dir):
    """Optimize on original TRAIN records only; return the final EMA backbone."""
    directory = Path(output_dir) / "finetuning"
    final_path = directory / "backbone.safetensors"
    contract = training_contract(config, train_dataset.records)
    fingerprint = digest(contract)
    settings = config["finetuning"]
    with cache_lock(directory):
        if final_path.with_suffix(".json").exists():
            meta = read_adapted_checkpoint(final_path, config["model"])
            if meta["contract_sha256"] != fingerprint:
                raise ValueError("Completed fine-tuning run has different data/settings; choose a new --output-dir")
            print("Reusing completed fine-tuned backbone (no optimizer steps)")
            return final_path
        started = time.perf_counter()
        resume = read_resume(directory, contract)
        mx.random.seed(config["runtime"]["seed"])
        rng = np.random.default_rng(config["runtime"]["seed"])
        if resume:
            _, resume_meta = resume
            hf_config = resume_meta["hf_config"]
            backbone = native_model(hf_config)
        else:
            initial = DINOv2Extractor(config["model"], config["preprocessing"]["image_size"])
            backbone, hf_config = initial.model, initial.hf_config
        backbone.unfreeze()
        student = SelfDistillationModel(backbone, settings)
        teacher = SelfDistillationModel(native_model(hf_config), settings)
        teacher.update(tree_map(lambda value: value, student.parameters()))
        teacher.freeze()
        teacher.eval()
        optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
        centers = {"cls": mx.zeros((1, settings["prototypes"])), "patches": mx.zeros((1, 1, settings["prototypes"]))}
        history, first_epoch, steps = [], 0, 0
        if resume:
            centers = restore_resume(resume[0], student, teacher, optimizer)
            rng.bit_generator.state = resume_meta["rng_state"]
            history, first_epoch, steps = resume_meta["history"], resume_meta["epoch"], resume_meta["steps"]
            print(f"Resuming after completed epoch {first_epoch}")
        mx.eval(student.parameters(), teacher.parameters(), centers)
        before = dict(tree_flatten(student.backbone.parameters()))
        teacher_before = dict(tree_flatten(teacher.backbone.parameters()))
        side = config["preprocessing"]["image_size"] // 14
        for epoch in range(first_epoch, settings["epochs"]):
            rows, seen = [], 0
            order = rng.permutation(len(train_dataset.records))
            for step, start in enumerate(range(0, len(order), settings["batch_size"])):
                if settings["max_steps_per_epoch"] is not None and step >= settings["max_steps_per_epoch"]:
                    break
                indices = order[start:start + settings["batch_size"]]
                views = []
                for i in indices:
                    with Image.open(train_dataset.records[int(i)].image_path) as image:
                        views.append(paired_views(image, config["preprocessing"], rng, settings["augmentation_strength"]))
                # [B,2,H,W,3] -> [2B,H,W,3], view-major, same image order in both views.
                teacher_images, student_images = [np.concatenate([np.stack([v[k][j] for v in views]) for j in range(2)]) for k in range(2)]
                masks = patch_masks(len(teacher_images), side * side, settings["mask_ratio"], rng)
                centers, row = train_step(student, teacher, optimizer, centers,
                    mx.array(teacher_images), mx.array(student_images), mx.array(masks), settings)
                rows.append(row)
                seen += len(indices)
                steps += 1
            if not rows:
                raise ValueError("No fine-tuning steps executed")
            history.append({"epoch": epoch + 1, "steps": len(rows), "images_seen": seen,
                            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}})
            print(f"Fine-tuning epoch {epoch + 1}: loss={history[-1]['loss']:.6f}; steps={len(rows)}")
            changed = any(bool(mx.any(before[k] != v).item()) for k, v in tree_flatten(student.backbone.parameters()))
            if not changed:
                raise RuntimeError("Backbone weights did not change; refusing to call this fine-tuning")
            if not any(bool(mx.any(teacher_before[k] != v).item()) for k, v in tree_flatten(teacher.backbone.parameters())):
                raise RuntimeError("EMA teacher backbone did not change; check learning rate and teacher momentum")
            save_resume(directory, epoch + 1, student, teacher, optimizer, centers, rng,
                        {"contract_sha256": fingerprint, "hf_config": hf_config, "history": history, "steps": steps})
            write_json(directory / "history.json", history)
        # Also restore this report if interruption occurred after the epoch pointer
        # advanced but before the report or final backbone was written.
        write_json(directory / "history.json", history)
        # Export only the EMA backbone. SSL projection heads are not feature outputs.
        teacher.backbone.eval()
        temporary = directory / f"backbone.{os.getpid()}.safetensors"
        teacher.backbone.save_weights(str(temporary))
        os.replace(temporary, final_path)
        restored = native_model(hf_config)
        restored.load_weights(str(final_path), strict=True)
        restored.freeze()
        restored.eval()
        with Image.open(train_dataset.records[0].image_path) as image:
            probe = mx.array(preprocess(image, **config["preprocessing"])[None])
        expected = teacher.backbone.forward_features(probe)
        actual = restored.forward_features(probe)
        for key in ("x_norm_clstoken", "x_norm_patchtokens"):
            mx.eval(expected[key], actual[key])
            np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(expected[key]), atol=1e-6, rtol=1e-5)
        write_json(final_path.with_suffix(".json"), {
            "format_version": 1, "status": "complete", "method": METHOD,
            "representation_source": "ema_teacher", "base_model": contract["base_model"],
            "weights_sha256": file_digest(final_path), "hf_config": hf_config,
            "contract": contract, "contract_sha256": fingerprint,
            "epochs_completed": settings["epochs"], "steps": steps, "history": history,
            "current_invocation_seconds": time.perf_counter() - started,
            "checkpoint_reload_verified": True,
            "scope": "Self-supervised OCT adaptation, not a reproduction of full DINOv2 pretraining",
        })
    return final_path
