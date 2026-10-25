"""Native gradients, EMA, masked loss and complete epoch-state round trips."""
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.mlx
if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
    pytest.skip("Native Apple Silicon required", allow_module_level=True)
from oct_dinov2_mlx.common import require_mlx, digest
require_mlx()
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from oct_dinov2_mlx.config import ROOT, load_config
from oct_dinov2_mlx.finetuning import (SelfDistillationModel, centered_targets, distillation_loss,
                                     read_resume, restore_resume, save_resume, train_step)


class TinyBackbone(nn.Module):
    """Synthetic 4-token encoder for fast optimization tests, not DINOv2."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)
        self.mask_token = mx.zeros((1, 8))
        self.context = nn.Linear(8, 8)

    def forward_features(self, images, masks=None):
        x = self.projection(images.reshape(images.shape[0], 4, 3))
        if masks is not None:
            x = mx.where(masks[..., None], self.mask_token, x)
        x = x + self.context(mx.mean(x, axis=1, keepdims=True))
        return {"x_norm_clstoken": mx.mean(x, axis=1), "x_norm_patchtokens": x}


def setup_models():
    mx.random.seed(42)
    settings = load_config(ROOT / "configs/smoke_test.yaml")["finetuning"]
    settings.update({"learning_rate": .001, "teacher_momentum": .9, "prototypes": 8,
                     "head_hidden_dim": 16, "head_bottleneck_dim": 8})
    student = SelfDistillationModel(TinyBackbone(), settings, dimension=8)
    teacher = SelfDistillationModel(TinyBackbone(), settings, dimension=8)
    teacher.update(tree_map(lambda v: v, student.parameters()))
    teacher.freeze()
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    centers = {"cls": mx.zeros((1, 8)), "patches": mx.zeros((1, 1, 8))}
    images = mx.array(np.random.default_rng(4).normal(size=(4, 2, 2, 3)).astype(np.float32))
    masks = mx.array([[True, False, True, False], [False, True, False, True]] * 2)
    return settings, student, teacher, optimizer, centers, images, masks


def test_backbone_updates_teacher_ema_and_stop_gradient():
    settings, student, teacher, optimizer, centers, images, masks = setup_models()
    before = {k: np.asarray(v).copy() for k, v in tree_flatten(teacher.parameters())}
    centers, metrics = train_step(student, teacher, optimizer, centers, images, images * .9, masks, settings)
    assert metrics["backbone_gradient_norm"] > 0
    assert np.isfinite(list(metrics.values())).all()
    assert not tree_flatten(teacher.trainable_parameters())
    current = dict(tree_flatten(student.parameters()))
    assert not np.array_equal(before["backbone.projection.weight"], np.asarray(current["backbone.projection.weight"]))
    for k, v in tree_flatten(teacher.parameters()):
        np.testing.assert_allclose(np.asarray(v), .9 * before[k] + .1 * np.asarray(current[k]), atol=1e-6)
    assert not np.allclose(np.asarray(centers["cls"]), 0)
    weights = mx.arange(8, dtype=mx.float32)
    gradient = mx.grad(lambda x: mx.sum(centered_targets(x, mx.zeros((1, 8)), .07) * weights))(mx.ones((1, 8)))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 8)))


def test_loss_uses_opposite_cls_view_and_only_masked_patch_locations():
    settings = load_config(ROOT / "configs/smoke_test.yaml")["finetuning"]
    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls = mx.array([[1., 2., 3.], [3., 2., 1.]])
            self.patches = mx.array([[[1., 0., -1.], [2., 1., 0.]], [[0., 1., 2.], [-1., 0., 1.]]])
        def __call__(self, images, masks):
            return self.cls, self.patches, self.patches
    model = Logits()
    q_cls = mx.array([[.7, .2, .1], [.1, .2, .7]])
    q_patch = mx.array([[[.2, .3, .5], [.7, .2, .1]], [[.1, .3, .6], [.6, .3, .1]]])
    masks = mx.array([[True, False], [False, True]])
    def ce(logits, targets):
        scores = logits / settings["student_temperature"]
        scores -= scores.max(axis=-1, keepdims=True)
        return -(targets * (scores - np.log(np.exp(scores).sum(axis=-1, keepdims=True)))).sum(-1)
    expected = ce(np.asarray(model.cls), np.asarray(q_cls)[::-1]).mean()
    expected += ce(np.asarray(model.patches), np.asarray(q_patch))[[0, 1], [0, 1]].mean()
    actual = distillation_loss(model, None, masks, q_cls, q_patch, settings)
    np.testing.assert_allclose(float(actual.item()), expected, rtol=1e-5)
    changed = np.asarray(q_patch).copy()
    changed[0, 1] = [0, 0, 1]
    changed[1, 0] = [1, 0, 0]
    np.testing.assert_allclose(float(distillation_loss(model, None, masks, q_cls, mx.array(changed), settings).item()), expected, rtol=1e-5)


def test_resume_restores_optimizer_teacher_centers_rng_and_next_step(tmp_path):
    settings, student, teacher, optimizer, centers, images, masks = setup_models()
    centers, _ = train_step(student, teacher, optimizer, centers, images, images * .9, masks, settings)
    rng = np.random.default_rng(99)
    rng.normal(size=10)
    contract = {"test_only": "complete epoch resume"}
    save_resume(tmp_path, 1, student, teacher, optimizer, centers, rng,
                {"contract_sha256": digest(contract), "history": [], "steps": 1})
    path, meta = read_resume(tmp_path, contract)
    with pytest.raises(ValueError, match="different data/settings"):
        read_resume(tmp_path, {"changed": True})
    _, restored_s, restored_t, restored_o, _, _, _ = setup_models()
    restored_c = restore_resume(path, restored_s, restored_t, restored_o)
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = meta["rng_state"]
    np.testing.assert_array_equal(rng.normal(size=10), restored_rng.normal(size=10))
    centers, metrics = train_step(student, teacher, optimizer, centers, images, images * .9, masks, settings)
    restored_c, restored_metrics = train_step(restored_s, restored_t, restored_o, restored_c, images, images * .9, masks, settings)
    np.testing.assert_allclose(metrics["loss"], restored_metrics["loss"], atol=1e-6)
    for original, restored in ((student.parameters(), restored_s.parameters()), (teacher.parameters(), restored_t.parameters()),
                               (optimizer.state, restored_o.state), (centers, restored_c)):
        other = dict(tree_flatten(restored))
        for key, value in tree_flatten(original):
            np.testing.assert_allclose(np.asarray(value), np.asarray(other[key]), atol=1e-6, rtol=1e-5)
