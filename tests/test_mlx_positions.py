"""Check position interpolation values AND gradients, including Mac dispatch."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.mlx
if os.environ.get("OCT_TEST_MLX") != "1":
    pytest.skip("Opt in with OCT_TEST_MLX=1", allow_module_level=True)

from oct_dinov2_mlx.common import require_mlx
require_mlx()
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.dinov2.config import ModelConfig
from oct_dinov2_mlx.mlx_model import DifferentiableDINOv2, resize_positions


def reference_matrix(source, target):
    """Independent float64 cubic resampling matrix, including border folding."""
    matrix = np.zeros((target, source), dtype=np.float64)
    for row in range(target):
        center = (row + 0.5) * source / target - 0.5
        for tap in range(int(np.floor(center)) - 1, int(np.floor(center)) + 3):
            distance = abs(center - tap)
            if distance <= 1:
                value = 1.25 * distance**3 - 2.25 * distance**2 + 1
            else:
                value = -.75 * distance**3 + 3.75 * distance**2 - 6 * distance + 3
            matrix[row, np.clip(tap, 0, source - 1)] += value
    return matrix


@pytest.mark.parametrize("size", [(16, 16), (3, 5), (42, 40)])
def test_bicubic_values_and_gradients_match_reference(size):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(1, 37, 37, 3)).astype(np.float32)
    weights = rng.normal(size=(1, *size, 3)).astype(np.float32)
    rows, columns = reference_matrix(37, size[0]), reference_matrix(37, size[1])
    reference = np.einsum("ih,bhwc,jw->bijc", rows, data, columns)
    # For a linear resampler, the adjoint is its matrix transpose.
    reference_gradient = np.einsum("ih,bijc,jw->bhwc", rows, weights, columns)
    actual = resize_positions(mx.array(data), *size)
    gradient = mx.grad(lambda x: mx.sum(resize_positions(x, *size) * mx.array(weights)))(mx.array(data))
    mx.eval(actual, gradient)
    np.testing.assert_allclose(np.asarray(actual), reference, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(gradient), reference_gradient, atol=3e-5, rtol=3e-5)


def test_model_position_gradient_never_dispatches_custom_metal_kernel(monkeypatch):
    from mlx_vlm.models import kernels
    monkeypatch.setattr(kernels, "_HAS_METAL", True)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **k: pytest.fail("Custom Metal interpolation has no VJP"))
    model = DifferentiableDINOv2(ModelConfig(hidden_size=12, num_attention_heads=3,
                                           num_hidden_layers=1, image_size=518))
    tokens = mx.zeros((1, 257, 12))
    _, grads = nn.value_and_grad(model, lambda m: mx.sum(m.interpolate_pos_encoding(tokens, 224, 224)))(model)
    mx.eval(grads)
    gradient = np.asarray(grads["pos_embed"])
    assert np.isfinite(gradient).all()
    assert np.any(gradient[:, 1:] != 0)
    np.testing.assert_allclose(gradient[:, 1:].astype(np.float64).sum(axis=1), 256, atol=2e-4)


def test_native_grid_keeps_positions_exactly():
    model = DifferentiableDINOv2(ModelConfig(hidden_size=12, num_attention_heads=3,
                                           num_hidden_layers=1, image_size=518))
    actual = model.interpolate_pos_encoding(mx.zeros((1, 1370, 12)), 518, 518)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(model.pos_embed))
