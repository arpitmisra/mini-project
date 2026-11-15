import os

import numpy as np
import pytest

pytestmark = pytest.mark.mlx
if os.environ.get("OCT_TEST_MLX") != "1":
    pytest.skip("Opt in with OCT_TEST_MLX=1 in MLX environment", allow_module_level=True)

from oct_dinov2_mlx.common import require_mlx
require_mlx()
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from oct_dinov2_mlx.lora import LoRAQKV


def test_lora_initial_equivalence_update_freeze_key_preservation_and_merge():
    mx.random.seed(9)
    base = nn.Linear(8, 24)
    model = LoRAQKV(base, rank=2, alpha=2)
    x = mx.random.normal((2, 4, 8))
    expected = base(x)
    before = np.asarray(base.weight).copy()
    np.testing.assert_array_equal(np.asarray(model(x)), np.asarray(expected))
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert set(trainable) == {"q_a", "q_b", "v_a", "v_b"}
    optimizer = optim.AdamW(.01)
    def loss(m):
        return mx.mean((m(x) - 1) ** 2)
    for _ in range(2):
        _, grads = nn.value_and_grad(model, loss)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    np.testing.assert_array_equal(np.asarray(base.weight), before)
    assert not np.array_equal(np.asarray(model(x)[..., :8]), np.asarray(expected[..., :8]))
    np.testing.assert_array_equal(np.asarray(model(x)[..., 8:16]), np.asarray(expected[..., 8:16]))
    np.testing.assert_allclose(np.asarray(model(x)), np.asarray(model.merged()(x)), atol=1e-6, rtol=1e-5)
