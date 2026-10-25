"""These tests execute only on Apple Silicon; never report a Windows pass."""
import os
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.mlx
if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
    pytest.skip("Native Apple Silicon required", allow_module_level=True)
from oct_dinov2_mlx.common import require_mlx

require_mlx()  # A broken Mac setup must fail, not silently skip the native tests.
import mlx.core as mx

def test_architecture_shapes_batch_and_frozen_parameters():
    from mlx.utils import tree_flatten
    from mlx_vlm.models.dinov2.config import ModelConfig
    from mlx_vlm.models.dinov2.dinov2 import Model
    # Small random architecture test, never described as pretrained extraction.
    model = Model(ModelConfig(hidden_size=384, num_attention_heads=6, num_hidden_layers=1, image_size=518))
    model.freeze()
    model.eval()
    before = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())}
    assert not tree_flatten(model.trainable_parameters())
    for batch in (1, 2):
        result = model.forward_features(mx.random.normal((batch, 224, 224, 3)))
        mx.eval(result["x_norm_clstoken"], result["x_norm_patchtokens"])
        assert result["x_norm_clstoken"].shape == (batch, 384)
        assert result["x_norm_patchtokens"].shape == (batch, 256, 384)
        assert result["x_norm_regtokens"].shape == (batch, 0, 384)
        assert np.isfinite(np.asarray(result["x_norm_patchtokens"])).all()
    for key, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(before[key], np.asarray(value))


def test_complete_hf_weight_conversion_without_network():
    from mlx.utils import tree_flatten
    from mlx_vlm.models.dinov2.config import ModelConfig
    from mlx_vlm.models.dinov2.dinov2 import Model
    from oct_dinov2_mlx.backbone import expected_hf_shapes, validate_shapes
    model = Model(ModelConfig(hidden_size=384, num_attention_heads=6, num_hidden_layers=12, image_size=518))
    # Synthetic zeros validate conversion coverage, not pretrained model quality.
    weights = {key: mx.zeros(shape) for key, shape in expected_hf_shapes().items()}
    for i, name in enumerate(("query", "key", "value")):
        weights[f"encoder.layer.0.attention.attention.{name}.bias"] = mx.full((384,), i + 1)
    converted = model.sanitize(weights)
    validate_shapes(converted, {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())})
    model.load_weights(list(converted.items()), strict=True)
    mx.eval(model.parameters())
    np.testing.assert_array_equal(np.asarray(model.blocks[0].attn.qkv.bias), np.repeat([1, 2, 3], 384))


@pytest.mark.pretrained
@pytest.mark.skipif(os.environ.get("OCT_TEST_PRETRAINED") != "1", reason="Opt in with OCT_TEST_PRETRAINED=1; downloads ~89 MB weights, no dataset")
def test_real_pretrained_extractor():
    from oct_dinov2_mlx.backbone import DINOv2Extractor
    from oct_dinov2_mlx.config import ROOT, load_config
    extractor = DINOv2Extractor(load_config(ROOT / "configs/smoke_test.yaml")["model"])
    images = np.random.default_rng(123).normal(size=(2, 224, 224, 3)).astype(np.float32)
    batch = extractor.extract(images)
    single = extractor.extract(images[:1])
    assert batch["cls"].shape == (2, 384)
    assert batch["patches"].shape == (2, 256, 384)
    np.testing.assert_allclose(batch["cls"][:1], single["cls"], atol=5e-4, rtol=1e-3)
    np.testing.assert_allclose(batch["patches"][:1], single["patches"], atol=5e-4, rtol=1e-3)


@pytest.mark.pretrained
@pytest.mark.skipif(os.environ.get("OCT_TEST_PRETRAINED") != "1", reason="Explicit pretrained integration opt-in required")
def test_pretrained_end_to_end_and_cache_reuse(oct_root, tmp_path, monkeypatch):
    from oct_dinov2_mlx.config import ROOT, load_config
    from oct_dinov2_mlx import pipeline
    config = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(tmp_path / "run"))
    config["dataset"]["max_per_class"] = 1
    config["features"]["cache_dir"] = str(tmp_path / "features")
    config["finetuning"]["epochs"] = 1
    config["finetuning"]["max_steps_per_epoch"] = 1
    first = pipeline.run(config)
    assert first["extraction"]["test"]["extracted"] == 4
    assert (tmp_path / "run/finetuning/backbone.safetensors").is_file()
    assert not (tmp_path / "run/metrics").exists()
    # A completed adapted backbone must load without the original HF cache.
    from oct_dinov2_mlx import backbone
    monkeypatch.setattr(backbone, "checkpoint_files", lambda *_: pytest.fail("Adapted extraction requested original weights"))
    adapted = backbone.DINOv2Extractor(config["model"], checkpoint=first["checkpoint"])
    assert "adaptation_method" in adapted.metadata
    assert all(not key.startswith(("global_head", "patch_head")) for key in mx.load(first["checkpoint"]))
    del adapted
    from oct_dinov2_mlx import finetuning
    monkeypatch.setattr(finetuning, "train_step", lambda *a: pytest.fail("Completed run retrained backbone"))
    from oct_dinov2_mlx.representations import export_dense_sample
    metadata = export_dense_sample(tmp_path / "run/features_manifest.json", tmp_path / "dense.npz")
    assert metadata["arrays"]["spatial_grid"]["shape"] == [16, 16, 384]
    monkeypatch.setattr(pipeline, "DINOv2Extractor", lambda *a, **k: pytest.fail("Cache reuse reloaded backbone"))
    second = pipeline.run(config)
    assert second["extraction"]["test"]["reused"] == 4
