"""Portable adaptation contracts; no actual MLX weights or features here."""
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from oct_dinov2_mlx import pipeline
from oct_dinov2_mlx.adaptation import METHOD, adapted_identity, read_adapted_checkpoint, training_contract
from oct_dinov2_mlx.augmentations import paired_views, patch_masks
from oct_dinov2_mlx.backbone import model_identity
from oct_dinov2_mlx.common import digest, file_digest, write_json
from oct_dinov2_mlx.config import ROOT, load_config
from oct_dinov2_mlx.data import KermanyOCTDataset


HF_CONFIG = {"model_type": "dinov2", "hidden_size": 384, "num_hidden_layers": 12,
             "num_attention_heads": 6, "image_size": 518, "patch_size": 14,
             "num_channels": 3, "mlp_ratio": 4, "layer_norm_eps": 1e-6,
             "hidden_act": "gelu", "qkv_bias": True, "use_swiglu_ffn": False}


def fake_checkpoint(path, config, contract=None, payload=b"test-only fake native weights"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    contract = contract or {"test_only": True}
    write_json(path.with_suffix(".json"), {
        "format_version": 1, "status": "complete", "method": METHOD,
        "representation_source": "ema_teacher", "base_model": model_identity(config["model"]),
        "weights_sha256": file_digest(path), "hf_config": HF_CONFIG,
        "contract": contract, "contract_sha256": digest(contract),
    })
    return path


def test_training_contract_excludes_validation_and_test(oct_root):
    cfg = load_config(ROOT / "configs/smoke_test.yaml")
    records = KermanyOCTDataset(oct_root, "train", max_per_class=1).records
    contract = training_contract(cfg, records)
    assert contract["labels_used_for_loss"] is False
    assert contract["train_count"] == 4
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="forbidden"):
            training_contract(cfg, records + [replace(records[0], split=split, source_split=split)])
        with pytest.raises(ValueError, match="forbidden"):
            training_contract(cfg, [replace(records[0], source_split=split)] + records[1:])
    changed = [replace(records[0], sha256="changed")] + records[1:]
    assert digest(training_contract(cfg, changed)) != digest(contract)
    cfg["finetuning"]["learning_rate"] *= 2
    assert digest(training_contract(cfg, records)) != digest(contract)


def test_adapted_hash_invalidates_feature_identity(tmp_path):
    cfg = load_config(ROOT / "configs/smoke_test.yaml")
    one = fake_checkpoint(tmp_path / "one.safetensors", cfg)
    two = fake_checkpoint(tmp_path / "two.safetensors", cfg, payload=b"different test weights")
    assert adapted_identity(one, cfg["model"]) != adapted_identity(two, cfg["model"])
    assert pipeline.identity_for(cfg, one) != pipeline.identity_for(cfg, two)
    assert pipeline.identity_for(cfg, one) != pipeline.identity_for(cfg)
    one.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_adapted_checkpoint(one, cfg["model"])


def test_adapted_checkpoint_requires_complete_matching_contract(tmp_path):
    cfg = load_config(ROOT / "configs/smoke_test.yaml")
    path = fake_checkpoint(tmp_path / "backbone.safetensors", cfg)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["contract"] = {"tampered": True}
    write_json(path.with_suffix(".json"), meta)
    with pytest.raises(ValueError, match="contract checksum"):
        read_adapted_checkpoint(path, cfg["model"])
    meta["status"] = "in_progress"
    write_json(path.with_suffix(".json"), meta)
    with pytest.raises(ValueError, match="completed"):
        read_adapted_checkpoint(path, cfg["model"])


def test_aligned_views_deterministic_and_spatially_matched():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (300, 400, 3), dtype=np.uint8))
    preprocessing = {"image_size": 224, "resize_shortest": 256}
    t, s = paired_views(image, preprocessing, np.random.default_rng(42), strength=0)
    np.testing.assert_array_equal(t, s)  # identical crop geometry, no photometric changes
    assert t.shape == (2, 224, 224, 3) and t.dtype == np.float32
    assert not np.array_equal(t[0], t[1])
    a = paired_views(image, preprocessing, np.random.default_rng(42))
    b = paired_views(image, preprocessing, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    assert np.isfinite(a).all()


def test_patch_masks_are_bounded_and_reproducible():
    a = patch_masks(4, 256, .4, np.random.default_rng(42))
    np.testing.assert_array_equal(a, patch_masks(4, 256, .4, np.random.default_rng(42)))
    assert a.dtype == bool and a.shape == (4, 256)
    assert (a.sum(1) == 102).all()
    with pytest.raises(ValueError):
        patch_masks(2, 1, .4, np.random.default_rng(42))


def test_extract_missing_adaptation_fails_before_dataset_download(monkeypatch, tmp_path):
    cfg = load_config(ROOT / "configs/default.yaml", output_dir=str(tmp_path))
    monkeypatch.setattr(pipeline, "require_mlx", lambda: None)
    monkeypatch.setattr(pipeline, "prepare_data", lambda *_: pytest.fail("Touched dataset without checkpoint"))
    with pytest.raises(FileNotFoundError, match="Fine-tuned backbone unavailable"):
        pipeline.run(cfg, "extract")


def test_inspect_api_is_also_bounded_and_offline(oct_root, tmp_path):
    cfg = load_config(ROOT / "configs/default.yaml", str(oct_root), str(tmp_path))
    result = pipeline.run(cfg, "inspect")
    assert result["images_decoded"] == {"train": 16, "val": 16, "test": 16}
    assert cfg["dataset"]["max_per_class"] is None  # caller's config is not mutated


def test_rejected_finetuning_preserves_prior_config(oct_root, tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(tmp_path))
    write_json(tmp_path / "config.resolved.json", {"previous_run": True})
    write_json(tmp_path / "dataset_summary.json", {"previous_dataset": True})
    fake_module = ModuleType("oct_dinov2_mlx.finetuning")
    def reject(*args):
        raise ValueError("different data/settings")
    fake_module.finetune = reject
    monkeypatch.setitem(sys.modules, "oct_dinov2_mlx.finetuning", fake_module)
    monkeypatch.setattr(pipeline, "require_mlx", lambda: None)
    with pytest.raises(ValueError, match="different data/settings"):
        pipeline.run(cfg)
    assert json.loads((tmp_path / "config.resolved.json").read_text()) == {"previous_run": True}
    assert json.loads((tmp_path / "dataset_summary.json").read_text()) == {"previous_dataset": True}


def test_default_flow_finetunes_train_only_then_extracts_adapted_features(oct_root, tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(tmp_path / "run"))
    cfg["dataset"]["max_per_class"] = 1
    cfg["features"]["cache_dir"] = str(tmp_path / "features")
    adapted = Path(cfg["runtime"]["output_dir"]) / "finetuning/backbone.safetensors"
    calls = []
    fake_module = ModuleType("oct_dinov2_mlx.finetuning")
    def fake_train(dataset, config, output_dir):
        assert dataset.split == "train"
        assert all(r.source_split == "train" for r in dataset.records)
        calls.append("finetune")
        return fake_checkpoint(adapted, config, training_contract(config, dataset.records))
    fake_module.finetune = fake_train
    monkeypatch.setitem(sys.modules, "oct_dinov2_mlx.finetuning", fake_module)
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = SimpleNamespace(get_peak_memory=lambda: 0)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx.core)
    monkeypatch.setattr(pipeline, "require_mlx", lambda: None)
    class FakeExtractor:
        metadata = {"test_only": True}
        def __init__(self, _, image_size, checkpoint):
            assert checkpoint == adapted and calls[0] == "finetune"
            calls.append("extractor")
        def extract(self, images):
            cls = images.mean((1, 2, 3))[:, None] + np.arange(384, dtype=np.float32)[None]
            return {"cls": cls, "patches": np.broadcast_to(cls[:, None], (len(images), 256, 384))}
    monkeypatch.setattr(pipeline, "DINOv2Extractor", FakeExtractor)
    result = pipeline.run(cfg)
    assert calls == ["finetune", "extractor"]
    manifest = json.loads(Path(result["feature_manifest"]).read_text())
    assert manifest["identity"]["model"]["weights_sha256"] == file_digest(adapted)
    assert not (Path(cfg["runtime"]["output_dir"]) / "metrics").exists()
    original_config = (Path(cfg["runtime"]["output_dir"]) / "config.resolved.json").read_bytes()
    monkeypatch.setattr(pipeline, "DINOv2Extractor", lambda *a, **k: pytest.fail("Valid cache reloaded backbone"))
    fake_module.finetune = lambda *a: pytest.fail("Extract retrained backbone")
    result = pipeline.run(cfg, "extract")
    assert all(v["extracted"] == 0 and v["reused"] == 4 for v in result["extraction"].values())
    assert (Path(cfg["runtime"]["output_dir"]) / "config.resolved.json").read_bytes() == original_config
