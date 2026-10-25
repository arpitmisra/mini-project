"""Regression tests for first-run errors and moving a research run to another Mac."""
import json
import shutil

import numpy as np
import pytest
import yaml

from oct_dinov2_mlx import pipeline
from oct_dinov2_mlx.cache import extract_cached
from oct_dinov2_mlx.common import write_json
from oct_dinov2_mlx.config import ROOT, load_config


@pytest.mark.parametrize("stage", ["all", "extract", "finetune"])
def test_unsupported_host_fails_before_dataset_access(stage, monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/default.yaml", output_dir=str(tmp_path))
    def unsupported():
        raise RuntimeError("Apple Silicon required")
    monkeypatch.setattr(pipeline, "require_mlx", unsupported)
    monkeypatch.setattr(pipeline, "prepare_data", lambda *_: pytest.fail("Dataset accessed before host check"))
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        pipeline.run(config, stage)


@pytest.mark.parametrize("supplied", [
    ["not", "a", "mapping"],
    {"features": {"batch_size": True}},
    {"features": {"cache_dir": None}},
    {"preprocessing": {"image_size": 224.0}},
    {"preprocessing": {"resize_shortest": "256"}},
    {"finetuning": {"learning_rate": float("nan")}},
    {"finetuning": {"learning_rate": "1e-3"}},
    {"finetuning": {"weight_decay": float("inf")}},
    {"dataset": {"validation_fraction": "0.1"}},
    {"dataset": {"version": -1}},
    {"finetuning": {"mask_ratio": 1}},
    {"finetuning": {"teacher_momentum": 0}},
    {"finetuning": {"student_temperature": 0}},
    {"finetuning": {"patch_loss_weight": 0}},
    {"finetuning": {"max_steps_per_epoch": True}},
    {"finetuning": {"augmentation_strength": 2}},
    {"finetuning": {"prototypes": 1}},
    {"features": {"save_patches": False}},
])
def test_bad_configuration_reports_value_error(supplied, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(supplied), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


class SyntheticCacheExtractor:
    metadata = {"test_only": True}

    def extract(self, images):
        # This tests file/control flow only; these are not DINOv2 embeddings.
        cls = images.mean((1, 2, 3))[:, None] + np.arange(384, dtype=np.float32)[None] / 384
        return {"cls": cls, "patches": np.broadcast_to(cls[:, None], (len(images), 256, 384))}


def make_cached_run(oct_root, directory):
    config = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(directory / "run"))
    config["dataset"]["max_per_class"] = 1
    identity = pipeline.identity_for(config)
    datasets, summary = pipeline.prepare_data(config)
    indices = {}
    for split, data in datasets.items():
        path, _ = extract_cached(data, identity, directory / "features", 2, 4, SyntheticCacheExtractor)
        indices[split] = "../" + path.relative_to(directory).as_posix()
    manifest = directory / "run/features_manifest.json"
    write_json(manifest, {"identity": identity, "indices": indices, "dataset": summary})
    return config, manifest


def test_relative_manifest_survives_moving_outputs(oct_root, tmp_path, monkeypatch):
    config, manifest = make_cached_run(oct_root, tmp_path / "original")
    moved = tmp_path / "professor-mac"
    shutil.copytree(manifest.parent.parent, moved)
    # Relative paths must resolve beside the manifest, regardless of the shell CWD.
    monkeypatch.chdir(tmp_path)
    caches, _ = pipeline.load_caches(moved / "run/features_manifest.json", pipeline.identity_for(config))
    assert all(cache.path.is_relative_to(moved) for cache in caches.values())
    assert caches["train"].load_cls()[0].shape == (4, 384)


@pytest.mark.parametrize("stage", ["train", "evaluate", "predict"])
def test_classifier_stages_are_rejected(stage, tmp_path):
    config = load_config(ROOT / "configs/smoke_test.yaml", output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Classification stages were removed"):
        pipeline.run(config, stage)
