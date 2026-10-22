import json
from types import SimpleNamespace

import numpy as np
import pytest

from oct_dinov2_mlx.backbone import checkpoint_files, expected_hf_shapes, model_identity, validate_shapes
from oct_dinov2_mlx.config import ROOT, load_config
from oct_dinov2_mlx.evaluation import binary_metrics, save_evaluation
from oct_dinov2_mlx.pipeline import run


def test_checkpoint_coverage_rejects_missing_extra_transposed():
    shapes = expected_hf_shapes()
    actual = {k: SimpleNamespace(shape=v) for k, v in shapes.items()}
    validate_shapes(actual, shapes)
    removed = actual.pop("embeddings.cls_token")
    with pytest.raises(ValueError, match="missing"):
        validate_shapes(actual, shapes)
    actual["embeddings.cls_token"] = removed
    actual["classifier.weight"] = SimpleNamespace(shape=(2, 384))
    with pytest.raises(ValueError, match="unexpected"):
        validate_shapes(actual, shapes)
    actual.pop("classifier.weight")
    actual["embeddings.patch_embeddings.projection.weight"] = SimpleNamespace(shape=(384, 14, 14, 3))
    with pytest.raises(ValueError, match="shapes"):
        validate_shapes(actual, shapes)


def test_metrics_and_one_class():
    result = binary_metrics([0, 0, 1, 1], [[.9, .1], [.3, .7], [.2, .8], [.4, .6]])
    assert result["accuracy"] == .75
    assert result["specificity_normal"] == .5
    assert result["sensitivity_abnormal"] == 1
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    one = binary_metrics([0, 0], [[.9, .1], [.8, .2]])
    assert one["roc_auc"] is None and one["pr_auc_average_precision"] is None
    json.dumps(one, allow_nan=False)


def test_invalid_probabilities():
    with pytest.raises(ValueError, match="Invalid"):
        binary_metrics([0], [[np.nan, .1]])


def test_evaluation_artifacts(tmp_path):
    records = [{"sample_id": str(i), "source_split": "test", "original_class": c} for i, c in enumerate(["NORMAL", "CNV"])]
    save_evaluation(tmp_path, np.array([0, 1]), np.array([[.8, .2], [.1, .9]]), records)
    for name in ("results.json", "predictions.csv", "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "classification_report.txt"):
        assert (tmp_path / name).stat().st_size > 0


def test_offline_portable_smoke(oct_root, tmp_path):
    config = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(tmp_path / "out"))
    assert not config["dataset"]["allow_download"]
    result = run(config, "inspect")
    assert result["images_decoded"] == {"train": 16, "val": 16, "test": 16}
    assert (tmp_path / "out/portable_smoke.json").exists()


def test_path_precedence_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("OCT_DATASET_PATH", str(tmp_path / "environment"))
    config = load_config(ROOT / "configs/smoke_test.yaml")
    assert config["dataset"]["local_path"] == str(tmp_path / "environment")
    config = load_config(ROOT / "configs/smoke_test.yaml", str(tmp_path / "cli"))
    assert config["dataset"]["local_path"] == str(tmp_path / "cli")
    path = tmp_path / "bad.yaml"
    path.write_text("features:\n  shard_size: 0\n")
    with pytest.raises(ValueError, match="positive"):
        load_config(path)
    config["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned"):
        model_identity(config["model"])


def test_local_model_fingerprint_requires_no_weights(tmp_path):
    config = load_config(ROOT / "configs/smoke_test.yaml")["model"]
    original = model_identity(config)
    config["local_path"] = str(tmp_path / "absent-model-folder")
    assert model_identity(config) == original


def test_noncanonical_weights_rejected_before_model_import(tmp_path):
    config = load_config(ROOT / "configs/smoke_test.yaml")["model"]
    config["local_path"] = str(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not the pretrained checkpoint")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint_files(config)
