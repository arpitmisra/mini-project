import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oct_dinov2_mlx.data import BINARY_CLASS_MAPPING, CLASSES, KAGGLE_DATASET, KermanyOCTDataset, check_split_integrity, derive_validation, discover_root, resolve_dataset
from oct_dinov2_mlx.preprocessing import preprocess


def test_local_preferred_and_nested(oct_root):
    def forbidden(_):
        pytest.fail("Local mode must never call KaggleHub")
    root, metadata = resolve_dataset({"local_path": str(oct_root.parent.parent), "allow_download": True}, forbidden)
    assert root == oct_root
    assert metadata["source"] == "local"


def test_unavailable_path_falls_back(oct_root, tmp_path):
    calls = []
    def download(handle):
        calls.append(handle)
        return oct_root.parent
    with pytest.warns(UserWarning):
        root, _ = resolve_dataset({"local_path": str(tmp_path / "missing"), "allow_download": True, "version": 3}, download)
    assert root == oct_root
    assert calls == [KAGGLE_DATASET + "/versions/3"]


def test_offline_missing_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="disabled"):
        resolve_dataset({"local_path": tmp_path / "missing", "allow_download": False}, lambda _: pytest.fail("network"))


def test_existing_invalid_does_not_download(tmp_path):
    with pytest.raises(ValueError, match="Expected one"):
        resolve_dataset({"local_path": tmp_path, "allow_download": True}, lambda _: pytest.fail("network"))


def test_download_failure_actionable():
    def fail(_):
        raise OSError("401")
    with pytest.raises(RuntimeError, match="kagglehub.login"):
        resolve_dataset({"local_path": None, "allow_download": True}, fail)


def test_missing_class(oct_root):
    shutil.rmtree(oct_root / "val/DME")
    with pytest.raises(ValueError, match="val.*DME"):
        discover_root(oct_root)


def test_ambiguous_roots(oct_root):
    shutil.copytree(oct_root, oct_root.parent / "copy")
    with pytest.raises(ValueError, match="found 2"):
        discover_root(oct_root.parent)


def test_lazy_labels_bounded_loading(oct_root, monkeypatch):
    real_open = Image.open
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("Images must load lazily"))
    dataset = KermanyOCTDataset(oct_root, "train", preprocess, max_per_class=2)
    assert len(dataset) == 8
    assert dataset.available_counts == {name: 6 for name in CLASSES}
    assert all(r.binary_label == BINARY_CLASS_MAPPING[r.original_class] for r in dataset.records)
    assert len({r.sample_id for r in dataset.records}) == 8
    monkeypatch.setattr(Image, "open", real_open)
    sample = dataset[0]
    assert sample["image"].shape == (224, 224, 3)
    assert sample["image"].dtype == np.float32
    assert sample["original_class"] == "NORMAL"
    Path(sample["image_path"]).unlink()
    with pytest.raises(ValueError, match="Cannot read OCT image"):
        dataset[0]


def test_content_overlap_fails(oct_root):
    shutil.copyfile(oct_root / "train/NORMAL/NORMAL-000.png", oct_root / "test/NORMAL/NORMAL-000.png")
    datasets = {s: KermanyOCTDataset(oct_root, s, max_per_class=1) for s in ("train", "test")}
    with pytest.raises(ValueError, match="identical image bytes"):
        check_split_integrity(datasets)


def test_train_derived_val_is_reproducible(oct_root):
    datasets = {s: KermanyOCTDataset(oct_root, s) for s in ("train", "val", "test")}
    with pytest.warns(UserWarning):
        a = derive_validation(datasets, .33, 42)
    with pytest.warns(UserWarning):
        b = derive_validation(datasets, .33, 42)
    assert a["val"].records == b["val"].records
    assert all(r.source_split == "train" for r in a["val"].records)
    assert a["test"].records == datasets["test"].records
    check_split_integrity(a)


def test_preprocessing_deterministic_grayscale():
    image = Image.fromarray(np.full((20, 40), 128, np.uint8))
    x = preprocess(image)
    np.testing.assert_array_equal(x, preprocess(image))
    expected = (np.float32(128 / 255) - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    np.testing.assert_allclose(x[0, 0], expected, atol=1e-6)
    assert np.isfinite(x).all()
