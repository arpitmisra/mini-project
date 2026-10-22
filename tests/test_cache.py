import json

import numpy as np
import pytest
from PIL import Image

from oct_dinov2_mlx.cache import FeatureCache, cache_lock, extract_cached, feature_identity
from oct_dinov2_mlx.data import KermanyOCTDataset
from oct_dinov2_mlx.preprocessing import preprocess, preprocessing_identity


class DeterministicTestExtractor:
    """Only exercises cache plumbing; these arrays are NOT DINOv2 embeddings."""
    metadata = {"type": "synthetic-test-double"}
    calls = 0

    def extract(self, images):
        self.calls += 1
        values = images.mean((1, 2, 3))[:, None] + np.arange(384, dtype=np.float32)[None] / 384
        return {"cls": values, "patches": np.broadcast_to(values[:, None], (len(images), 256, 384))}


@pytest.fixture
def cache_setup(oct_root, tmp_path):
    data = KermanyOCTDataset(oct_root, "train", preprocess, max_per_class=2)
    identity = feature_identity({"test_only": True}, preprocessing_identity({"image_size": 224, "resize_shortest": 256}), True, "float16")
    return data, identity, tmp_path / "features"


def test_roundtrip_reuse_without_model_or_images(cache_setup, monkeypatch):
    data, identity, directory = cache_setup
    extractor = DeterministicTestExtractor()
    path, stats = extract_cached(data, identity, directory, 2, 4, lambda: extractor)
    assert stats["extracted"] == 8 and stats["reused"] == 0
    cached = FeatureCache(path, identity)
    cls, labels = cached.load_cls()
    assert cls.shape == (8, 384)
    assert labels.tolist() == [r.binary_label for r in data.records]
    sample = cached.load_sample(data.records[0].sample_id)
    assert sample["patches"].shape == (256, 384)
    assert sample["patches"].dtype == np.float16
    monkeypatch.setattr(KermanyOCTDataset, "__getitem__", lambda *_: pytest.fail("cache hit decoded an image"))
    _, stats = extract_cached(data, identity, directory, 1, 4, lambda: pytest.fail("cache hit initialized model"))
    assert stats["reused"] == 8 and stats["extracted"] == 0


def test_corruption_repairs_only_affected_shard(cache_setup):
    data, identity, directory = cache_setup
    path, _ = extract_cached(data, identity, directory, 2, 4, DeterministicTestExtractor)
    index = json.loads(path.read_text())
    (path.parent / index["shards"][0]["file"]).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="Corrupt"):
        FeatureCache(path)
    _, stats = extract_cached(data, identity, directory, 2, 4, DeterministicTestExtractor)
    assert stats["extracted"] == 4 and stats["reused"] == 4


def test_image_content_change_invalidates_shard(cache_setup):
    data, identity, directory = cache_setup
    extract_cached(data, identity, directory, 2, 4, DeterministicTestExtractor)
    Image.new("RGB", (16, 16), "red").save(data.records[0].image_path)
    changed = KermanyOCTDataset(data.root, "train", preprocess, max_per_class=2)
    _, stats = extract_cached(changed, identity, directory, 2, 4, DeterministicTestExtractor)
    assert stats["extracted"] == 4 and stats["reused"] == 4


def test_identity_change_rejects_and_reextracts(cache_setup):
    data, identity, directory = cache_setup
    path, _ = extract_cached(data, identity, directory, 2, 4, DeterministicTestExtractor)
    other = {**identity, "model": {"test_only": "changed_revision"}}
    with pytest.raises(ValueError, match="mismatch"):
        FeatureCache(path, other)
    _, stats = extract_cached(data, other, directory, 2, 4, DeterministicTestExtractor)
    assert stats["extracted"] == 8


def test_interruption_resumes_completed_shards(cache_setup):
    data, identity, directory = cache_setup
    extractor = DeterministicTestExtractor()
    original = extractor.extract
    def fail(images):
        if extractor.calls == 2:
            raise RuntimeError("simulated interruption")
        return original(images)
    extractor.extract = fail
    with pytest.raises(RuntimeError, match="simulated"):
        extract_cached(data, identity, directory, 2, 4, lambda: extractor)
    _, stats = extract_cached(data, identity, directory, 2, 4, DeterministicTestExtractor)
    assert stats["extracted"] == 4 and stats["reused"] == 4


def test_lock_blocks_concurrent_writers(tmp_path):
    with cache_lock(tmp_path):
        with pytest.raises(RuntimeError, match="being written"):
            with cache_lock(tmp_path):
                pass
