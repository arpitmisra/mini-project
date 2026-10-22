"""Content-addressed, atomic, resumable feature shards, with lazy model construction."""
from __future__ import annotations

import json
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .common import digest, file_digest, write_json

CACHE_VERSION = 1


def feature_identity(model, preprocessing, save_patches, patch_dtype):
    return {"schema": CACHE_VERSION, "model": model, "preprocessing": preprocessing, "save_patches": save_patches, "cls_dtype": "float32", "patch_dtype": patch_dtype}


@contextmanager
def cache_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Feature cache is being written: {lock}. If an earlier process crashed, verify it has stopped before removing only this lock file.") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def validate_shard(path, metadata, identity, records):
    try:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if meta["identity"] != json.loads(json.dumps(identity)) or meta["records"] != records or meta["sha256"] != file_digest(path):
            return False
        with np.load(path, allow_pickle=False) as saved:
            n = len(records)
            if saved["cls"].shape != (n, 384) or saved["cls"].dtype != np.float32 or not np.isfinite(saved["cls"]).all():
                return False
            if saved["labels"].tolist() != [r["binary_label"] for r in records]:
                return False
            if saved["sample_ids"].tolist() != [r["sample_id"] for r in records]:
                return False
            if saved["original_classes"].tolist() != [r["original_class"] for r in records]:
                return False
            if identity["save_patches"]:
                p = (identity["preprocessing"]["image_size"] // 14) ** 2
                if saved["patches"].shape != (n, p, 384) or saved["patches"].dtype != np.dtype(identity["patch_dtype"]) or not np.isfinite(saved["patches"]).all():
                    return False
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def extract_cached(dataset, identity, cache_dir, batch_size, shard_size, extractor_factory):
    start = time.perf_counter()
    directory = Path(cache_dir) / digest(identity)[:20] / dataset.split
    all_records = [r.identity() for r in dataset.records]
    index_path = directory / ("index-" + digest(all_records)[:20] + ".json")
    entries, extracted, reused = [], 0, 0
    extractor = None
    with cache_lock(directory):
        for offset in range(0, len(dataset), shard_size):
            end = min(len(dataset), offset + shard_size)
            records = all_records[offset:end]
            key = digest({"identity": identity, "records": records})
            path, meta_path = directory / f"{key}.npz", directory / f"{key}.json"
            if validate_shard(path, meta_path, identity, records):
                reused += len(records)
            else:
                if extractor is None:
                    extractor = extractor_factory()
                cls_batches, patch_batches = [], []
                for first in range(offset, end, batch_size):
                    images = np.stack([dataset[i]["image"] for i in range(first, min(first + batch_size, end))])
                    try:
                        features = extractor.extract(images)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Feature batch failed at {first}; completed shards are reusable. If memory is exhausted, reduce features.batch_size. {exc}") from exc
                    cls_batches.append(np.asarray(features["cls"], dtype=np.float32))
                    if identity["save_patches"]:
                        patch_batches.append(np.asarray(features["patches"], dtype=identity["patch_dtype"]))
                arrays = {"cls": np.concatenate(cls_batches), "labels": np.array([r["binary_label"] for r in records], np.int64), "sample_ids": np.array([r["sample_id"] for r in records]), "original_classes": np.array([r["original_class"] for r in records])}
                if patch_batches:
                    arrays["patches"] = np.concatenate(patch_batches)
                if not all(np.isfinite(arrays[k]).all() for k in ("cls", "patches") if k in arrays):
                    raise ValueError("Features overflowed or contain NaN; use float32 patches if necessary")
                temp = path.with_suffix(f".{os.getpid()}.tmp")
                with temp.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temp, path)
                write_json(meta_path, {"identity": identity, "records": records, "sha256": file_digest(path), "backbone": extractor.metadata})
                if not validate_shard(path, meta_path, identity, records):
                    raise ValueError(f"Feature shard validation failed: {path}")
                extracted += len(records)
            entries.append({"file": path.name, "metadata": meta_path.name, "count": len(records)})
        elapsed = time.perf_counter() - start
        index = {"schema": CACHE_VERSION, "identity": identity, "split": dataset.split, "count": len(dataset), "records": [r.metadata() for r in dataset.records], "shards": entries}
        write_json(index_path, index)
    stats = {"extracted": extracted, "reused": reused, "elapsed_seconds": elapsed, "extraction_and_io_images_per_second": extracted / elapsed if extracted else None, "cache_bytes": sum((directory / s["file"]).stat().st_size for s in entries)}
    return index_path, stats


class FeatureCache:
    def __init__(self, index_path, expected_identity=None, verify=True):
        self.path = Path(index_path)
        self.index = json.loads(self.path.read_text(encoding="utf-8"))
        if self.index.get("schema") != CACHE_VERSION:
            raise ValueError("Unknown cache schema")
        if expected_identity and self.index["identity"] != json.loads(json.dumps(expected_identity)):
            raise ValueError("Feature cache model/preprocessing/storage configuration mismatch")
        if self.index["count"] != len(self.index["records"]) or self.index["count"] != sum(s["count"] for s in self.index["shards"]):
            raise ValueError("Incomplete feature index")
        if verify:
            offset = 0
            for entry in self.index["shards"]:
                expected = []
                for record in self.index["records"][offset:offset + entry["count"]]:
                    item = {k: v for k, v in record.items() if k != "image_path"}
                    if item.get("sha256"):
                        item.pop("mtime_ns")
                    expected.append(item)
                shard = self.path.parent / entry["file"]
                if not validate_shard(shard, self.path.parent / entry["metadata"], self.index["identity"], expected):
                    raise ValueError(f"Corrupt or stale feature shard: {shard}. Rerun extraction to repair it.")
                offset += entry["count"]

    def load_cls(self):
        # Dense spatial tensors are never concatenated across the dataset.
        xs, ys = [], []
        for entry in self.index["shards"]:
            with np.load(self.path.parent / entry["file"], allow_pickle=False) as data:
                xs.append(data["cls"])
                ys.append(data["labels"])
        return np.concatenate(xs), np.concatenate(ys)

    def load_sample(self, sample_id):
        for entry in self.index["shards"]:
            with np.load(self.path.parent / entry["file"], allow_pickle=False) as data:
                ids = data["sample_ids"].tolist()
                if sample_id in ids:
                    i = ids.index(sample_id)
                    return {key: data[key][i] for key in data.files}
        raise KeyError(sample_id)
