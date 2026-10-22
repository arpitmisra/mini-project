"""Export a cached dense representation without a model or original images."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .cache import FeatureCache
from .common import file_digest, write_json
from .localization import patches_to_grid


def export_dense_sample(manifest_path, output_path, split="test", sample_id=None):
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Dense representation output must end in .npz")
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Export already exists; choose a new output path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["identity"]["save_patches"]:
        raise ValueError("This cache contains CLS only; extract with features.save_patches=true")
    index_path = Path(manifest["indices"][split])
    if not index_path.is_absolute():
        index_path = manifest_path.parent / index_path
    # Verify the selected shard below. Do not read every dense shard for one scan.
    cache = FeatureCache(index_path, manifest["identity"], verify=False)
    if cache.index["split"] != split or not cache.index["records"]:
        raise ValueError("Requested split is empty or differs from the feature index")
    sample_id = sample_id or cache.index["records"][0]["sample_id"]
    record = next((r for r in cache.index["records"] if r["sample_id"] == sample_id), None)
    if record is None:
        raise ValueError(f"Sample {sample_id!r} is not present in split {split}")
    if record["split"] != split:
        raise ValueError("Sample split differs from the requested split")
    sample = cache.load_sample(sample_id, verify=True)
    image_size = cache.index["identity"]["preprocessing"]["image_size"]
    grid = patches_to_grid(sample["patches"], image_size)
    rows, cols = np.indices(grid.shape[:2], dtype=np.int32)
    row_col = np.column_stack((rows.ravel(), cols.ravel()))
    x, y = row_col[:, 1] * 14, row_col[:, 0] * 14
    arrays = {
        "cls": sample["cls"], "patches": sample["patches"], "spatial_grid": grid,
        "patch_row_col": row_col,
        "patch_bounds_xyxy": np.column_stack((x, y, x + 14, y + 14)),
        "sample_id": np.asarray(sample_id),
        "original_class": np.asarray(record["original_class"]),
        "binary_label": np.asarray(record["binary_label"], dtype=np.int64),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "format_version": 1, "sample": record,
        "feature_identity": cache.index["identity"],
        "dataset_source": manifest.get("dataset", {}).get("source", {}),
        "source_manifest": str(manifest_path), "source_index": str(index_path.resolve()),
        "output": str(output_path), "sha256": file_digest(output_path),
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "coordinates": {
            "space": "preprocessed center crop; not original image pixels",
            "image_size": image_size, "patch_size": 14,
            "order": "row-major: token_index = row * grid_width + column",
            "bounds": "[x0,y0,x1,y1], upper bounds exclusive",
            "interpretation": "Patch location is its input anchor; self-attention includes context from other patches.",
        },
        "precision": "Preserves cache dtype; float16 patches were rounded from float32 inference values.",
        "use": "Dense features for research; not a lesion mask or an attention/CAM heatmap.",
    }
    write_json(metadata_path, metadata)
    return metadata
