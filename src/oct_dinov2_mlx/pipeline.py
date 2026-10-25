from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np

from .backbone import DINOv2Extractor, model_identity
from .cache import FeatureCache, extract_cached, feature_identity
from .common import require_mlx, system_info, write_json
from .config import ROOT, load_config
from .data import KermanyOCTDataset, Record, check_split_integrity, derive_validation, resolve_dataset
from .adaptation import adapted_identity, read_adapted_checkpoint
from .preprocessing import preprocess, preprocessing_identity


def identity_for(config, checkpoint=None):
    model = adapted_identity(checkpoint, config["model"]) if checkpoint else model_identity(config["model"])
    return feature_identity(model, preprocessing_identity(config["preprocessing"]), config["features"]["save_patches"], config["features"]["patch_dtype"])


def prepare_data(config):
    root, source = resolve_dataset(config["dataset"])
    if (root / "SYNTHETIC_DATASET.json").exists():
        source["synthetic"] = True
        source["purpose"] = "plumbing smoke test only; these are not OCT scans"
    transform = partial(preprocess, **config["preprocessing"])
    datasets = {s: KermanyOCTDataset(root, s, transform, config["dataset"]["max_per_class"], config["dataset"]["hash_images"]) for s in ("train", "val", "test")}
    check_split_integrity(datasets)
    if config["dataset"]["validation_fraction"] is not None:
        datasets = derive_validation(datasets, config["dataset"]["validation_fraction"], config["runtime"]["seed"])
        check_split_integrity(datasets)
    summary = {"root": str(root), "source": source, "splits": {s: d.statistics() for s, d in datasets.items()}, "integrity_scope": "selected images only" if config["dataset"]["max_per_class"] else "all indexed images", "content_hash_check": config["dataset"]["hash_images"], "patient_independence": "not established; no patient IDs inferred"}
    print(json.dumps(summary, indent=2))
    return datasets, summary


def load_caches(manifest_path, identity):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity"] != json.loads(json.dumps(identity)):
        raise ValueError("Manifest identity differs from this model/preprocessing/feature configuration")
    indices = {}
    for split in ("train", "val", "test"):
        path = Path(manifest["indices"][split])
        indices[split] = path if path.is_absolute() else manifest_path.parent / path
    caches = {s: FeatureCache(indices[s], identity) for s in ("train", "val", "test")}
    datasets = {}
    for split, cache in caches.items():
        if cache.index["split"] != split:
            raise ValueError(f"Manifest assigns {cache.index['split']} features to {split}")
        datasets[split] = KermanyOCTDataset(".", split, records=[Record(**r) for r in cache.index["records"]])
        if any(r.split != split for r in datasets[split].records):
            raise ValueError("Inconsistent split metadata in cache")
    check_split_integrity(datasets)
    return caches, manifest


def run(config, stage="all", checkpoint_path=None):
    if stage not in ("all", "system", "inspect", "finetune", "extract"):
        raise ValueError("Classification stages were removed. Use all, finetune, extract, inspect, or system.")
    config = deepcopy(config)
    output = Path(config["runtime"]["output_dir"])
    info = system_info()
    print(f"Host: {info['platform']}; Python architecture: {info['machine']}")
    if stage == "system":
        print(json.dumps(info, indent=2))
        return info
    if checkpoint_path is not None and stage != "extract":
        raise ValueError("--checkpoint is for extract only; all/finetune adapt the pinned pretrained backbone")
    if stage == "inspect":
        config["dataset"]["allow_download"] = False
        config["dataset"]["max_per_class"] = min(config["dataset"]["max_per_class"] or 4, 4)
    else:
        require_mlx()  # Always precedes dataset/model access.
    checkpoint = Path(checkpoint_path or output / "finetuning/backbone.safetensors")
    if stage == "extract":
        # Never silently substitute pretrained weights for an absent adapted model.
        read_adapted_checkpoint(checkpoint, config["model"])
    datasets, dataset_summary = prepare_data(config)
    output.mkdir(parents=True, exist_ok=True)
    if stage == "inspect":
        write_json(output / "inspection_dataset_summary.json", dataset_summary)
        write_json(output / "inspection_config.resolved.json", config)
        checked = {}
        for split, dataset in datasets.items():
            for i in range(len(dataset)):
                x = dataset[i]["image"]
                if x.shape != (config["preprocessing"]["image_size"],) * 2 + (3,) or not np.isfinite(x).all():
                    raise ValueError("Invalid preprocessing output")
            checked[split] = len(dataset)
        result = {"status": "portable dataset/preprocessing check passed; no model executed", "images_decoded": checked, "system": info}
        write_json(output / "portable_smoke.json", result)
        print(json.dumps(result, indent=2))
        return result
    if stage in ("all", "finetune"):
        from .finetuning import finetune
        checkpoint = finetune(datasets["train"], config, output)
    adaptation = read_adapted_checkpoint(checkpoint, config["model"])
    # A rejected/reconfigured run must not overwrite the earlier run's provenance.
    write_json(output / ("extraction_dataset_summary.json" if stage == "extract" else "dataset_summary.json"), dataset_summary)
    write_json(output / ("extraction_config.resolved.json" if stage == "extract" else "config.resolved.json"), config)
    metadata = {"system": info, "config": config, "dataset": dataset_summary,
                "checkpoint": str(checkpoint.resolve()), "adaptation": adaptation,
                "protocol": "Self-supervised adaptation on TRAIN only; EMA backbone dense extraction; no classifier"}
    if stage == "finetune":
        write_json(output / "experiment_metadata.json", metadata)
        return {"checkpoint": str(checkpoint)}
    identity = identity_for(config, checkpoint)
    manifest_path = output / "features_manifest.json"
    shared = []
    def factory():
        if not shared:
            shared.append(DINOv2Extractor(config["model"], config["preprocessing"]["image_size"], checkpoint=checkpoint))
        return shared[0]
    indices, stats = {}, {}
    for split, dataset in datasets.items():
        path, stat = extract_cached(dataset, identity, config["features"]["cache_dir"], config["features"]["batch_size"], config["features"]["shard_size"], factory)
        try:
            indices[split] = Path(os.path.relpath(path.resolve(), manifest_path.resolve().parent)).as_posix()
        except ValueError:  # Windows caches can be on a different drive.
            indices[split] = str(path.resolve())
        stats[split] = stat
        print(f"{split}: extracted {stat['extracted']}, reused {stat['reused']}")
    write_json(manifest_path, {"identity": identity, "indices": indices, "dataset": dataset_summary,
                               "adapted_checkpoint": str(checkpoint.resolve())})
    write_json(output / "extraction_benchmark.json", stats)
    metadata.update({"feature_manifest": str(manifest_path.resolve()), "feature_identity": identity})
    import mlx.core as mx
    metadata["mlx_peak_bytes"] = mx.get_peak_memory()
    write_json(output / ("extraction_metadata.json" if stage == "extract" else "experiment_metadata.json"), metadata)
    print(f"Dense representations saved. Manifest: {manifest_path}")
    return {"checkpoint": str(checkpoint), "feature_manifest": str(manifest_path), "extraction": stats}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on OCT with self-supervision, then save dense representations only")
    parser.add_argument("--config", default=str(ROOT / "configs/smoke_test.yaml"))
    parser.add_argument("--dataset-path", help="Existing local root overrides OCT_DATASET_PATH and YAML")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage", choices=["all", "system", "inspect", "finetune", "extract"], default="all",
                        help="all (default): fine-tune/reuse completed adaptation, then extract; no classifier")
    parser.add_argument("--checkpoint", help="For extract: existing fine-tuned backbone.safetensors (default: output-dir/finetuning/backbone.safetensors)")
    parser.add_argument("--offline", action="store_true", help="Disable both dataset and pretrained checkpoint network downloads")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.dataset_path, args.output_dir)
        if args.offline:
            config["dataset"]["allow_download"] = config["model"]["allow_download"] = False
        run(config, args.stage, args.checkpoint)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
