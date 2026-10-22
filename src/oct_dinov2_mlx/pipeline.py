from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

from .backbone import DINOv2Extractor, model_identity
from .cache import FeatureCache, extract_cached, feature_identity
from .common import require_mlx, system_info, write_json
from .config import ROOT, load_config
from .data import KermanyOCTDataset, Record, check_split_integrity, derive_validation, resolve_dataset
from .evaluation import save_evaluation
from .preprocessing import preprocess, preprocessing_identity


def identity_for(config):
    return feature_identity(model_identity(config["model"]), preprocessing_identity(config["preprocessing"]), config["features"]["save_patches"], config["features"]["patch_dtype"])


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


def run(config, stage="all", manifest_path=None, image_path=None, checkpoint_path=None):
    output = Path(config["runtime"]["output_dir"])
    info = system_info()
    print(f"Host: {info['platform']}; Python architecture: {info['machine']}")
    if stage == "system":
        print(json.dumps(info, indent=2))
        return info
    if stage in ("all", "extract", "train", "evaluate", "predict"):
        require_mlx()  # Fail before any dataset/model download on unsupported hosts.
    identity = identity_for(config)
    if stage == "predict":
        if not image_path or not checkpoint_path:
            raise ValueError("predict requires --image and --checkpoint")
        from PIL import Image
        from .training import load_classifier, predict_probabilities
        model = load_classifier(checkpoint_path, identity)
        extractor = DINOv2Extractor(config["model"], config["preprocessing"]["image_size"])
        with Image.open(image_path) as image:
            x = preprocess(image, **config["preprocessing"])[None]
        features = extractor.extract(x)
        probability = predict_probabilities(model, features["cls"])[0]
        result = {"image": str(Path(image_path).resolve()), "prediction": ["NORMAL", "ABNORMAL"][probability.argmax()], "probability_abnormal": float(probability[1]), "cls_first_8": features["cls"][0, :8].tolist(), "patch_shape": list(features["patches"].shape), "use": "experimental research only"}
        print(json.dumps(result, indent=2))
        return result
    manifest_path = Path(manifest_path or output / "features_manifest.json")
    if stage in ("all", "inspect", "extract"):
        # Platform checks precede dataset retrieval, including extract-only runs.
        datasets, dataset_summary = prepare_data(config)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "dataset_summary.json", dataset_summary)
        write_json(output / "config.resolved.json", config)
        if stage == "inspect":
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
        shared = []
        def factory():
            if not shared:
                shared.append(DINOv2Extractor(config["model"], config["preprocessing"]["image_size"]))
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
        write_json(manifest_path, {"identity": identity, "indices": indices, "dataset": dataset_summary})
        write_json(output / "extraction_benchmark.json", stats)
        shared.clear()  # Release backbone before classifier training.
        if stage == "extract":
            return stats
    caches, manifest = load_caches(manifest_path, identity)
    from .training import load_classifier, predict_probabilities, train_classifier
    write_json(output / ("evaluation_config.resolved.json" if stage == "evaluate" else "config.resolved.json"), config)
    if stage in ("all", "train"):
        model, training_stats = train_classifier(caches["train"], caches["val"], config["training"], output, config["runtime"]["seed"])
    else:
        path = checkpoint_path or output / "checkpoints/best_classifier.safetensors"
        model = load_classifier(path, identity)
        checkpoint_metadata = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        training_stats = {"checkpoint": str(path), "training": "not run during evaluation", "checkpoint_training_config": checkpoint_metadata["training"], "checkpoint_seed": checkpoint_metadata["seed"]}
    if stage == "train":
        write_json(output / "experiment_metadata.json", {"system": info, "config": config, "feature_manifest": str(manifest_path.resolve()), "feature_identity": identity, "dataset": manifest["dataset"], "training": training_stats, "test_evaluated": False})
        print(json.dumps(training_stats, indent=2))
        return training_stats
    test_x, test_y = caches["test"].load_cls()
    probabilities = predict_probabilities(model, test_x, config["training"]["batch_size"])
    metrics = save_evaluation(output / "metrics", test_y, probabilities, caches["test"].index["records"])
    sample_id = caches["test"].index["records"][0]["sample_id"]
    sample = caches["test"].load_sample(sample_id)
    example = {"sample_id": sample_id, "cls_shape": list(sample["cls"].shape), "cls_first_8": sample["cls"][:8].tolist(), "patch_shape": list(sample["patches"].shape) if "patches" in sample else None}
    write_json(output / "feature_example.json", example)
    import mlx.core as mx
    memory = {"mlx_peak_bytes": mx.get_peak_memory()}
    try:
        import resource
        memory["process_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    metadata_name = "evaluation_metadata.json" if stage == "evaluate" else "experiment_metadata.json"
    write_json(output / metadata_name, {"system": info, "config": config, "feature_manifest": str(manifest_path.resolve()), "feature_identity": identity, "dataset": manifest["dataset"], "training": training_stats, "memory": memory, "protocol": "Frozen pretrained DINOv2; supervised MLX linear probe; validation-only selection; final test evaluation", "clinical_validation": False})
    print(json.dumps({"metrics": metrics, "output_dir": str(output), "feature_example": example}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Local-first OCT DINOv2 MLX research pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs/smoke_test.yaml"))
    parser.add_argument("--dataset-path", help="Existing local root overrides OCT_DATASET_PATH and YAML")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage", choices=["all", "system", "inspect", "extract", "train", "evaluate", "predict"], default="all")
    parser.add_argument("--feature-manifest", help="Train/evaluate directly from existing caches without dataset/model access")
    parser.add_argument("--checkpoint")
    parser.add_argument("--image")
    parser.add_argument("--offline", action="store_true", help="Disable both dataset and checkpoint network downloads")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.dataset_path, args.output_dir)
        if args.offline:
            config["dataset"]["allow_download"] = config["model"]["allow_download"] = False
        if args.stage == "inspect" and config["dataset"]["max_per_class"] is None:
            # This command is a bounded smoke test, even with the full configuration.
            config["dataset"]["max_per_class"] = 4
        if args.stage == "inspect":
            config["dataset"]["allow_download"] = False
        run(config, args.stage, args.feature_manifest, args.image, args.checkpoint)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
