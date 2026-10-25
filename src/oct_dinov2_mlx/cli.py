"""Small front door to the existing pipeline and verified dense exporter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .cache import FeatureCache
from .common import file_digest
from .config import ROOT, load_config
from .pipeline import run
from .representations import export_dense_sample


def show(output_dir, split="test", sample_id=None, all_values=False):
    """Export once, validate on reuse, and print arrays without importing MLX."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "features_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"No completed feature manifest at {manifest_path}. Run 'python oct.py run' "
            "with your dataset first. Use --full for full-run results or --output-dir for a custom run."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = Path(manifest["indices"][split])
    if not index_path.is_absolute():
        index_path = manifest_path.parent / index_path
    cache = FeatureCache(index_path, manifest["identity"], verify=False)
    if cache.index["split"] != split or not cache.index["records"]:
        raise ValueError("Requested split is empty or differs from the feature index")
    records = cache.index["records"]
    selected = sample_id or records[0]["sample_id"]
    record = next((item for item in records if item["sample_id"] == selected), None)
    if record is None:
        raise ValueError(f"Sample {selected!r} is not in {index_path}")
    # Default export is easy to locate. Explicit samples get stable safe filenames.
    filename = "dense_sample.npz" if sample_id is None else (
        "samples/" + hashlib.sha256(f"{split}:{selected}".encode()).hexdigest()[:20] + ".npz"
    )
    if sample_id is None and split != "test":
        filename = f"dense_sample_{split}.npz"
    output_path = output_dir / filename
    sidecar = output_path.with_suffix(".json")
    if output_path.exists() or sidecar.exists():
        if not output_path.is_file() or not sidecar.is_file():
            raise ValueError(f"Incomplete export pair at {output_path}; move that pair aside, then rerun show")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (metadata.get("feature_identity") != manifest["identity"]
                or metadata.get("sample") != record
                or metadata.get("sha256") != file_digest(output_path)):
            raise ValueError(f"Export does not match this run or is corrupt: {output_path}; move the export pair aside, then rerun show")
    else:
        export_dense_sample(manifest_path, output_path, split, selected)
    print(f"Image: {selected}")
    print(f"Original image: {record['image_path']}")
    print(f"Split index (all sample IDs): {index_path.resolve()}")
    print(f"Arrays: {output_path.resolve()}")
    print(f"Metadata: {sidecar.resolve()}")
    with np.load(output_path, allow_pickle=False) as arrays:
        with np.printoptions(threshold=sys.maxsize if all_values else 40, precision=5):
            for key in ("cls", "patches", "spatial_grid"):
                value = arrays[key]
                print(f"\n{key}: shape={value.shape}, dtype={value.dtype}\n{value}")
            grid = arrays["spatial_grid"]
            row, col = min(7, grid.shape[0] - 1), min(11, grid.shape[1] - 1)
            print(f"\nOne patch descriptor, row {row} / column {col}:\n", grid[row, col])
    if not all_values:
        print("\n'...' abbreviates the display only. All values are saved; use --all-values to print them.")
    print("These are learned feature vectors in the preprocessed crop, not lesion masks or class predictions.")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on OCT, then save and view dense features.")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="Small local smoke run; add --full for the full experiment")
    execute.add_argument("dataset", nargs="?", help="Local OCT directory (quote paths with spaces); otherwise use OCT_DATASET_PATH/default")
    execute.add_argument("--offline", action="store_true", help="Prohibit dataset and pretrained-weight downloads")
    display = commands.add_parser("show", help="Export and display one saved scan; no model or original images needed")
    display.add_argument("--split", choices=("train", "val", "test"), default="test")
    display.add_argument("--sample-id", help="Exact sample_id from the split index; defaults to the first scan")
    display.add_argument("--all-values", action="store_true", help="Print every array value (very long output)")
    for command in (execute, display):
        command.add_argument("--full", action="store_true", help="Use full-run settings/results instead of the small smoke run")
        command.add_argument("--output-dir", help="Custom run directory; use the same value with run and show")
    args = parser.parse_args(argv)
    try:
        config = load_config(ROOT / "configs" / ("default.yaml" if args.full else "smoke_test.yaml"),
                             local_path=getattr(args, "dataset", None), output_dir=args.output_dir)
        if args.command == "show":
            show(config["runtime"]["output_dir"], args.split, args.sample_id, args.all_values)
            return 0
        if args.offline:
            config["dataset"]["allow_download"] = config["model"]["allow_download"] = False
        print("FULL experiment: train-only self-supervised adaptation, then dense extraction." if args.full else
              "SMOKE check: at most 48 images, two training updates, then dense extraction. Dataset downloads disabled.")
        print(f"Results: {config['runtime']['output_dir']}")
        result = run(config)
        print(f"\nFinished. Features: {result['feature_manifest']}")
        next_command = "python oct.py show" + (" --full" if args.full else "")
        if args.output_dir:
            next_command += f' --output-dir "{args.output_dir}"'
        print(f"Next: {next_command}")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
