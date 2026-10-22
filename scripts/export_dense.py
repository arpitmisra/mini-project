"""Export one scan's CLS, patch matrix, spatial grid and crop coordinates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oct_dinov2_mlx.representations import export_dense_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-id", help="Exact sample_id from the split index; defaults to its first record")
    parser.add_argument("--output", required=True, help="New .npz path; matching .json metadata is written alongside")
    args = parser.parse_args()
    try:
        result = export_dense_sample(args.feature_manifest, args.output, args.split, args.sample_id)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": result["output"], "sample_id": result["sample"]["sample_id"], "arrays": result["arrays"], "coordinates": result["coordinates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
