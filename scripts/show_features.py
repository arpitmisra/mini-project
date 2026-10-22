"""Inspect one real cached representation without importing MLX."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oct_dinov2_mlx.cache import FeatureCache
from oct_dinov2_mlx.localization import spatial_grid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("index", help="Path to a split index-*.json")
parser.add_argument("--sample-id")
args = parser.parse_args()
cache = FeatureCache(args.index)
sample_id = args.sample_id or cache.index["records"][0]["sample_id"]
sample = cache.load_sample(sample_id)
print("Sample:", sample_id)
print("CLS shape:", sample["cls"].shape, "First 8 values:", sample["cls"][:8])
if "patches" in sample:
    print("Patch matrix:", sample["patches"].shape, "Spatial grid:", spatial_grid(cache, sample_id).shape)
else:
    print("Patch persistence was disabled for this cache.")
