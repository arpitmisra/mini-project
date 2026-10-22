"""Create tiny artificial images for Mac plumbing tests when no OCT data is present.

These are NOT OCT scans and results must never be treated as research metrics.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from oct_dinov2_mlx.common import write_json
from oct_dinov2_mlx.config import ROOT
from oct_dinov2_mlx.data import CLASSES, SPLITS


def create_dataset(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    rng = np.random.default_rng(42)
    for split in SPLITS:
        for label, name in enumerate(CLASSES):
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in range(4):
                image = rng.integers(0, 60, (64, 96), dtype=np.uint8)
                image[12 + label * 8:16 + label * 8] = 220
                Image.fromarray(image).save(directory / f"ARTIFICIAL-{name}-{i}.png")
    write_json(root / "SYNTHETIC_DATASET.json", {"synthetic": True, "purpose": "plumbing smoke test only; not retinal OCT data", "seed": 42})
    print(f"Created 48 artificial test images at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs/synthetic-smoke-data"))
    create_dataset(parser.parse_args().output)
