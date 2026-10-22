from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oct_dinov2_mlx.data import CLASSES, SPLITS


@pytest.fixture
def oct_root(tmp_path):
    root = tmp_path / "download/nested/OCT2017"
    rng = np.random.default_rng(123)
    for split in SPLITS:
        for name in CLASSES:
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in range(6):
                Image.fromarray(rng.integers(0, 256, (32, 48), dtype=np.uint8)).save(directory / f"{name}-{i:03}.png")
            (directory / ".DS_Store").write_text("not an image")
    return root
