"""Spatial feature interface for future validated localization methods."""
from typing import Protocol

import numpy as np


class Localizer(Protocol):
    def explain(self, normalized_image: np.ndarray, class_index: int) -> dict:
        """Return method, raw patch-grid scores and preprocessing geometry."""
        ...


def spatial_grid(cache, sample_id):
    sample = cache.load_sample(sample_id)
    if "patches" not in sample:
        raise ValueError("This cache contains CLS only; extract with features.save_patches=true")
    side = cache.index["identity"]["preprocessing"]["image_size"] // 14
    return sample["patches"].reshape(side, side, 384)
