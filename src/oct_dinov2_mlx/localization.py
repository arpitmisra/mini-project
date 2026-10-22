"""Spatial feature interface for future validated localization methods."""
from typing import Protocol

import numpy as np


class Localizer(Protocol):
    def explain(self, normalized_image: np.ndarray, class_index: int) -> dict:
        """Return method, raw patch-grid scores and preprocessing geometry."""
        ...


def patches_to_grid(patches, image_size=224):
    """Row-major tokens -> [row, column, channel] on the preprocessed crop."""
    if type(image_size) is not int or image_size < 14 or image_size % 14:
        raise ValueError("image_size must be a positive multiple of 14")
    side = image_size // 14
    patches = np.asarray(patches)
    if patches.shape != (side * side, 384):
        raise ValueError(f"Expected patches [{side * side},384], received {patches.shape}")
    return patches.reshape(side, side, 384)


def spatial_grid(cache, sample_id):
    sample = cache.load_sample(sample_id)
    if "patches" not in sample:
        raise ValueError("This cache contains CLS only; extract with features.save_patches=true")
    return patches_to_grid(sample["patches"], cache.index["identity"]["preprocessing"]["image_size"])
