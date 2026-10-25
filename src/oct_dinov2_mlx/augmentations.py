"""Aligned OCT teacher/student views; no diagnosis labels or vertical flips."""
import numpy as np
from PIL import ImageEnhance, ImageFilter

from .preprocessing import normalize_rgb, resize_rgb


def paired_views(image, preprocessing, rng, strength=0.2):
    """Two crop geometries, each shared by its teacher and student view.

    Patch distillation is within the matching geometry. Global distillation
    crosses the two crops. Only photometric transforms differ within a pair.
    """
    base = resize_rgb(image, preprocessing["resize_shortest"])
    size = preprocessing["image_size"]
    teacher, student = [], []
    for _ in range(2):
        left = int(rng.integers(0, base.width - size + 1))
        top = int(rng.integers(0, base.height - size + 1))
        crop = base.crop((left, top, left + size, top + size))
        # Do not flip retinal depth/anatomy. Keep spatial correspondence exact.
        for target, amount in ((teacher, strength / 4), (student, strength)):
            view = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(1 - amount, 1 + amount)))
            view = ImageEnhance.Contrast(view).enhance(float(rng.uniform(1 - amount, 1 + amount)))
            if target is student and amount > 0 and rng.random() < 0.5:
                view = view.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.1, 1.0))))
            target.append(normalize_rgb(view))
    return np.stack(teacher), np.stack(student)


def patch_masks(batch_size, patch_count, ratio, rng):
    count = max(1, min(patch_count - 1, int(round(patch_count * ratio))))
    if patch_count < 2 or not 0 < ratio < 1:
        raise ValueError("Masking requires at least two patches and 0 < mask_ratio < 1")
    masks = np.zeros((batch_size, patch_count), dtype=bool)
    for row in masks:
        row[rng.choice(patch_count, count, replace=False)] = True
    return masks
