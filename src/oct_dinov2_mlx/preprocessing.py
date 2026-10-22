from __future__ import annotations

import numpy as np
from PIL import Image, __version__ as pillow_version

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocessing_identity(config):
    return {**config, "algorithm": "rgb-bicubic-shortest-center-crop-v1", "mean": MEAN, "std": STD, "layout": "NHWC", "pillow_version": pillow_version}


def preprocess(image: Image.Image, image_size=224, resize_shortest=256) -> np.ndarray:
    """HF checkpoint preprocessing: RGB, shortest-side bicubic, center crop, normalize."""
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 1 or resize_shortest < image_size:
        raise ValueError("Invalid image or preprocessing dimensions")
    if w <= h:
        target = (resize_shortest, int(h * resize_shortest / w))
    else:
        target = (int(w * resize_shortest / h), resize_shortest)
    image = image.resize(target, Image.Resampling.BICUBIC)
    left, top = (target[0] - image_size) // 2, (target[1] - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))
    x = np.asarray(image, dtype=np.float32) / np.float32(255)
    return np.ascontiguousarray((x - np.array(MEAN, np.float32)) / np.array(STD, np.float32))
