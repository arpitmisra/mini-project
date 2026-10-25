from __future__ import annotations

import os
import math
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, local_path=None, output_dir=None) -> dict:
    base = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("The YAML configuration must contain a mapping of sections")
    cfg = deepcopy(base)
    for section, values in supplied.items():
        if section not in base or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        if set(values) - set(base[section]):
            raise ValueError(f"Unknown configuration keys in {section}: {set(values) - set(base[section])}")
        cfg[section].update(values)
    cfg["dataset"]["local_path"] = local_path or os.environ.get("OCT_DATASET_PATH") or cfg["dataset"]["local_path"]
    if output_dir:
        cfg["runtime"]["output_dir"] = output_dir
    for section, key in (("dataset", "local_path"), ("model", "local_path"), ("features", "cache_dir"), ("runtime", "output_dir")):
        if cfg[section][key] is not None and not isinstance(cfg[section][key], str):
            raise ValueError(f"{section}.{key} must be a path string or null")
        if section in ("features", "runtime") and not cfg[section][key]:
            raise ValueError(f"{section}.{key} must be a nonempty path")
        if cfg[section][key]:
            p = Path(os.path.expandvars(str(cfg[section][key]))).expanduser()
            cfg[section][key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    for section, key in (("dataset", "allow_download"), ("dataset", "hash_images"), ("model", "allow_download"), ("features", "save_patches")):
        if not isinstance(cfg[section][key], bool):
            raise ValueError(f"{section}.{key} must be a YAML boolean (true/false, without quotes)")
    for section, key in (("features", "batch_size"), ("features", "shard_size"), ("finetuning", "epochs"), ("finetuning", "batch_size"), ("finetuning", "head_hidden_dim"), ("finetuning", "head_bottleneck_dim"), ("finetuning", "prototypes")):
        if type(cfg[section][key]) is not int or cfg[section][key] < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")
    limit = cfg["dataset"]["max_per_class"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("dataset.max_per_class must be null or positive")
    fraction = cfg["dataset"]["validation_fraction"]
    if fraction is not None and (type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")
    size = cfg["preprocessing"]["image_size"]
    shortest = cfg["preprocessing"]["resize_shortest"]
    if type(size) is not int or type(shortest) is not int or size < 14 or size % 14 or shortest < size:
        raise ValueError("image_size must be a multiple of 14; resize_shortest must be at least image_size")
    if cfg["features"]["patch_dtype"] not in ("float16", "float32"):
        raise ValueError("patch_dtype must be float16 or float32")
    if not cfg["features"]["save_patches"]:
        raise ValueError("Dense-representation mode requires features.save_patches=true")
    if size < 28:
        raise ValueError("Masked patch adaptation requires image_size >= 28")
    for key in ("learning_rate", "weight_decay", "student_temperature", "teacher_temperature", "global_loss_weight", "patch_loss_weight", "gradient_clip"):
        value = cfg["finetuning"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"finetuning.{key} must be finite and {'nonnegative' if key == 'weight_decay' else 'positive'}")
    for key in ("teacher_momentum", "center_momentum", "mask_ratio"):
        value = cfg["finetuning"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"finetuning.{key} must be between 0 and 1")
    strength = cfg["finetuning"]["augmentation_strength"]
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 0.5:
        raise ValueError("augmentation_strength must be between 0 and 0.5")
    steps = cfg["finetuning"]["max_steps_per_epoch"]
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("max_steps_per_epoch must be null or a positive integer")
    if cfg["finetuning"]["prototypes"] < 2:
        raise ValueError("Self-supervised projection requires at least two prototypes")
    if type(cfg["runtime"]["seed"]) is not int or not 0 <= cfg["runtime"]["seed"] < 2**32:
        raise ValueError("runtime.seed must be an integer in [0, 2**32)")
    version = cfg["dataset"]["version"]
    if version is not None and (type(version) is not int or version < 1):
        raise ValueError("dataset.version must be null or a positive integer")
    return cfg
