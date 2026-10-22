"""Pinned MLX-VLM DINOv2 adapter. No PyTorch forward or conversion required."""
from __future__ import annotations

import json
from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .common import file_digest, require_mlx

MODEL_REPO = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
MLX_VLM_VERSION = "0.7.6"


def model_identity(config):
    if config["repo_id"] != MODEL_REPO or config["revision"] != MODEL_REVISION:
        raise ValueError("Only the verified pinned facebook/dinov2-small checkpoint is supported. Add architecture/coverage/parity tests before changing it.")
    # Identity is available offline without touching weights: cached head experiments
    # must work even when the model cache or original image directory has been removed.
    return {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "weights_sha256": MODEL_SHA256, "architecture": "dinov2_vits14", "embedding_dim": 384, "patch_size": 14, "register_tokens": 0, "implementation": f"mlx-vlm=={MLX_VLM_VERSION}", "adapter_version": 1, "compute_dtype": "float32"}


def validate_hf_config(config):
    expected = {"model_type": "dinov2", "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "image_size": 518, "patch_size": 14, "num_channels": 3, "mlp_ratio": 4, "layer_norm_eps": 1e-6, "hidden_act": "gelu", "qkv_bias": True, "use_swiglu_ffn": False}
    mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if config.get("num_register_tokens", 0) != 0:
        mismatches["num_register_tokens"] = (config["num_register_tokens"], 0)
    for key, value in (("interpolate_offset", 0.0), ("interpolate_antialias", False)):
        if config.get(key, value) != value:
            mismatches[key] = (config[key], value)
    if mismatches:
        raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")


def expected_hf_shapes():
    shapes = {"embeddings.cls_token": (1, 1, 384), "embeddings.mask_token": (1, 384), "embeddings.position_embeddings": (1, 1370, 384), "embeddings.patch_embeddings.projection.weight": (384, 3, 14, 14), "embeddings.patch_embeddings.projection.bias": (384,), "layernorm.weight": (384,), "layernorm.bias": (384,)}
    for i in range(12):
        prefix = f"encoder.layer.{i}."
        for projection in ("query", "key", "value"):
            shapes[prefix + f"attention.attention.{projection}.weight"] = (384, 384)
            shapes[prefix + f"attention.attention.{projection}.bias"] = (384,)
        for layer, size in (("attention.output.dense", (384, 384)), ("mlp.fc1", (1536, 384)), ("mlp.fc2", (384, 1536))):
            shapes[prefix + layer + ".weight"] = size
            shapes[prefix + layer + ".bias"] = (size[0],)
        for norm in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                shapes[prefix + norm + "." + kind] = (384,)
        shapes[prefix + "layer_scale1.lambda1"] = (384,)
        shapes[prefix + "layer_scale2.lambda1"] = (384,)
    return shapes


def validate_shapes(actual, expected):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    wrong = {k: (tuple(actual[k].shape), shape) for k, shape in expected.items() if k in actual and tuple(actual[k].shape) != tuple(shape)}
    if missing or extra or wrong:
        raise ValueError(f"Incomplete/incompatible checkpoint: missing={sorted(missing)}, unexpected={sorted(extra)}, shapes={wrong}")


def checkpoint_files(config):
    model_identity(config)
    if config.get("local_path"):
        root = Path(config["local_path"])
        paths = {name: root / name for name in ("config.json", "model.safetensors")}
        if any(not p.is_file() for p in paths.values()):
            raise FileNotFoundError("Local model folder requires config.json and model.safetensors in HF format")
    else:
        from huggingface_hub import hf_hub_download
        try:
            paths = {name: Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION, local_files_only=not config["allow_download"])) for name in ("config.json", "model.safetensors")}
        except Exception as exc:
            raise RuntimeError("DINOv2 checkpoint unavailable. Allow the initial Hugging Face download (~89 MB), or set model.local_path to an existing HF-format checkpoint folder.") from exc
    if file_digest(paths["model.safetensors"]) != MODEL_SHA256:
        raise ValueError("DINOv2 weights SHA-256 does not match the published pinned Meta checkpoint. Restore the original model.safetensors; converted or modified weights are not accepted.")
    validate_hf_config(json.loads(paths["config.json"].read_text(encoding="utf-8")))
    return paths


class DINOv2Extractor:
    def __init__(self, config, image_size=224):
        require_mlx()
        try:
            installed_version = version("mlx-vlm")
        except PackageNotFoundError as exc:
            raise RuntimeError(f"MLX-VLM is missing. Install requirements.txt (requires mlx-vlm=={MLX_VLM_VERSION}).") from exc
        if installed_version != MLX_VLM_VERSION:
            raise RuntimeError(f"Install mlx-vlm=={MLX_VLM_VERSION}; cache identity requires the audited implementation.")
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.models.dinov2.config import ModelConfig
        from mlx_vlm.models.dinov2.dinov2 import Model
        paths = checkpoint_files(config)
        hf_config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
        validate_hf_config(hf_config)
        allowed = {f.name for f in fields(ModelConfig)}
        self.model = Model(ModelConfig(**{k: v for k, v in hf_config.items() if k in allowed}))
        weights = mx.load(str(paths["model.safetensors"]))
        validate_shapes(weights, expected_hf_shapes())
        sanitized = self.model.sanitize(weights)
        expected = {k: tuple(v.shape) for k, v in tree_flatten(self.model.parameters())}
        validate_shapes(sanitized, expected)
        self.model.load_weights(list(sanitized.items()), strict=True)
        self.model.freeze()
        self.model.eval()
        mx.eval(self.model.parameters())
        self.image_size = image_size
        self.metadata = {**model_identity(config), "tensor_count_source": len(weights), "tensor_count_mlx": len(sanitized)}

    def extract(self, images):
        import mlx.core as mx
        if images.ndim != 4 or tuple(images.shape[1:]) != (self.image_size, self.image_size, 3):
            raise ValueError(f"Expected NHWC [B,{self.image_size},{self.image_size},3], received {images.shape}")
        result = self.model.forward_features(mx.array(images, dtype=mx.float32))
        cls, patches = result["x_norm_clstoken"], result["x_norm_patchtokens"]
        mx.eval(cls, patches)  # Materialize before timing, writing or freeing the batch.
        cls, patches = np.asarray(cls), np.asarray(patches)
        count = (self.image_size // 14) ** 2
        if cls.shape != (len(images), 384) or patches.shape != (len(images), count, 384):
            raise ValueError(f"Invalid DINOv2 output shapes: {cls.shape}, {patches.shape}")
        if not np.isfinite(cls).all() or not np.isfinite(patches).all() or np.std(cls) == 0 or np.std(patches) == 0:
            raise ValueError("Nonfinite or trivial pretrained representations")
        return {"cls": cls, "patches": patches, "embedding_dimension": 384, "patch_grid": [self.image_size // 14] * 2, "patch_size": 14, "model_name": "dinov2_vits14"}
