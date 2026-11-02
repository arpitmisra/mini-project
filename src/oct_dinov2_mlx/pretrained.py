"""Portable frozen DINOv2 baseline using the maintained Transformers model."""
from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .backbone import checkpoint_files, model_identity
from .cache import extract_cached, feature_identity
from .common import system_info, write_json
from .pipeline import prepare_data
from .preprocessing import preprocessing_identity


def check_device(device="cpu"):
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install the baseline first: bash oct.sh setup (Mac), or python -m pip install -r requirements.txt") from exc
    if torch.__version__.split("+")[0] != "2.9.1" or transformers.__version__ != "4.57.6":
        raise RuntimeError("Baseline requires torch==2.9.1 and transformers==4.57.6. Rerun setup in the dedicated environment.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use --device cpu; native Apple Silicon Python/macOS is needed for MPS.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu.")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    return torch


def identity_for(config, device="cpu", backend="torch"):
    identity = model_identity(config["model"])
    if backend == "torch":
        # Preserve existing PyTorch identities independently of optional MLX installation.
        identity.pop("mlx_version", None)
        identity.pop("execution_device", None)
        identity.update(implementation="transformers==4.57.6", torch_version=version("torch"),
                        adapter_version=1, device=device, training="none: frozen pretrained baseline")
    elif backend != "mlx":
        raise ValueError("backend must be torch or mlx")
    return feature_identity(identity, preprocessing_identity(config["preprocessing"]),
                            True, config["features"]["patch_dtype"])


class PretrainedExtractor:
    def __init__(self, config, image_size=224, device="cpu"):
        torch = check_device(device)
        from safetensors.torch import load_file
        from transformers import Dinov2Config, Dinov2Model
        paths = checkpoint_files(config)
        hf = Dinov2Config.from_json_file(str(paths["config.json"]))
        hf._attn_implementation = "eager"
        # No remote code, no random-weight fallback, strict complete loading.
        self.model = Dinov2Model(hf)
        self.model.load_state_dict(load_file(str(paths["model.safetensors"])), strict=True)
        self.model.requires_grad_(False).eval().float().to(device)
        if device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.device, self.image_size = device, image_size
        self.metadata = {**model_identity(config), "implementation": "transformers==4.57.6",
                         "torch_version": torch.__version__, "device": device,
                         "training": "none: frozen pretrained baseline", "adapter_version": 1}
        self.metadata.pop("mlx_version", None)
        self.metadata.pop("execution_device", None)

    def extract(self, images):
        import torch
        if images.ndim != 4 or tuple(images.shape[1:]) != (self.image_size, self.image_size, 3):
            raise ValueError(f"Expected NHWC images [B,{self.image_size},{self.image_size},3]")
        x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float().to(self.device)
        with torch.inference_mode():
            tokens = self.model(pixel_values=x).last_hidden_state.cpu().numpy()
        count = (self.image_size // 14) ** 2
        if tokens.shape != (len(images), count + 1, 384):
            raise ValueError(f"Unexpected DINOv2 output: {tokens.shape}")
        if not np.isfinite(tokens).all() or np.std(tokens) == 0:
            raise ValueError("Invalid/trivial pretrained features")
        return {"cls": tokens[:, 0].copy(), "patches": tokens[:, 1:].copy()}


def run(config, device="cpu", backend="torch"):
    if backend == "mlx":
        from .common import require_mlx, mlx_device_name
        require_mlx()
        if mlx_device_name() != device:
            raise ValueError("MLX device policy differs from the requested extraction device")
    elif backend == "torch":
        check_device(device)
    else:
        raise ValueError("backend must be torch or mlx")
    datasets, summary = prepare_data(config)
    identity = identity_for(config, device, backend)
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "features_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["identity"] != json.loads(json.dumps(identity)):
            raise ValueError("This run folder contains a different model/device/preprocessing identity. Choose a new --output-dir.")
    shared = []
    def factory():
        if not shared:
            print(f"Loading verified pretrained DINOv2 with {backend} on {device}...", flush=True)
            if backend == "mlx":
                from .backbone import DINOv2Extractor
                shared.append(DINOv2Extractor(config["model"], config["preprocessing"]["image_size"]))
            else:
                shared.append(PretrainedExtractor(config["model"], config["preprocessing"]["image_size"], device))
        return shared[0]
    indices, stats = {}, {}
    for split, dataset in datasets.items():
        index, stat = extract_cached(dataset, identity, config["features"]["cache_dir"],
                                    config["features"]["batch_size"], config["features"]["shard_size"], factory)
        try:
            indices[split] = Path(os.path.relpath(index.resolve(), output.resolve())).as_posix()
        except ValueError:
            indices[split] = str(index.resolve())
        stats[split] = stat
        print(f"{split}: extracted {stat['extracted']}, reused {stat['reused']}", flush=True)
    write_json(manifest_path, {"identity": identity, "indices": indices, "dataset": summary,
                               "protocol": "Frozen pretrained DINOv2; no training and no segmentation"})
    write_json(output / "config.resolved.json", config)
    write_json(output / "dataset_summary.json", summary)
    write_json(output / "extraction_benchmark.json", stats)
    write_json(output / "experiment_metadata.json", {"system": system_info(), "identity": identity,
               "protocol": "Frozen pretrained baseline", "config": config, "dataset": summary})
    return {"feature_manifest": str(manifest_path), "extraction": stats}


def self_test(config, device="cpu"):
    """Two artificial inputs, real weights, batch agreement and saved-array roundtrip."""
    from PIL import Image
    from .preprocessing import preprocess
    extractor = PretrainedExtractor(config["model"], config["preprocessing"]["image_size"], device)
    pixels = np.random.default_rng(42).integers(0, 256, (280, 350, 3), dtype=np.uint8)
    x = preprocess(Image.fromarray(pixels), **config["preprocessing"])
    batch = extractor.extract(np.stack([x, x[:, ::-1].copy()]))
    single = extractor.extract(x[None])
    errors = {}
    for name in ("cls", "patches"):
        np.testing.assert_allclose(batch[name][:1], single[name], atol=5e-4, rtol=1e-3)
        errors[name] = float(np.max(np.abs(batch[name][:1] - single[name])))
    if np.allclose(batch["patches"][0], batch["patches"][1]):
        raise ValueError("Distinct test images produced identical features")
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "artificial_features.npz", **batch)
    with np.load(output / "artificial_features.npz", allow_pickle=False) as saved:
        for key in batch:
            np.testing.assert_array_equal(saved[key], batch[key])
    report = {"passed": True, "images": 2, "synthetic_inputs": True, "real_pretrained_weights": True,
              "device": device, "batch_vs_single_max_error": errors, "system": system_info(),
              "model": extractor.metadata, "shapes": {k: list(v.shape) for k, v in batch.items()}}
    write_json(output / "self_test.json", report)
    print(f"PASS: real DINOv2 inference, batch consistency and saved-array reload. Report: {output / 'self_test.json'}")
    return report
