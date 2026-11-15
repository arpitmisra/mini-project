"""Single-image attention/similarity inspection; no dataset access or training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import digest, file_digest, write_json
from .preprocessing import preprocess, preprocessing_identity


def similarities(patches):
    """Cosine and Pearson across descriptor channels, not statistical causality."""
    def gram(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        unit = x / np.maximum(norm, 1e-12)
        return np.clip(unit @ unit.T, -1, 1).astype(np.float32), norm[:, 0] > 1e-12
    cosine, valid_cosine = gram(patches)
    pearson, valid_pearson = gram(patches - patches.mean(axis=-1, keepdims=True))
    return dict(cosine=cosine, pearson=pearson, valid_cosine=valid_cosine, valid_pearson=valid_pearson)


def forward_inspection(extractor, images, layer, backend):
    if backend == "torch":
        import torch
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(extractor.device)
        with torch.inference_mode():
            result = extractor.model(pixel_values=tensor, output_attentions=True)
        return {"cls": result.last_hidden_state[:, 0].cpu().numpy(),
                "patches": result.last_hidden_state[:, 1:].cpu().numpy(),
                "attention": result.attentions[layer].cpu().numpy()}
    import mlx.core as mx
    model = extractor.model
    x = model.prepare_tokens(mx.array(images))
    attention = None
    for i, block in enumerate(model.blocks):
        if i == layer:
            z = block.norm1(x)
            b, n, c = z.shape
            attn = block.attn
            qkv = attn.qkv(z).reshape(b, n, 3, attn.num_heads, attn.head_dim).transpose(2, 0, 3, 1, 4)
            attention = mx.softmax((qkv[0] @ qkv[1].swapaxes(-1, -2)) * attn.scale, axis=-1)
        x = block(x)
    tokens = model.norm(x)
    mx.eval(tokens, attention)
    return {"cls": np.asarray(tokens[:, 0]), "patches": np.asarray(tokens[:, 1:]), "attention": np.asarray(attention)}


def validate_attention(attention):
    if attention.ndim != 3 or attention.shape[-2:] != (257, 257) or attention.shape[0] != 6:
        raise ValueError(f"Expected 6 x 257 x 257 attention, received {attention.shape}")
    if not np.isfinite(attention).all() or np.min(attention) < -1e-7:
        raise ValueError("Invalid attention weights")
    np.testing.assert_allclose(attention.sum(-1), 1, atol=2e-5)


def inspect_image(config, image_path=None, backend="torch", device="cpu", layer=11, checkpoint=None, output_dir=None):
    if not 0 <= layer < 12:
        raise ValueError("layer must be 0 through 11")
    if config["preprocessing"]["image_size"] != 224:
        raise ValueError("The dashboard currently supports 224px / 16x16 patches")
    if checkpoint and backend != "mlx":
        raise ValueError("Adapted checkpoints are native MLX; use --backend mlx")
    if backend == "mlx":
        import os
        os.environ["OCT_MLX_DEVICE"] = device
    if image_path:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        sample = {"path": str(Path(image_path).resolve()), "sha256": file_digest(image_path), "kind": "user image"}
    else:
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (280, 350, 3), dtype=np.uint8))
        sample = {"kind": "synthetic random noise", "seed": 42, "sha256": digest(image.tobytes().hex())}
    from .backbone import model_identity
    identity = model_identity(config["model"])
    if backend == "torch":
        from .pretrained import identity_for
        identity = identity_for(config, device)["model"]
    if checkpoint:
        from .adaptation import read_adapted_checkpoint, adapted_identity
        meta = read_adapted_checkpoint(checkpoint, config["model"])
        identity = adapted_identity(checkpoint, config["model"])
        training_normalization = meta["contract"]["preprocessing"].get("normalization", "imagenet")
    else:
        training_normalization = "imagenet"
    contract = {"version": 1, "sample": sample, "model": identity, "device": device,
                "layer": layer, "backend": backend, "preprocessing": preprocessing_identity(config["preprocessing"])}
    output = Path(output_dir) if output_dir else Path("outputs/inspection") / digest(contract)[:16]
    output.mkdir(parents=True, exist_ok=True)
    report_path, arrays_path = output / "report.json", output / "arrays.npz"
    from .dashboard import render_dashboard
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["contract"] != json.loads(json.dumps(contract)) or report["arrays_sha256"] != file_digest(arrays_path):
            raise ValueError("Inspection output has different settings or corrupt arrays; choose a new --output-dir")
        with np.load(arrays_path, allow_pickle=False) as saved:
            arrays = dict(saved)
        render_dashboard(output / "dashboard.html", arrays, report)
        print(f"Reused inspection (no inference): {(output / 'dashboard.html').resolve()}")
        return output
    if backend == "mlx":
        import os
        os.environ["OCT_MLX_DEVICE"] = device
        from .backbone import DINOv2Extractor
        extractor = DINOv2Extractor(config["model"], checkpoint=checkpoint)
    else:
        from .pretrained import PretrainedExtractor
        extractor = PretrainedExtractor(config["model"], device=device)
    settings = dict(config["preprocessing"])
    settings["normalization"] = "unit"
    unit = preprocess(image, **settings)
    arrays = {"crop_rgb": np.rint(unit * 255).astype(np.uint8)}
    report = {"contract": contract, "training_normalization": training_normalization,
              "scope": "Single-image input-normalization ablation; not a segmentation accuracy benchmark", "modes": {}}
    # Warm up outside reported timings. Only forward calls are measured.
    extractor.extract(unit[None])
    for mode in ("imagenet", "unit"):
        settings["normalization"] = mode
        x = preprocess(image, **settings)[None]
        start = time.perf_counter()
        result = forward_inspection(extractor, x, layer, backend)
        elapsed = time.perf_counter() - start
        repeated = extractor.extract(x)
        for key in ("cls", "patches"):
            np.testing.assert_allclose(result[key], repeated[key], atol=5e-4, rtol=1e-3)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite features")
        attention = result["attention"][0]
        validate_attention(attention)
        values = {k: v[0] for k, v in result.items()}
        values.update(similarities(values["patches"]))
        arrays.update({mode + "_" + k: v for k, v in values.items()})
        report["modes"][mode] = {"input_min": float(x.min()), "input_max": float(x.max()),
            "forward_seconds_single_observation": elapsed,
            "patch_norm_mean": float(np.linalg.norm(values["patches"], axis=-1).mean()),
            "attention_entropy_mean": float(-(attention * np.log(np.maximum(attention, 1e-12))).sum(-1).mean()),
            "repeat_max_abs_error": float(np.max(np.abs(result["patches"] - repeated["patches"]))),
            "finite": True, "attention_rows_sum_to_one": True}
    a, b = arrays["imagenet_patches"], arrays["unit_patches"]
    report["comparison"] = {"patch_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "same_location_cosine_mean": float(np.mean(np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12))),
        "cosine_matrix_mean_absolute_change": float(np.abs(arrays["imagenet_cosine"] - arrays["unit_cosine"]).mean())}
    np.savez_compressed(arrays_path, **arrays)
    with np.load(arrays_path, allow_pickle=False) as saved:
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], saved[key])
    report["arrays_sha256"] = file_digest(arrays_path)
    report["checks_passed"] = True
    write_json(report_path, report)
    render_dashboard(output / "dashboard.html", arrays, report)
    print(f"PASS: shapes, finite outputs, attention row sums, repeated inference and saved-array reload.\nDashboard: {(output / 'dashboard.html').resolve()}")
    return output
