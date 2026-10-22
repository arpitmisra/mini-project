"""Explicit optional PyTorch reference verification; never used by the pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from oct_dinov2_mlx.backbone import DINOv2Extractor, checkpoint_files
from oct_dinov2_mlx.common import require_mlx, system_info, write_json
from oct_dinov2_mlx.config import ROOT, load_config
from oct_dinov2_mlx.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional real OCT image; otherwise deterministic artificial pixels")
    parser.add_argument("--config", default=str(ROOT / "configs/smoke_test.yaml"))
    parser.add_argument("--output", default=str(ROOT / "outputs/parity.json"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    require_mlx()
    config = load_config(args.config)
    if args.offline:
        config["model"]["allow_download"] = False
    import torch
    from safetensors.torch import load_file
    from transformers import Dinov2Config, Dinov2Model
    paths = checkpoint_files(config["model"])
    hf_config = Dinov2Config.from_json_file(str(paths["config.json"]))
    # Eager reference avoids implementation-dependent fused attention numerics.
    hf_config._attn_implementation = "eager"
    reference = Dinov2Model(hf_config).float().cpu().eval()
    reference.load_state_dict(load_file(str(paths["model.safetensors"])), strict=True)
    if args.image:
        with Image.open(args.image) as image:
            pixels = preprocess(image, **config["preprocessing"])
    else:
        pixels = preprocess(Image.fromarray(np.random.default_rng(42).integers(0, 256, (280, 350, 3), dtype=np.uint8)), **config["preprocessing"])
    images = np.stack([pixels, pixels[:, ::-1].copy()])
    extractor = DINOv2Extractor(config["model"], config["preprocessing"]["image_size"])
    mlx_output = extractor.extract(images)
    with torch.no_grad():
        output = reference(torch.from_numpy(images.transpose(0, 3, 1, 2).copy())).last_hidden_state.numpy()
    expected = {"cls": output[:, 0], "patches": output[:, 1:]}
    results, passed = {}, True
    for key in ("cls", "patches"):
        difference = np.abs(mlx_output[key] - expected[key])
        matches = np.allclose(mlx_output[key], expected[key], atol=5e-4, rtol=1e-3)
        passed &= bool(matches)
        results[key] = {"shape": list(expected[key].shape), "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()), "rms_error": float(np.sqrt(np.mean(difference ** 2))), "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(expected[key]), 1e-12)), "passed": bool(matches)}
    write_json(Path(args.output), {"passed": passed, "tolerances": {"atol": 5e-4, "rtol": 1e-3}, "results": results, "model": extractor.metadata, "system": system_info(), "image": args.image or "deterministic artificial pixels"})
    print(json.dumps(results, indent=2))
    if not passed:
        raise SystemExit("Parity FAILED. Inspect preprocessing, interpolation and dependency versions before trusting research results.")


if __name__ == "__main__":
    main()
