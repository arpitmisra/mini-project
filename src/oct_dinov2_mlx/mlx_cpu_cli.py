"""Exercise the existing MLX DINOv2 adapter on CPU; no PyTorch model execution."""
import argparse
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

from .backbone import DINOv2Extractor
from .cli import show
from .common import require_mlx, system_info, write_json
from .config import ROOT, load_config
from .preprocessing import preprocess
from .pretrained import run


def self_test(config):
    require_mlx()
    import mlx.core as mx
    x = mx.array([[1., 2.], [3., 4.]])
    product = x @ x
    gradient = mx.grad(lambda value: mx.sum(value * value))(x)
    mx.eval(product, gradient)
    np.testing.assert_allclose(np.asarray(product), [[7, 10], [15, 22]])
    np.testing.assert_allclose(np.asarray(gradient), [[2, 4], [6, 8]])
    extractor = DINOv2Extractor(config["model"], config["preprocessing"]["image_size"])
    pixels = np.random.default_rng(42).integers(0, 256, (280, 350, 3), dtype=np.uint8)
    image = preprocess(Image.fromarray(pixels), **config["preprocessing"])
    batch = extractor.extract(np.stack([image, image[:, ::-1].copy()]))
    single = extractor.extract(image[None])
    arrays, errors = {}, {}
    for key in ("cls", "patches"):
        arrays[key] = batch[key]
        np.testing.assert_allclose(batch[key][:1], single[key], atol=5e-4, rtol=1e-3)
        errors[key] = float(np.max(np.abs(batch[key][:1] - single[key])))
    if np.allclose(arrays["patches"][0], arrays["patches"][1]):
        raise ValueError("Distinct test images produced identical features")
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "artificial_features.npz", **arrays)
    with np.load(output / "artificial_features.npz", allow_pickle=False) as saved:
        for key, values in arrays.items():
            np.testing.assert_array_equal(saved[key], values)
    result = {"passed": True, "framework": "MLX", "device": str(mx.default_device()),
              "system": system_info(), "model": extractor.metadata,
              "synthetic_inputs": True, "real_pretrained_weights": True, "images": 2,
              "core_matmul_and_gradient_passed": True, "batch_vs_single_max_error": errors,
              "shapes": {key: list(value.shape) for key, value in arrays.items()}}
    write_json(output / "self_test.json", result)
    print(f"PASS: MLX CPU arithmetic, gradients, real DINOv2, batch consistency and saved arrays. {output / 'self_test.json'}")
    return result


def main(argv=None):
    os.environ["OCT_MLX_DEVICE"] = "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "run", "show"))
    parser.add_argument("dataset", nargs="?", help="Existing local OCT dataset directory, for run")
    parser.add_argument("--full", action="store_true", help="All images; permits Kaggle fallback if the local dataset is unavailable")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--sample-id")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--all-values", action="store_true")
    args = parser.parse_args(argv)
    directory = "mlx-cpu-self-test" if args.command == "test" else ("mlx-cpu" if args.full else "mlx-cpu-smoke")
    try:
        config = load_config(ROOT / "configs" / ("default.yaml" if args.full else "smoke_test.yaml"),
                             args.dataset, args.output_dir or str(ROOT / "outputs" / directory))
        if args.offline:
            config["dataset"]["allow_download"] = config["model"]["allow_download"] = False
        if args.command == "test":
            self_test(config)
        elif args.command == "show":
            show(config["runtime"]["output_dir"], args.split, args.sample_id, args.all_values)
        else:
            if not args.full:
                config["dataset"]["max_per_class"] = 1
                config["dataset"]["allow_download"] = False
            print("MLX CPU: frozen pretrained extraction; no training. " + ("Full dataset." if args.full else "At most 12 scans; no dataset download."), flush=True)
            run(config, device="cpu", backend="mlx")
            print("Finished. View with the same environment: python mlx_cpu.py show" + (" --full" if args.full else "") + (f' --output-dir "{args.output_dir}"' if args.output_dir else ""))
        return 0
    except Exception as exc:
        log = ROOT / "outputs/mlx-cpu-last-error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR: {exc}\nDiagnostic: {log}", file=sys.stderr)
        return 1
