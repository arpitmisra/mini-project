"""Bounded inspection and self-supervised adaptation commands."""
import argparse
import os
from pathlib import Path

from .config import ROOT, load_config


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Any image, attention dashboard and normalization comparison")
    inspect.add_argument("image", nargs="?", help="Image file; omitted = deterministic random noise")
    inspect.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    inspect.add_argument("--device", choices=("cpu", "mps", "cuda", "gpu"), default="cpu")
    inspect.add_argument("--layer", type=int, choices=range(12), default=11)
    inspect.add_argument("--checkpoint", help="Merged adapted MLX backbone.safetensors; use --backend mlx")
    tune = commands.add_parser("finetune", help="MLX LoRA or full teacher/student adaptation, then dense export")
    tune.add_argument("dataset", nargs="?", help="Local dataset; no dataset download unless --full")
    tune.add_argument("--mode", choices=("lora", "full"), default="lora")
    tune.add_argument("--normalization", choices=("imagenet", "unit"), default="imagenet")
    tune.add_argument("--full", action="store_true", help="Use full dataset and full experiment settings")
    tune.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    tune.add_argument("--rank", type=int, default=8)
    for command in (inspect, tune):
        command.add_argument("--offline", action="store_true")
        command.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            if args.backend == "mlx" and args.device not in ("cpu", "gpu"):
                raise ValueError("MLX device must be cpu or gpu")
            if args.backend == "torch" and args.device == "gpu":
                raise ValueError("PyTorch device must be cpu, mps or cuda")
            config = load_config(ROOT / "configs/smoke_test.yaml")
            config["model"]["allow_download"] = not args.offline
            from .inspection import inspect_image
            inspect_image(config, args.image, args.backend, args.device, args.layer, args.checkpoint, args.output_dir)
        else:
            if not 1 <= args.rank <= 384:
                raise ValueError("rank must be between 1 and 384")
            os.environ["OCT_MLX_DEVICE"] = args.device
            directory = args.output_dir or str(ROOT / "outputs" / f"{args.mode}-{args.normalization}-{'full' if args.full else 'smoke'}")
            config = load_config(ROOT / "configs" / ("default.yaml" if args.full else "smoke_test.yaml"), args.dataset, directory)
            config["finetuning"].update(mode=args.mode, lora_rank=args.rank, lora_alpha=float(args.rank))
            config["preprocessing"]["normalization"] = args.normalization
            if not args.full:
                config["dataset"].update(max_per_class=1, allow_download=False)
                config["finetuning"].update(epochs=1, max_steps_per_epoch=1)
            if args.offline:
                config["dataset"]["allow_download"] = config["model"]["allow_download"] = False
            print(f"{args.mode}: self-supervised TRAIN-only adaptation. " + ("Full experiment." if args.full else "One-step smoke check; at most 12 images indexed. Not an efficacy benchmark."), flush=True)
            from .pipeline import run
            run(config)
        return 0
    except Exception as exc:
        import traceback
        log = ROOT / "outputs/research-error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR: {exc}\nDetails: {log}")
        return 1
