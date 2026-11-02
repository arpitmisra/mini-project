"""Explicit MLX CPU test/run/show commands, including native Windows."""
import os
from pathlib import Path
import sys

os.environ["OCT_MLX_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oct_dinov2_mlx.mlx_cpu_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
