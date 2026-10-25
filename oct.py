"""Simple run/show commands for the OCT adaptation pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oct_dinov2_mlx.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
