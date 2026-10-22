"""Source-checkout entry point; also works before an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from oct_dinov2_mlx.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
