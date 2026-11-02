from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def system_info() -> dict:
    packages = {}
    for name in ("torch", "safetensors", "mlx", "mlx-vlm", "numpy", "Pillow", "kagglehub", "scikit-learn", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "packages": packages}
    if platform.system() == "Darwin":
        import subprocess
        try:
            info["hardware"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            info["hardware"] = "unavailable"
        info["macos"] = platform.mac_ver()[0]
    return info


def mlx_device_name():
    """Explicit policy, available without importing MLX (including offline identities)."""
    selected = os.environ.get("OCT_MLX_DEVICE")
    if selected is None:
        selected = "gpu" if platform.system() == "Darwin" and platform.machine().lower() == "arm64" else "cpu"
    if selected not in ("cpu", "gpu"):
        raise ValueError("OCT_MLX_DEVICE must be cpu or gpu")
    return selected


def require_mlx() -> None:
    selected = mlx_device_name()
    try:
        import mlx.core as mx
        import mlx.nn  # noqa: F401
        import mlx.optimizers  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError("MLX runtime unavailable. For CPU install requirements-mlx-cpu.txt in a separate environment; Windows needs both mlx and mlx-cpu.") from exc
    if selected == "gpu" and not mx.metal.is_available():
        raise RuntimeError("MLX Metal is unavailable. Set OCT_MLX_DEVICE=cpu to use the installed CPU backend.")
    mx.set_default_device(mx.cpu if selected == "cpu" else mx.gpu)
