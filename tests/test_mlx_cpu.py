"""Portable device-policy and bounded MLX entry-point regressions."""
from pathlib import Path

import pytest

from oct_dinov2_mlx import common, mlx_cpu_cli
from oct_dinov2_mlx.backbone import model_identity
from oct_dinov2_mlx.config import ROOT, load_config


def test_device_policy_defaults_and_override(monkeypatch):
    monkeypatch.delenv("OCT_MLX_DEVICE", raising=False)
    monkeypatch.setattr(common.platform, "system", lambda: "Windows")
    monkeypatch.setattr(common.platform, "machine", lambda: "AMD64")
    assert common.mlx_device_name() == "cpu"
    monkeypatch.setattr(common.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(common.platform, "machine", lambda: "arm64")
    assert common.mlx_device_name() == "gpu"
    monkeypatch.setenv("OCT_MLX_DEVICE", "cpu")
    assert common.mlx_device_name() == "cpu"
    monkeypatch.setenv("OCT_MLX_DEVICE", "nonsense")
    with pytest.raises(ValueError, match="must be"):
        common.mlx_device_name()


def test_mlx_device_changes_cache_identity(monkeypatch):
    config = load_config(ROOT / "configs/smoke_test.yaml")
    monkeypatch.setenv("OCT_MLX_DEVICE", "cpu")
    cpu = model_identity(config["model"])
    monkeypatch.setenv("OCT_MLX_DEVICE", "gpu")
    gpu = model_identity(config["model"])
    assert cpu != gpu
    assert cpu["execution_device"] == "cpu"
    assert cpu["weights_sha256"] == gpu["weights_sha256"]


def test_mlx_command_is_bounded_offline_and_uses_mlx(tmp_path, monkeypatch):
    monkeypatch.setenv("OCT_MLX_DEVICE", "cpu")
    seen = []
    monkeypatch.setattr(mlx_cpu_cli, "run", lambda config, **kwargs: seen.append((config, kwargs)))
    assert mlx_cpu_cli.main(["run", str(tmp_path / "OCT images"), "--offline", "--output-dir", str(tmp_path / "out")]) == 0
    config, kwargs = seen[0]
    assert config["dataset"]["max_per_class"] == 1
    assert not config["dataset"]["allow_download"]
    assert not config["model"]["allow_download"]
    assert kwargs == {"device": "cpu", "backend": "mlx"}
    assert Path(config["runtime"]["output_dir"]) == tmp_path / "out"


def test_mlx_show_uses_separate_results(monkeypatch):
    monkeypatch.setenv("OCT_MLX_DEVICE", "cpu")
    seen = []
    monkeypatch.setattr(mlx_cpu_cli, "show", lambda *args: seen.append(args))
    assert mlx_cpu_cli.main(["show"]) == 0
    assert Path(seen[0][0]) == ROOT / "outputs/mlx-cpu-smoke"
