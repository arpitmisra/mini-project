"""Real checkpoint integration, opt-in so routine tests stay offline."""
import os
from pathlib import Path

import numpy as np
import pytest

from oct_dinov2_mlx.config import ROOT, load_config


@pytest.mark.baseline
@pytest.mark.skipif(os.environ.get("OCT_TEST_BASELINE") != "1", reason="Opt in for ~89 MB real pretrained weights, no dataset download")
def test_real_baseline_and_cached_export(oct_root, tmp_path, monkeypatch):
    from oct_dinov2_mlx import pretrained
    from oct_dinov2_mlx.cli import show
    config = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root), str(tmp_path / "run"))
    config["dataset"]["max_per_class"] = 1
    config["features"]["cache_dir"] = str(tmp_path / "features")
    first = pretrained.run(config)
    assert first["extraction"]["test"]["extracted"] == 4
    exported = show(tmp_path / "run")
    with np.load(exported, allow_pickle=False) as data:
        assert data["patches"].shape == (256, 384)
        assert np.isfinite(data["patches"]).all()
        assert np.std(data["patches"].astype(np.float32)) > 0
        np.testing.assert_array_equal(data["spatial_grid"][7, 11], data["patches"][123])
    assert not (tmp_path / "run/finetuning").exists()
    monkeypatch.setattr(pretrained, "PretrainedExtractor", lambda *a, **k: pytest.fail("Reuse loaded model"))
    config["model"]["allow_download"] = False
    second = pretrained.run(config)
    assert all(s["extracted"] == 0 and s["reused"] == 4 for s in second["extraction"].values())
    assert show(tmp_path / "run") == exported


@pytest.mark.baseline
@pytest.mark.skipif(os.environ.get("OCT_TEST_BASELINE") != "1", reason="Opt in for real model inference")
def test_real_batch_consistency_and_roundtrip(tmp_path):
    from oct_dinov2_mlx.pretrained import self_test
    config = load_config(ROOT / "configs/smoke_test.yaml", output_dir=str(tmp_path))
    result = self_test(config)
    assert result["passed"]
    assert result["shapes"]["patches"] == [2, 256, 384]
    assert (tmp_path / "self_test.json").is_file()
