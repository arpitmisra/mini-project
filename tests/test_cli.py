"""Convenience commands must preserve bounded defaults and the real pipeline."""
from pathlib import Path

import pytest

from oct_dinov2_mlx import cli
from oct_dinov2_mlx.config import ROOT


@pytest.mark.parametrize("full", [False, True])
def test_run_routes_to_correct_config_and_same_pipeline(full, tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "OCT data with spaces"
    output = tmp_path / "my results"
    seen = []
    monkeypatch.setattr(cli, "run", lambda config: seen.append(config) or {"feature_manifest": "mock-manifest.json"})
    args = ["run", str(dataset), "--output-dir", str(output)] + (["--full"] if full else [])
    assert cli.main(args) == 0
    config = seen[0]
    assert Path(config["dataset"]["local_path"]) == dataset
    assert Path(config["runtime"]["output_dir"]) == output
    assert config["dataset"]["allow_download"] is full
    assert config["dataset"]["max_per_class"] == (None if full else 4)
    assert config["finetuning"]["max_steps_per_epoch"] == (None if full else 2)
    assert config["finetuning"]["epochs"] == (5 if full else 1)
    assert config["features"]["save_patches"] is True
    assert "Next: python oct.py show" in capsys.readouterr().out


def test_offline_and_environment_path_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("OCT_DATASET_PATH", str(tmp_path))
    seen = []
    monkeypatch.setattr(cli, "run", lambda config: seen.append(config) or {"feature_manifest": "mock"})
    assert cli.main(["run", "--full", "--offline"]) == 0
    assert Path(seen[0]["dataset"]["local_path"]) == tmp_path
    assert not seen[0]["dataset"]["allow_download"]
    assert not seen[0]["model"]["allow_download"]


@pytest.mark.parametrize("full,directory", [(False, "smoke-adapted"), (True, "oct-adapted")])
def test_show_routes_to_matching_run(full, directory, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "show", lambda *args: seen.append(args))
    assert cli.main(["show"] + (["--full"] if full else [])) == 0
    assert Path(seen[0][0]) == ROOT / "outputs" / directory
    assert seen[0][1:] == ("test", None, False)


def test_missing_results_are_actionable_without_starting_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda *_: pytest.fail("show must not run model"))
    assert cli.main(["show", "--output-dir", str(tmp_path)]) == 1
    assert "No completed feature manifest" in capsys.readouterr().err


def test_failure_does_not_print_finished(monkeypatch, capsys):
    def fail(*_):
        raise RuntimeError("test native failure")
    monkeypatch.setattr(cli, "run", fail)
    assert cli.main(["run"]) == 1
    printed = capsys.readouterr()
    assert "Finished" not in printed.out
    assert "test native failure" in printed.err
