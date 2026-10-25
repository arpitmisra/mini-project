"""Dense storage/coordinates contract. Test arrays are not DINOv2 outputs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oct_dinov2_mlx.cache import extract_cached
from oct_dinov2_mlx.common import file_digest, write_json
from oct_dinov2_mlx.config import ROOT, load_config
from oct_dinov2_mlx.data import KermanyOCTDataset
from oct_dinov2_mlx.localization import patches_to_grid
from oct_dinov2_mlx.pipeline import identity_for
from oct_dinov2_mlx.preprocessing import preprocess
from oct_dinov2_mlx.representations import export_dense_sample


class PositionTestExtractor:
    metadata = {"test_only": "position-coded arrays, not DINOv2"}

    def extract(self, images):
        # Distinct rows/columns reveal accidental spatial transposes.
        patches = np.arange(256, dtype=np.float32)[:, None] + np.arange(384, dtype=np.float32)[None] / 512
        return {"cls": np.tile(np.arange(384, dtype=np.float32), (len(images), 1)),
                "patches": np.tile(patches[None], (len(images), 1, 1))}


@pytest.fixture
def dense_run(oct_root, tmp_path):
    config = load_config(ROOT / "configs/smoke_test.yaml", str(oct_root))
    identity = identity_for(config)
    data = KermanyOCTDataset(oct_root, "test", preprocess, max_per_class=1)
    path, _ = extract_cached(data, identity, tmp_path / "features", 2, 2, PositionTestExtractor)
    manifest = tmp_path / "features_manifest.json"
    write_json(manifest, {"identity": identity, "indices": {"test": path.relative_to(tmp_path).as_posix()},
                          "dataset": {"source": {"synthetic": True}}})
    return manifest, path, data


def test_default_and_smoke_preserve_dense_features():
    for name in ("default", "smoke_test"):
        assert load_config(ROOT / f"configs/{name}.yaml")["features"]["save_patches"] is True


def test_export_exact_arrays_and_crop_coordinates(dense_run, tmp_path, monkeypatch):
    manifest, _, data = dense_run
    monkeypatch.setattr(KermanyOCTDataset, "__getitem__", lambda *_: pytest.fail("Export read original image"))
    output = tmp_path / "export/scan.npz"
    metadata = export_dense_sample(manifest, output, sample_id=data.records[-1].sample_id)
    assert metadata["sha256"] == file_digest(output)
    assert metadata["dataset_source"]["synthetic"] is True
    assert metadata["sample"]["original_class"] == "DRUSEN"
    assert "not original image pixels" in metadata["coordinates"]["space"]
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["arrays"]["patches"]["dtype"] == "float16"
    with np.load(output, allow_pickle=False) as saved:
        assert saved["cls"].shape == (384,)
        assert saved["patches"].shape == (256, 384)
        assert saved["spatial_grid"].shape == (16, 16, 384)
        assert saved["patches"].dtype == np.float16
        assert int(saved["binary_label"]) == 1
        expected = PositionTestExtractor().extract(np.zeros((1, 1)))["patches"][0].astype(np.float16)
        np.testing.assert_array_equal(saved["patches"], expected)
        for row, col in ((0, 0), (0, 1), (1, 0), (7, 11), (15, 15)):
            token = row * 16 + col
            np.testing.assert_array_equal(saved["spatial_grid"][row, col], expected[token])
            np.testing.assert_array_equal(saved["patch_row_col"][token], [row, col])
            np.testing.assert_array_equal(saved["patch_bounds_xyxy"][token], [14 * col, 14 * row, 14 * (col + 1), 14 * (row + 1)])


def test_export_only_reads_selected_shard_but_checks_its_integrity(dense_run, tmp_path):
    manifest, index_path, data = dense_run
    index = json.loads(index_path.read_text(encoding="utf-8"))
    # An unrelated shard is not needed to export one verified sample.
    (index_path.parent / index["shards"][0]["file"]).write_bytes(b"corrupt unrelated shard")
    export_dense_sample(manifest, tmp_path / "good.npz", sample_id=data.records[-1].sample_id)
    with pytest.raises(ValueError, match="Corrupt"):
        export_dense_sample(manifest, tmp_path / "bad.npz", sample_id=data.records[0].sample_id)
    assert not (tmp_path / "bad.npz").exists()


def test_export_refuses_cls_only_manifest(dense_run, tmp_path):
    manifest, _, _ = dense_run
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content["identity"]["save_patches"] = False
    write_json(manifest, content)
    with pytest.raises(ValueError, match="CLS only"):
        export_dense_sample(manifest, tmp_path / "scan.npz")


def test_export_rejects_unknown_sample_and_overwrite(dense_run, tmp_path):
    manifest, _, _ = dense_run
    output = tmp_path / "scan.npz"
    with pytest.raises(ValueError, match="not present"):
        export_dense_sample(manifest, output, sample_id="test/missing.jpeg")
    with pytest.raises(ValueError, match="end in .npz"):
        export_dense_sample(manifest, tmp_path / "scan.json")
    export_dense_sample(manifest, output)
    with pytest.raises(FileExistsError, match="already exists"):
        export_dense_sample(manifest, output)


def test_export_cli_from_another_working_directory(dense_run, tmp_path):
    manifest, _, _ = dense_run
    output = tmp_path / "cli.npz"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/export_dense.py"),
                             "--feature-manifest", str(manifest), "--output", str(output)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["arrays"]["spatial_grid"]["shape"] == [16, 16, 384]


def test_spatial_grid_rejects_incompatible_shape():
    with pytest.raises(ValueError, match="Expected patches"):
        patches_to_grid(np.zeros((257, 384)))  # CLS must not become a spatial token.
    with pytest.raises(ValueError, match="multiple of 14"):
        patches_to_grid(np.zeros((256, 384)), image_size=225)


def test_simple_show_exports_reuses_and_detects_damage(dense_run, monkeypatch, capsys):
    from oct_dinov2_mlx import cli
    manifest, _, _ = dense_run
    output = cli.show(manifest.parent)
    before = output.stat().st_mtime_ns
    printed = capsys.readouterr().out
    assert "shape=(16, 16, 384)" in printed
    assert "All values are saved" in printed
    monkeypatch.setattr(cli, "export_dense_sample", lambda *_: pytest.fail("Repeated show must reuse export"))
    cli.show(manifest.parent)
    assert output.stat().st_mtime_ns == before
    output.write_bytes(b"damaged export")
    with pytest.raises(ValueError, match="corrupt"):
        cli.show(manifest.parent)


def test_simple_show_selects_another_scan_and_rejects_stale_export(dense_run):
    from oct_dinov2_mlx.cli import show
    manifest, _, data = dense_run
    first = show(manifest.parent)
    selected = show(manifest.parent, sample_id=data.records[-1].sample_id)
    assert selected != first
    with np.load(selected, allow_pickle=False) as saved:
        assert str(saved["sample_id"]) == data.records[-1].sample_id
    sidecar = first.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["feature_identity"] = {"wrong": "identity"}
    write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="does not match"):
        show(manifest.parent)


def test_simple_show_cli_works_from_other_directory_without_images(dense_run, tmp_path):
    manifest, _, data = dense_run
    for record in data.records:
        Path(record.image_path).unlink()
    result = subprocess.run([sys.executable, str(ROOT / "oct.py"), "show", "--output-dir", str(manifest.parent)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "spatial_grid: shape=(16, 16, 384)" in result.stdout
    assert (manifest.parent / "dense_sample.npz").is_file()
