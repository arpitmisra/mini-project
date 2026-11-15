import json

import numpy as np
import pytest
from PIL import Image

from oct_dinov2_mlx.inspection import similarities, validate_attention
from oct_dinov2_mlx.preprocessing import MEAN, STD, preprocess, preprocessing_identity
from oct_dinov2_mlx.augmentations import paired_views


def test_normalization_changes_only_scale_and_has_separate_identity():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (280, 350, 3), dtype=np.uint8))
    unit = preprocess(image, normalization="unit")
    standard = preprocess(image)
    np.testing.assert_allclose(standard, (unit - np.array(MEAN, np.float32)) / np.array(STD, np.float32))
    config = dict(image_size=224, resize_shortest=256)
    assert preprocessing_identity(config) == preprocessing_identity({**config, "normalization": "imagenet"})
    assert preprocessing_identity(config) != preprocessing_identity({**config, "normalization": "unit"})
    for normalization in ("unit", "imagenet"):
        t, s = paired_views(image, {**config, "normalization": normalization}, np.random.default_rng(1), strength=0)
        np.testing.assert_array_equal(t, s)
        if normalization == "unit":
            assert t.min() >= 0 and t.max() <= 1


def test_relationships_agree_with_independent_calculation_and_handle_constants():
    features = np.array([[1, 2, 4], [3, 2, -1], [5, 5, 5], [0, 0, 0]], np.float32)
    result = similarities(features)
    np.testing.assert_allclose(result["pearson"][:2, :2], np.corrcoef(features[:2]), atol=1e-6)
    expected = np.dot(features[0], features[1]) / (np.linalg.norm(features[0]) * np.linalg.norm(features[1]))
    np.testing.assert_allclose(result["cosine"][0, 1], expected, atol=1e-7)
    assert result["valid_pearson"].tolist() == [True, True, False, False]
    assert result["valid_cosine"].tolist() == [True, True, True, False]
    for key in ("cosine", "pearson"):
        np.testing.assert_array_equal(result[key], result[key].T)


def test_attention_checks_include_cls_mass():
    a = np.full((6, 257, 257), 1 / 257, np.float32)
    validate_attention(a)
    assert np.all(a[:, 1:, 1:].sum(-1) < 1)
    with pytest.raises(AssertionError):
        validate_attention(a * .5)


def test_dashboard_escapes_untrusted_metadata(tmp_path):
    from oct_dinov2_mlx.dashboard import render_dashboard
    arrays = {"crop_rgb": np.zeros((224, 224, 3), np.uint8)}
    for mode in ("unit", "imagenet"):
        for key in ("attention", "cosine", "pearson", "valid_cosine", "valid_pearson"):
            arrays[mode + "_" + key] = np.ones((1,), np.float32)
    path = tmp_path / "viewer.html"
    render_dashboard(path, arrays, {"sample": "</script><script>alert(1)</script>"})
    html = path.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    payload = html.split('<script id="payload" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)["report"]["sample"].startswith("</script>")
