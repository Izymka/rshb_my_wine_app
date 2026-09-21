import numpy as np
import pytest
from PIL import Image

from wine_scanner.decide import FEATURE_NAMES, Decider, PairFeatures, derive, matrix
from wine_scanner.decide.features import text_pair_features
from wine_scanner.embed.branches import branch_image


def pair(item_id, score, rank, text_score, text_rank):
    return PairFeatures(
        item_id, score, rank, text_score, text_rank, rank, 10, 8, 0.8, 0.2, 1, 3, 0.9
    )


def test_faiss_scalar_distances_gaps_and_text_top5():
    rows = [pair("a", 0.9, 0, 0.2, 1), pair("b", 0.7, 1, 0.8, 0)]
    values = derive(rows)
    assert values[0]["vis_distance"] == pytest.approx(0.1)
    assert values[1]["vis_distance"] == pytest.approx(0.3)
    assert values[0]["vis_top12_gap"] == pytest.approx(0.2)
    assert values[0]["txt_top1_score"] == 0.8
    assert values[0]["txt_top2_score"] == 0.2
    assert values[0]["txt_top5_score"] == -1
    assert np.asarray(matrix(values)).shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix(values)).all()


def test_empty_ocr_does_not_provide_a_perfect_match():
    missing = text_pair_features("", "", [])
    assert missing["name_jaro_winkler"] == 0
    assert missing["name_levenshtein_distance"] == 1
    exact = text_pair_features("Шато Тамань", "Шато Тамань", ["Шато Тамань"])
    assert exact["name_jaro_winkler"] == 1
    assert exact["name_levenshtein_distance"] == 0


def test_photometric_branch_preserves_geometry_and_color_source():
    array = np.arange(90 * 60 * 3, dtype=np.uint8).reshape(90, 60, 3)
    image = Image.fromarray(array)
    for mode in ("gray", "clahe2", "clahe4"):
        result = np.asarray(branch_image(image, mode))
        assert result.shape == array.shape
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(np.asarray(image), array)


def test_catboost_native_roundtrip_preserves_probabilities(tmp_path):
    from catboost import CatBoostClassifier

    rows = [pair(str(i), i / 10, i, 0.2, i) for i in range(10)]
    values = np.asarray(matrix(derive(rows)))
    booster = CatBoostClassifier(
        iterations=5, depth=2, verbose=False, allow_writing_files=False, thread_count=1
    )
    booster.fit(values, [0] * 5 + [1] * 5)
    decider = Decider(booster, 1.0, 0.0, 0.5)
    decider.save(tmp_path)
    restored = Decider.load(tmp_path)
    expected = decider.score(rows)
    actual = restored.score(rows)
    assert [r.item_id for r in actual] == [r.item_id for r in expected]
    assert np.allclose([r.probability for r in actual], [r.probability for r in expected])
    assert (tmp_path / "model.cbm").exists()


def test_ransac_rejects_outliers_and_reports_coverage():
    from wine_scanner.rerank.xfeat import XFeatMatcher, features_to_dict

    rng = np.random.default_rng(12)
    reference = rng.uniform(10, 180, size=(70, 2)).astype(np.float32)
    query = reference + np.array([8, 5], dtype=np.float32)
    query[50:] = rng.uniform(10, 180, size=(20, 2))
    matcher = object.__new__(XFeatMatcher)
    matcher._correspondences = lambda q, c: (query, reference)
    desc = {"keypoints": reference, "image_size": (220, 220)}
    result = matcher.match(desc, desc)
    assert result.matches == 70
    assert 49 <= result.inliers <= 53
    assert result.homography_ok
    assert 0.2 < result.query_coverage < 0.8
    assert result.normalized_reproj_error < 0.01
    assert "homography" not in features_to_dict(result)


def test_retrained_weights_invalidate_crop_cache(tmp_path):
    from types import SimpleNamespace

    from wine_scanner.detect import CachedCropper

    weights = tmp_path / "checkpoint"
    weights.mkdir()
    file = weights / "model.safetensors"
    file.write_bytes(b"version one")
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (10, 10)).save(source)
    detector = SimpleNamespace(source=str(weights), cache_tag="same-name", margin=0.08)
    first = CachedCropper(detector, tmp_path / "cache")._key(source)
    file.write_bytes(b"version two")
    second = CachedCropper(detector, tmp_path / "cache")._key(source)
    assert first != second


def test_crop_cache_is_pixel_identical_on_first_call_and_hit(tmp_path):
    from types import SimpleNamespace

    from wine_scanner.detect import CachedCropper

    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (37, 41, 3), np.uint8))
    source = tmp_path / "image.png"
    image.save(source)
    detector = SimpleNamespace(cache_tag="fake", crop=lambda im: im.crop((3, 4, 30, 31)))
    cache = CachedCropper(detector, tmp_path / "cache")
    first = cache(source, image)
    second = cache(source, image)
    assert np.array_equal(np.asarray(first), np.asarray(second))
