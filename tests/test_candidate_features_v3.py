import numpy as np
import pytest
from PIL import Image

from wine_scanner.decide import FEATURE_NAMES, Decider, PairFeatures, derive, matrix
from wine_scanner.decide.features import text_pair_features
from wine_scanner.embed.branches import branch_image


def pair(item_id, score, rank, text_score, text_rank):
    return PairFeatures(
        item_id,
        score,
        rank,
        text_score,
        text_rank,
        rank,
        10,
        8,
        0.8,
        0.2,
        1,
        3,
        0.9,
        geometry_query_coverage=0.5,
        geometry_candidate_coverage=0.5,
        geometry_normalized_error=0.1,
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
    assert values[0]["rrf_margin"] == 1
    assert values[0]["geometry_support"] > 0
    assert values[0]["geometry_margin"] == pytest.approx(0.0)
    assert np.asarray(matrix(values)).shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix(values)).all()


def test_empty_ocr_does_not_provide_a_perfect_match():
    missing = text_pair_features("", "", [])
    assert missing["name_jaro_winkler"] == 0
    assert missing["name_levenshtein_distance"] == 1
    exact = text_pair_features("Шато Тамань", "Шато Тамань", ["Шато Тамань"])
    assert exact["name_jaro_winkler"] == 1
    assert exact["name_levenshtein_distance"] == 0
    fields = text_pair_features(
        "Фанагория Каберне Совиньон",
        "Каберне Совиньон",
        ["Фанагория"],
        winery="Фанагория",
        grapes="Каберне Совиньон",
    )
    assert fields["winery_jaro_winkler"] > 0.7
    assert fields["grapes_token_set_ratio"] == 1


def test_family_margins_compare_only_siblings():
    rows = [
        PairFeatures("a", 0.9, 0, 0.8, 0, 0, 20, 16, 0.8, 0.2, 1, 3, 0.9, family="A"),
        PairFeatures("b", 0.8, 1, 0.6, 1, 1, 10, 8, 0.8, 0.2, 1, 3, 0.9, family="A"),
        PairFeatures("c", 0.95, 2, 0.1, 2, 2, 30, 20, 0.7, 0.2, 1, 3, 0.9, family="B"),
    ]
    values = {row["item_id"]: row for row in derive(rows)}
    assert values["a"]["family_vis_margin"] == pytest.approx(0.1)
    assert values["a"]["family_txt_margin"] == pytest.approx(0.2)
    assert values["c"]["family_size"] == 1
    assert values["c"]["family_vis_margin"] == 0


def test_reading_margins_against_strongest_sibling():
    def twin(item_id, family, **text):
        return PairFeatures(item_id, 0.9, 0, 0.5, 0, 0, 10, 8, 0.8, 0.2, 1, 3, 0.9,
                            family=family, **text)

    rows = [
        # Этикетка читается как «белое полусладкое Рислинг» — подтверждает «a», а не «b».
        twin("a", "A", disc_hit=0.8, color_match=1, style_match=1, grape_match=1),
        twin("b", "A", disc_hit=0.2, color_match=1, style_match=-1, grape_match=-1),
        twin("c", "B", disc_hit=0.9, color_match=-1),
    ]
    values = {row["item_id"]: row for row in derive(rows)}
    assert values["a"]["attr_agree"] == 3
    assert values["b"]["attr_agree"] == -1
    assert values["a"]["family_disc_margin"] == pytest.approx(0.6)
    assert values["a"]["family_style_margin"] == 2
    assert values["a"]["family_grape_margin"] == 2
    assert values["a"]["family_color_margin"] == 0
    assert values["a"]["family_attr_margin"] == 4
    assert values["b"]["family_attr_margin"] == -4
    # Без соседок по винодельне отрыв нулевой, чужая «c» с сильным disc_hit не в счёт.
    assert values["c"]["family_disc_margin"] == 0
    assert values["c"]["family_attr_margin"] == 0


def test_ranks_are_recomputed_among_remaining_candidates():
    # Верный кандидат «a» выброшен (аугментация незнакомого или исключённая карточка):
    # лучший из оставшихся по картинке должен стать нулевым, а не сохранить дыру.
    rows = [pair("b", 0.8, 1, 0.5, 2), pair("c", 0.6, 3, 0.9, 0), pair("d", 0.5, 999, 0.1, 999)]
    values = {row["item_id"]: row for row in derive(rows)}
    assert [values[i]["vis_rank"] for i in "bcd"] == [0, 1, 999]
    assert [values[i]["txt_rank"] for i in "bcd"] == [1, 0, 999]
    assert [values[i]["rrf_rank"] for i in "bcd"] == [0, 1, 2]
    assert values["b"]["rrf_margin"] == 1
    assert values["c"]["txt_in_top5"] == 1
    # Без дыр производные совпадают с сырыми рангами.
    full = derive([pair("a", 0.9, 0, 0.9, 0), *rows])
    assert [row["vis_rank"] for row in full] == [0, 1, 2, 999]


def test_previous_version_model_still_loads():
    from wine_scanner.decide.features import FEATURE_VERSION

    names = FEATURE_NAMES[: FEATURE_NAMES.index("grapes_token_set_ratio") + 1]
    decider = Decider(None, 1.0, 0.0, 0.5, feature_names=names, meta={"feature_version": 4})
    assert decider.meta["feature_version"] == 4
    assert FEATURE_VERSION == 5
    with pytest.raises(ValueError):
        Decider(None, 1.0, 0.0, 0.5, feature_names=names, meta={"feature_version": 2})


def test_ranker_changes_order_but_classifier_keeps_confidence():
    class Confidence:
        def predict_proba(self, values):
            return np.column_stack((np.zeros(len(values)), np.asarray([0.9, 0.2])))

    class Ranker:
        def predict(self, values):
            return np.asarray([0.1, 0.8])

    rows = [pair("a", 0.9, 0, 0.2, 1), pair("b", 0.7, 1, 0.8, 0)]
    scored = Decider(Confidence(), 1.0, 0.0, 0.5, ranker=Ranker()).score(rows)
    assert [row.item_id for row in scored] == ["b", "a"]
    assert scored[0].probability == pytest.approx(0.2)


def test_legacy_feature_prefix_stays_usable_after_new_features_are_appended():
    class LegacyConfidence:
        def predict_proba(self, values):
            assert values.shape[1] == 3
            return np.column_stack((np.zeros(len(values)), np.full(len(values), 0.8)))

    decider = Decider(
        LegacyConfidence(),
        1.0,
        0.0,
        0.5,
        feature_names=FEATURE_NAMES[:3],
        meta={"feature_version": 3},
    )
    assert decider.score([pair("a", 0.9, 0, 0.8, 0)])[0].item_id == "a"


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


def test_ranker_native_roundtrip_preserves_order(tmp_path):
    from catboost import CatBoostClassifier, CatBoostRanker

    rows = [pair(str(i), i / 10, i, 0.2, i) for i in range(10)]
    values = np.asarray(matrix(derive(rows)))
    labels = [0] * 5 + [1] * 5
    confidence = CatBoostClassifier(
        iterations=5, depth=2, verbose=False, allow_writing_files=False, thread_count=1
    ).fit(values, labels)
    ranker = CatBoostRanker(
        iterations=5,
        depth=2,
        loss_function="YetiRank",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    ).fit(values, labels, group_id=[0] * len(rows))
    importance = ranker.get_feature_importance(type="PredictionValuesChange")
    assert len(importance) == len(FEATURE_NAMES)
    decider = Decider(confidence, 1.0, 0.0, 0.5, ranker=ranker)
    decider.save(tmp_path)
    restored = Decider.load(tmp_path)
    assert restored.ranker is not None
    assert [row.item_id for row in restored.score(rows)] == [
        row.item_id for row in decider.score(rows)
    ]
    assert (tmp_path / "ranker.cbm").exists()


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
