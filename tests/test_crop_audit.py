from pathlib import Path

from scripts.audit_rtdetr_crops import artifact_relative
from wine_scanner.catalog import is_holdout
from wine_scanner.detect.audit import annotations, iou, transfer_box


def test_transfer_clips_and_measures_lost_area():
    assert transfer_box([20, 30, 60, 70], [10, 10, 100, 100]) == ([10, 20, 50, 60], 1)
    assert transfer_box([0, 0, 20, 20], [10, 0, 30, 20]) == ([0, 0, 10, 20], 0.5)
    assert transfer_box([0, 0, 20, 20], [20, 0, 30, 20]) == (None, 0)
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1


def test_artifact_export_name_is_short_deterministic_and_unicode_safe(tmp_path):
    root = tmp_path / "data"
    source = root / "a" / ("этикетка_" + "оченьдлинная" * 50 + ".jpg")
    first = artifact_relative(source, root)
    second = artifact_relative(source, root)
    assert first == second
    assert first.parent == Path("a")
    assert first.suffix == ".jpg"
    assert len(first.name) == 24


def test_annotation_classes_are_not_merged(tmp_path):
    (tmp_path / "_annotations.csv").write_text(
        "bottle,0,0,20,30,a.jpg,40,60\nlabel,5,10,10,10,a.jpg,40,60\n", encoding="utf-8"
    )
    row = next(iter(annotations(tmp_path).values()))
    assert row["bottle"] == [[0, 0, 20, 30]]
    assert row["label"] == [[5, 10, 15, 20]]


def test_holdout_absolute_and_derived_paths():
    for name in (
        "data/test/a.jpg",
        "data/live/a.jpg",
        "data/derived/rtdetr_audit/bottles/test/frames_annotated",
    ):
        assert is_holdout(name)
        assert is_holdout(Path(name).resolve())
    assert not is_holdout("data/derived/rtdetr_audit/bottles/train/frames_annotated")
