import csv
import sys

import pytest
from PIL import Image

from scripts import import_live_photos as imp


def _folder(root, name):
    folder = root / "incoming" / name
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(folder / "a.jpg")
    return folder


@pytest.fixture
def roots(tmp_path, monkeypatch):
    test, train = tmp_path / "test", tmp_path / "train"
    monkeypatch.setattr(imp, "TEST_ROOT", test)
    monkeypatch.setattr(imp, "ROOTS", {"test": test, "train": train})
    return test, train


def test_known_wine_goes_to_train_with_source_group(tmp_path, roots):
    _, train = roots
    folder = _folder(tmp_path, "some-slug")
    root, rows = imp.import_folder(
        folder, {"some-slug"}, "train", False, known_dest="train", source="web"
    )
    assert root == train
    assert rows[0]["true_slug"] == "some-slug"
    assert rows[0]["group"] == "web" and rows[0]["source"] == "web"


def test_known_wine_defaults_to_test_live(tmp_path, roots):
    test, _ = roots
    folder = _folder(tmp_path, "some-slug")
    root, rows = imp.import_folder(folder, {"some-slug"}, "train", False)
    assert root == test
    assert rows[0]["group"] == "live" and rows[0]["source"] == "live"


def test_known_dest_train_refuses_wine_from_test(tmp_path, roots, monkeypatch):
    test, _ = roots
    test.mkdir()
    with (test / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=imp.FIELDS)
        writer.writeheader()
        writer.writerow({"path": "x.jpg", "wine_id": "held", "true_slug": "held"})
    _folder(tmp_path, "held")
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("slug\nheld\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["imp", "--incoming", str(tmp_path / "incoming"), "--catalog", str(catalog),
         "--known-dest", "train", "--keep"],
    )
    with pytest.raises(SystemExit, match="held"):
        imp.main()
