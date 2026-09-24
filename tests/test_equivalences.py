"""Дубли каталога: таблица решений человека и подстановка канонического slug в ответ."""

import pytest

from wine_scanner.catalog import load_equivalences
from wine_scanner.pipeline import Candidate, canonicalize

HEADER = "slug_a,slug_b,decision,canonical_slug,reviewer_note\n"


def write(tmp_path, rows: str):
    path = tmp_path / "eq.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_only_equivalent_pairs_count(tmp_path):
    eq = load_equivalences(write(tmp_path, (
        "a,b,equivalent,,дубль\n"
        "c,d,equivalent,d,у c страница 404\n"
        "e,f,duplicate_photo_not_equivalent,,разные вина\n"
    )))
    assert eq.same("a", "b") and eq.same("b", "a") and eq.same("a", "a")
    assert not eq.same("e", "f")
    assert eq.canonical == {"c": "d"}


def test_missing_table_means_strict(tmp_path):
    eq = load_equivalences(tmp_path / "нет.csv")
    assert not eq.same("a", "b") and eq.canonical == {}


def test_canonical_must_be_in_pair(tmp_path):
    with pytest.raises(ValueError):
        load_equivalences(write(tmp_path, "a,b,equivalent,z,\n"))


def cand(item_id, p):
    return Candidate(item_id, p, {"name": item_id}, {})


def test_canonicalize_promotes_existing_canonical_and_drops_duplicate():
    out = canonicalize([cand("c", 0.7), cand("x", 0.2), cand("d", 0.1)], {"c": "d"}, {})
    assert [c.item_id for c in out] == ["d", "x"]
    assert out[0].probability == pytest.approx(0.7)


def test_canonicalize_renames_when_canonical_not_among_candidates():
    out = canonicalize([cand("c", 0.7), cand("x", 0.2)], {"c": "d"}, {"d": {"name": "Д"}})
    assert [c.item_id for c in out] == ["d", "x"] and out[0].payload == {"name": "Д"}
