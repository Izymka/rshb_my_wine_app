"""Проверка сведения разных алфавитов к общей форме.

Собственный тестовый набор состоит из вин с латинскими этикетками, поэтому кириллическую
ветку на нём измерить нельзя. Пока не появился каталог платформы, единственный способ
убедиться, что она работает, — проверить её на примерах здесь.
"""

import pytest
from rapidfuzz import fuzz

from wine_scanner.ocr import normalize, tokens, variants


def similarity(a: str, b: str) -> float:
    return max(fuzz.token_set_ratio(v, normalize(b)) for v in variants(a)) / 100


def test_accents_are_stripped():
    assert normalize("Château Tamagne") == "chateau tamagne"
    assert normalize("Rosé Brut") == "rose brut"


def test_cyrillic_transliterated():
    assert normalize("Фанагория") == "fanagoriya"
    assert normalize("Абрау-Дюрсо") == "abrau dyurso"


def test_same_wine_across_alphabets_matches():
    """Кириллица на этикетке против латиницы в каталоге и наоборот."""
    assert similarity("ФАНАГОРИЯ", "Fanagoria") > 0.85
    assert similarity("Fanagoria Cabernet", "Фанагория Каберне") > 0.75


def test_homoglyph_misreading_recovered():
    """OCR прочитал кириллическое слово латинскими двойниками."""
    # «МОСКВА» латинскими буквами, как это часто выдаёт распознавание
    assert similarity("MOCKBA", "Москва") > 0.8


def test_different_wines_stay_apart():
    assert similarity("Фанагория", "Абрау-Дюрсо") < 0.5
    assert similarity("Mouton Cadet", "Vang Dalat") < 0.5


def test_tokens_drop_single_characters():
    assert tokens("A Château 12 % vol") == ["chateau", "12", "vol"]


# --- Сворачивание fold(): фонетика и псевдонимы ------------------------------------------

from wine_scanner.ocr import fold, fold_tokens  # noqa: E402


@pytest.mark.parametrize(
    ("label", "catalog"),
    [
        ("Château Tamagne", "Шато Тамань"),
        ("Fanagoria", "Фанагория"),
        ("Abrau-Durso", "Абрау-Дюрсо"),
        ("Myskhako", "Мысхако"),
        ("Cabernet Sauvignon", "Каберне Совиньон"),
        ("Pinot Noir", "Пино Нуар"),
        ("Chardonnay", "Шардоне"),
        ("Massandra", "Массандра"),
        ("Rouge", "Красное"),
        ("Rosé Brut", "Розовое брют"),
        ("Semi-Sweet White", "Полусладкое белое"),
        ("Golubitskoe Estate", "Поместье Голубицкое"),
        ("Portwein", "Портвейн"),
        ("Zweigelt", "Цвайгельт"),
    ],
)
def test_fold_joins_label_and_catalog_spellings(label, catalog):
    assert fold(label) == fold(catalog)


def test_fold_keeps_different_wineries_apart():
    assert fold("Массандра") != fold("Мысхако")
    # «Новый Свет» не должен превращаться в сладкое: sweet и svet после схлопывания совпадают,
    # поэтому «sweet» в псевдонимах нет намеренно.
    assert "sladkoe" not in fold("Новый Свет")


def test_fold_tokens_drop_single_letters():
    assert fold_tokens("A Chateau 12 % vol") == ["shato", "12", "vol"]
