"""Аналоги из других виноделен: цвет обязателен, своя винодельня исключена, стиль учитывается."""

from wine_scanner.analogues import Analogues

PAYLOADS = {
    "port-a": {
        "name": "Портвейн красный Алушта",
        "winery": "Массандра",
        "category": "Красное",
        "region": "Крым",
        "grapes": "Красные сорта винограда",
        "description": "креплёное сладкое вино",
    },
    "port-b": {
        "name": "Портвейн красное креплёное",
        "winery": "Коктебель",
        "category": "Красное",
        "region": "Крым",
        "grapes": "Каберне Совиньон",
        "description": "креплёное сладкое вино",
    },
    "dry-red": {
        "name": "Каберне Совиньон сухое",
        "winery": "Фанагория",
        "category": "Красное",
        "region": "Кубань",
        "grapes": "Каберне Совиньон",
        "description": "сухое красное",
    },
    "white": {
        "name": "Шардоне",
        "winery": "Коктебель",
        "category": "Белое",
        "region": "Крым",
        "grapes": "Шардоне",
        "description": "креплёное сладкое вино",
    },
    "sibling": {
        "name": "Портвейн белый Алушта",
        "winery": "Массандра",
        "category": "Красное",
        "region": "Крым",
        "grapes": "",
        "description": "креплёное сладкое вино",
    },
}


def test_analogues_prefer_same_style_other_winery():
    analogues = Analogues(list(PAYLOADS), list(PAYLOADS.values()))
    result = analogues.for_item("port-a", k=5)
    slugs = [r["slug"] for r in result]
    assert slugs[0] == "port-b"  # креплёное красное из Крыма другой винодельни
    assert "white" not in slugs  # другой цвет — не аналог
    assert "sibling" not in slugs  # своя винодельня исключена
    assert "dry-red" in slugs and slugs.index("dry-red") > slugs.index("port-b")


def test_unknown_item_gives_nothing():
    analogues = Analogues(list(PAYLOADS), list(PAYLOADS.values()))
    assert analogues.for_item("nope") == []
