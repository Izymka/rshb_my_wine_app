"""Карточка вина со страницы сайта: разбор __NUXT_DATA__ (devalue) и sitemap."""

import json

from wine_scanner.site import load_sitemap, nuxt_payload, site_wine

# Срез устройства настоящей страницы (gevyurcztraminer-oranzh, 17.10.2026): объекты хранят
# номера элементов массива, обёртки реактивности — ["ShallowReactive", i].
DATA = [
    ["ShallowReactive", 1],
    {"data": 2, "state": 28},
    ["ShallowReactive", 3],
    {"wine-gevyurcztraminer-oranzh": 4},
    {"wine": 5, "similarWines": 27},
    {
        "alcohol": 6,
        "category": 7,
        "color": 9,
        "description": 10,
        "grapes": 11,
        "image": 14,
        "manufacturer": 17,
        "region": 19,
        "slug": 21,
        "temperature": 22,
        "title": 23,
    },
    12,
    {"name": 8},
    "Оранжевое сухое",
    "Золотистый с розовым отливом",
    "Брожение на мезге.",
    [12],
    {"name": 13},
    "Гевюрцтраминер",
    {"altText": -1, "url": 16},
    None,
    "/uploads/Gevyurcztraminer_Oranzh_f3d914f332.webp",
    {"name": 18},
    "Усадьба Перовских",
    {"name": 20},
    "Крым",
    "gevyurcztraminer-oranzh",
    "10-12",
    "Гевюрцтраминер Оранж",
    None,
    None,
    None,
    [],
    {},
]


def page(data) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f'<html><script type="application/json" id="__NUXT_DATA__">{payload}</script></html>'


def test_site_wine_maps_fields_to_catalog_columns():
    card = site_wine(page(DATA), "gevyurcztraminer-oranzh")
    assert card["name"] == "Гевюрцтраминер Оранж"
    # Категория в виде дампа — только цвет.
    assert card["category"] == "Оранжевое"
    assert card["winery"] == "Усадьба Перовских"
    assert card["region"] == "Крым"
    assert card["grapes"] == "Гевюрцтраминер"
    assert card["photo_name"] == "Gevyurcztraminer_Oranzh_f3d914f332.webp"
    assert card["alcohol"] == 12


def test_site_wine_rejects_other_slug_and_missing_payload():
    assert site_wine(page(DATA), "other-wine") is None
    assert site_wine("<html></html>", "gevyurcztraminer-oranzh") is None
    assert nuxt_payload("<html></html>") is None


def test_load_sitemap_from_cache(tmp_path):
    cache = tmp_path / "wines-sitemap.xml"
    cache.write_text(
        "<urlset><url><loc>https://vino-svoe.ru/wines/a-wine</loc><lastmod>2026-09-08</lastmod>"
        "<image:image><image:loc>https://api.vino-svoe.ru/uploads/A_1234567890.webp</image:loc>"
        "</image:image></url></urlset>",
        encoding="utf-8",
    )
    table = load_sitemap(cache, refresh=False)
    assert table.to_dict("records") == [
        {"slug": "a-wine", "lastmod": "2026-09-08", "file": "A_1234567890.webp"}
    ]
