"""Сайт платформы «Своё Вино»: sitemap вин, оригиналы эталонов, карточка со страницы вина.

Общая часть `scripts/build_catalog.py` (сборка каталога из дампа) и `scripts/add_site_wines.py`
(заведение вин, появившихся на сайте после дампа).
"""

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

SITEMAP = "https://vino-svoe.ru/wines-sitemap.xml"
# Нулевые размеры — «не уменьшать»: отдаёт оригинал, проверено на нескольких карточках.
SITE_ORIGINAL = "https://api.vino-svoe.ru/v1/img/str-api/0/0/resize/uploads/{file}"
USER_AGENT = "Mozilla/5.0 (wine-scanner catalog builder)"
SITEMAP_ENTRY = re.compile(
    r"<url><loc>https://vino-svoe\.ru/wines/(?P<slug>[^<]+)</loc>"
    r"(?:<lastmod>(?P<lastmod>[^<]*)</lastmod>)?.*?<image:loc>[^<]*/uploads/(?P<file>[^<]+)</image:loc>",
    re.S,
)

BACKGROUND = (255, 255, 255)


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(url: str, timeout: float = 20.0, retries: int = 1) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception:  # noqa: BLE001 — любая сетевая беда лечится одинаково: повтором
            if attempt == retries:
                return None
            time.sleep(1.0)
    return None


def load_sitemap(cache_path: Path, refresh: bool) -> pd.DataFrame:
    """Опубликованные вина сайта: slug, файл эталона, дата правки.

    Свежий sitemap качается, если кэша нет или попросили обновить; иначе читается кэш — сборка
    каталога не должна зависеть от сети. Пустая таблица означает «sitemap недоступен», и тогда
    всё сопоставление идёт по транслитерации.
    """
    if refresh or not cache_path.exists():
        payload = fetch(SITEMAP)
        if payload:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
    if not cache_path.exists():
        return pd.DataFrame(columns=["slug", "lastmod", "file"])
    xml = cache_path.read_text(encoding="utf-8")
    rows = [(m["slug"], m["lastmod"] or "", m["file"]) for m in SITEMAP_ENTRY.finditer(xml)]
    return pd.DataFrame(rows, columns=["slug", "lastmod", "file"])


def flatten(source: Path, target: Path) -> tuple[int, int, str]:
    """Сохранить эталон композитным RGB. Возвращает размер и исходный режим."""
    image = ImageOps.exif_transpose(Image.open(source))
    width, height = image.size
    mode = image.mode
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, BACKGROUND)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        image = canvas
    else:
        image = image.convert("RGB")
    image.save(target, format="PNG")
    return width, height, mode


# --- Карточка вина со страницы сайта ---------------------------------------------------------
#
# Страница https://vino-svoe.ru/wines/<slug> — Nuxt с серверным рендером: всё, что знает
# страница, лежит в <script id="__NUXT_DATA__"> в формате devalue — плоский JSON-массив, где
# объект хранит не значения, а номера элементов массива. Карточка вина — `data["wine-<slug>"].wine`
# (проверено 17.10.2026 на gevyurcztraminer-oranzh) — несёт все поля дампа: title, category,
# color, region, grapes, description, manufacturer, image.

SITE_WINE_URL = "https://vino-svoe.ru/wines/{slug}"
_NUXT_DATA = re.compile(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.S)
# Обёртки реактивности Vue: ["Reactive", i] — это просто значение i.
_DEVALUE_WRAPPERS = {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "EmptyRef"}
# Отрицательные номера devalue — константы.
_DEVALUE_CONSTANTS = {-1: None, -2: None, -3: float("nan"), -4: float("inf"), -5: -float("inf")}


def nuxt_payload(html: str) -> dict | None:
    """Развёрнутый `__NUXT_DATA__` страницы или None, если блока нет."""
    match = _NUXT_DATA.search(html)
    if match is None:
        return None
    data = json.loads(match.group(1))

    def resolve(i, depth=0):
        if isinstance(i, int) and i < 0:
            return _DEVALUE_CONSTANTS.get(i)
        if depth > 64:
            raise ValueError("__NUXT_DATA__: слишком глубокая вложенность")
        value = data[i]
        if isinstance(value, dict):
            return {k: resolve(v, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                if value[0] in _DEVALUE_WRAPPERS:
                    return resolve(value[1], depth + 1) if len(value) > 1 else None
                return None  # Date, Set, Map и прочее — карточке не нужно
            return [resolve(v, depth + 1) for v in value]
        return value

    root = resolve(0)
    return root if isinstance(root, dict) else None


def _name(value) -> str:
    """Поле-справочник Strapi ({"name": ...}) или строка как есть."""
    if isinstance(value, dict):
        value = value.get("name")
    return str(value).strip() if value else ""


def site_wine(html: str, slug: str) -> dict | None:
    """Карточка вина со страницы сайта в колонках `catalog.csv`; None — карточки нет.

    `image_url` — путь эталона на сайте (`/uploads/<файл>` или полный URL), `photo_name` —
    имя файла, как у дампа. Поля, которых нет в дампе (крепость, температура, рейтинг), тоже
    отдаются: таблица их не хранит, но отчёту они пригодятся.
    """
    payload = nuxt_payload(html)
    if not payload:
        return None
    entry = (payload.get("data") or {}).get(f"wine-{slug}") or {}
    wine = entry.get("wine") if isinstance(entry, dict) else None
    if not isinstance(wine, dict) or wine.get("slug") != slug:
        return None
    image = wine.get("image") or {}
    url = image.get("url") if isinstance(image, dict) else None
    grapes = wine.get("grapes") or []
    return {
        "name": str(wine.get("title") or "").strip(),
        # На сайте «Белое сухое», в дампе — только цвет («Белое»): каталог держит вид дампа.
        "category": (_name(wine.get("category")).split() or [""])[0],
        "color": _name(wine.get("color")),
        "region": _name(wine.get("region")),
        "grapes": ", ".join(n for n in (_name(g) for g in grapes) if n),
        "description": str(wine.get("description") or "").strip(),
        "winery": _name(wine.get("manufacturer")),
        "slug": slug,
        "photo_name": Path(url).name if url else "",
        "image_url": url or "",
        "alcohol": wine.get("alcohol"),
        "temperature": wine.get("temperature"),
    }
