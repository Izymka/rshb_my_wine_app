"""Загрузка каталога вин.

Источников три: открытый X-Wines (на нём обкатывался пайплайн), свой тестовый набор и каталог
платформы «Своё Вино», выданный на хакатоне. Всё, что от каталога нужно остальному коду, —
это список CatalogItem, поэтому каждый источник подключается своей функцией, а индекс, поиск
и метрики про разницу между ними не знают.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

XWINES_ROOT = Path("data/third-party datasets/X-Wines Slim v1.0")
OWN_ROOT = Path("data/own")

IMAGE_SUFFIXES = {".heic", ".jpg", ".jpeg", ".png"}

# Папки внутри data/own, которые не являются винами.
NON_WINE_DIRS = {"shelf_raw"}


@dataclass
class CatalogItem:
    item_id: str
    image_path: Path
    payload: dict = field(default_factory=dict)


@dataclass
class Query:
    """Живое фото: путь, правильный ответ и группа сложности.

    `wine_id` — человеческое имя вина (папка), по нему бенчмарк группирует кадры; у своего
    набора оно совпадает с `true_id`. `source` — откуда кадр: own / live / eval. `note` —
    что на кадре трудного, если это видно глазами (blur, far, partial, соседи).
    """

    path: Path
    true_id: str
    group: str
    wine_id: str = ""
    source: str = "own"
    note: str = ""

    @property
    def known(self) -> bool:
        """Есть ли верный ответ в каталоге. Незнакомцы размечены `unknown:<имя>`."""
        return not self.true_id.startswith(UNKNOWN_PREFIX)


# Метка запроса, у которого верного ответа в каталоге нет. На таких кадрах проверяется отказ;
# соглашение общее для build_platform_features.py, train_decider.py и бенчмарка платформы.
UNKNOWN_PREFIX = "unknown:"


def load_xwines(root: Path = XWINES_ROOT) -> list[CatalogItem]:
    """Собрать карточки X-Wines: картинка этикетки плюс поля из CSV.

    Картинки лежат как {WineID}.jpeg во вложенной папке, часть строк CSV без картинки —
    такие пропускаем.
    """
    images_dir = root / "XWines_Slim_1K_labels-80" / "XWines_Slim_1K_labels-80"
    wines = pd.read_csv(root / "XWines_Slim_1K_wines.csv")

    items = []
    for row in wines.itertuples():
        image_path = images_dir / f"{row.WineID}.jpeg"
        if not image_path.exists():
            continue
        items.append(
            CatalogItem(
                item_id=str(row.WineID),
                image_path=image_path,
                payload={
                    "name": row.WineName,
                    "winery": row.WineryName,
                    "country": row.Country,
                    "region": row.RegionName,
                    "type": row.Type,
                    "grapes": row.Grapes,
                    "vintages": row.Vintages,
                },
            )
        )
    return items


def load_own(root: Path = OWN_ROOT) -> tuple[list[CatalogItem], list[Query]]:
    """Разобрать свой набор на каталог и запросы.

    Одна папка — одно вино. Внутри catalog.* идёт в индекс, всё остальное становится запросами.
    Правильный ответ — имя папки, группа сложности — префикс имени файла до подчёркивания
    (angle_01.HEIC -> группа angle). Отдельной таблицы соответствий нет намеренно: она бы
    разошлась с файлами при первой же пересъёмке.
    """
    catalog: list[CatalogItem] = []
    queries: list[Query] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name in NON_WINE_DIRS:
            continue

        meta_path = folder / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        # Фильтруем по расширениям, а не берём всё подряд: Finder насыпает .DS_Store,
        # а iPhone кладёт рядом с HEIC ещё и MP4 от Live Photo.
        images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]

        catalog_images = [p for p in images if p.stem.lower() == "catalog"]
        if not catalog_images:
            raise FileNotFoundError(f"в {folder} нет catalog.* — без эталона вино искать не в чем")

        catalog.append(CatalogItem(folder.name, catalog_images[0], payload))
        for path in images:
            if path in catalog_images:
                continue
            queries.append(
                Query(
                    path=path,
                    true_id=folder.name,
                    group=path.stem.split("_")[0],
                    wine_id=folder.name,
                )
            )

    return catalog, queries


# ---------------------------------------------------------------------------------------------
# Каталог платформы «Своё Вино».
#
# Организаторы выдали два не связанных между собой артефакта: CSV-дамп карточек, где у каждой
# записано человеческое имя фото («Поместье Голубицкое Рислинг.webp»), и папку uploads Strapi,
# где тот же файл лежит как Pomeste_Golubiczkoe_Risling_02c2e84311.webp. Ключа между ними нет,
# поэтому связь приходится восстанавливать по имени: Strapi транслитерирует и добавляет хэш.
# Всё, что здесь ниже, — про это восстановление и про то, как собранный каталог отдаётся
# остальному коду в виде тех же CatalogItem, что и X-Wines со своим набором.
# ---------------------------------------------------------------------------------------------

PLATFORM_ROOT = Path("data/catalog")
# Эталоны платформы — вырезанные бутылки, композит на белом, `originals/<slug>.png`. Это
# исходники: их режет детектор при сборке индекса, руками они не правятся.
PLATFORM_IMAGES = PLATFORM_ROOT / "originals"
PLATFORM_DUMP = Path("data/strapi_output0709.csv")
UPLOADS_ROOT = Path("data/uploads")
EVAL_ROOT = Path("data/eval")

# Колонки дампа -> имена, которыми пользуется остальной код. Порядок сохранён.
DUMP_COLUMNS = {
    "Название вина": "name",
    "Категория": "category",
    "Цвет": "color",
    "Регион": "region",
    "Сорт винограда": "grapes",
    "Описание": "description",
    "Винодельня": "winery",
    "Slug": "slug",
    "Название фото": "photo_name",
}

# Таблица транслитерации Strapi, восстановленная по парам «имя в дампе — файл в uploads».
# Не ГОСТ и не ISO 9: ц -> cz, но х -> h, а не x. Подобрана так, чтобы сходились все примеры,
# которые удалось проверить руками; на дампе от 07.09 даёт 2096 совпадений из 2103.
_STRAPI_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "cz",
    "ч": "ch", "ш": "sh", "щ": "shh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}  # fmt: skip

# Файлы uploads: Strapi хранит оригинал как <имя>_<10 hex>.<ext>, а рядом кладёт уменьшенные
# копии с префиксом размера. Нам нужны только оригиналы.
_STRAPI_FILE = re.compile(r"^(?P<stem>.+)_(?P<hash>[0-9a-f]{10})\.(?P<ext>[A-Za-z0-9]+)$")
_STRAPI_SIZE_PREFIX = re.compile(r"^(large|medium|small|thumbnail)_")


def strapi_key(name: str) -> str:
    """Свести имя файла к ключу, одинаковому для дампа и для uploads.

    Транслитерируем кириллицу, выбрасываем всё, кроме латиницы и цифр, приводим к нижнему
    регистру. Расширение и хэш вызывающая сторона отрезает сама: у дампа расширение своё
    (там почти всегда .webp), у uploads — своё, и они не обязаны совпадать.
    """
    text = unicodedata.normalize("NFKC", name)
    text = "".join(_STRAPI_TRANSLIT.get(ch.lower(), ch) for ch in text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def index_uploads(root: Path = UPLOADS_ROOT) -> dict[str, list[Path]]:
    """Оригиналы uploads, сгруппированные по ключу strapi_key.

    Один ключ может дать несколько файлов: одно и то же имя загружали в Strapi не раз, и каждая
    загрузка получила свой хэш. Разбирать, какой из них привязан к карточке, — дело вызывающего.
    """
    groups: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or _STRAPI_SIZE_PREFIX.match(path.name):
            continue
        match = _STRAPI_FILE.match(path.name)
        if match is None:
            continue
        groups.setdefault(strapi_key(match.group("stem")), []).append(path)
    return groups


def load_platform_dump(path: Path = PLATFORM_DUMP) -> pd.DataFrame:
    """Дамп каталога как таблица: одна строка на slug.

    В выгрузке каждая строка встречается дважды — это артефакт экспорта, а не два вина, поэтому
    точные дубли снимаются молча. Дубли по slug с разным содержимым были бы уже ошибкой данных,
    и их мы не скрываем: assert ниже упадёт с внятным сообщением.
    """
    frame = pd.read_csv(path).rename(columns=DUMP_COLUMNS).drop_duplicates()
    duplicated = frame["slug"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"slug повторяется с разным содержимым: {sorted(frame.loc[duplicated, 'slug'])[:5]}"
        )
    frame["name"] = frame["name"].str.strip()
    frame["winery"] = frame["winery"].str.strip()
    return frame.reset_index(drop=True)


_YEAR_IN_TEXT = re.compile(r"(?<!\d)(20[0-3]\d)(?!\d)")


def vintage_from_name(name: str, slug: str) -> int | None:
    """Год урожая, если он записан в названии или в slug.

    Ничего умнее регулярного выражения здесь не нужно: у платформы год либо стоит в названии
    («Алиготе Баррель, 2024»), либо замыкает slug (aligote-barrel-2024), либо его нет вовсе.
    """
    for text in (name, slug):
        match = _YEAR_IN_TEXT.search(text)
        if match:
            return int(match.group(1))
    return None


def load_platform(root: Path = PLATFORM_ROOT) -> list[CatalogItem]:
    """Каталог платформы из data/catalog, собранного scripts/build_catalog.py.

    Карточка без картинки в каталог не попадает: искать её нечем, а держать в индексе
    пустую строку хуже, чем честно не знать вино. Сколько таких — печатается, чтобы
    пропажа не прошла незамеченной.
    """
    table = pd.read_csv(root / "catalog.csv", dtype={"vintage": "Int64"})
    items: list[CatalogItem] = []
    skipped = 0
    for row in table.itertuples(index=False):
        image_path = root / "originals" / f"{row.slug}.png"
        if not image_path.exists():
            skipped += 1
            continue
        payload = {
            "name": row.name,
            "category": row.category,
            "color": row.color,
            "region": row.region,
            "grapes": row.grapes if isinstance(row.grapes, str) else "",
            "description": row.description,
            "winery": row.winery,
            "slug": row.slug,
            "vintage": int(row.vintage) if pd.notna(row.vintage) else None,
        }
        items.append(CatalogItem(item_id=row.slug, image_path=image_path, payload=payload))
    if skipped:
        print(f"карточек без картинки пропущено: {skipped}")
    return items


TEST_ROOT = Path("data/test")
TRAIN_ROOT = Path("data/train")
TEST_MANIFEST = TEST_ROOT / "manifest.csv"
TRAIN_MANIFEST = TRAIN_ROOT / "manifest.csv"

# --- Тестовый набор изолирован ---------------------------------------------------------------
#
# Договорённость 17.09.2026, раскладка 18.09.2026: учить что угодно можно на выданном каталоге,
# синтетике из него, открытых датасетах и `data/train` (свой набор `data/own` — импорт, снят во
# Вьетнаме — и та часть съёмок российских вин, что не вошла в тест). `data/test` — все снятые
# вина, которые есть в каталоге, плюс 50 незнакомых с российской полки (выбраны случайно, seed
# 2026, из 62), — и публичные кадры организаторов `data/eval` — только тест. Ни решающий слой,
# ни детектор, ни порог отказа их не видят. Внутри `data/test` лежат и кадры под разметку
# детектора (`frames_*`) с вырезками (`labels`) — они изолированы тем же путём.
# С 23.09.2026 кадры organizer_100 копируются в `data/train` (scripts/import_organizer_photos.py,
# без вин и файлов теста): по своему пути в `data/eval` они остаются holdout, копии — train.
# Проверка — по пути файла: это единственное, что есть у любого запроса в любом файле
# признаков, и подделать её случайно нельзя.
HOLDOUT_ROOTS = (TEST_ROOT, EVAL_ROOT, Path("data/live"), Path("data/live_labels"))


def is_holdout(path: str | Path) -> bool:
    """Кадр из изолированного тестового набора — учиться на нём нельзя."""
    resolved = Path(path).resolve()
    if any(resolved.is_relative_to(root.resolve()) for root in HOLDOUT_ROOTS):
        return True
    derived = Path("data/derived").resolve()
    return resolved.is_relative_to(derived) and bool(
        {"test", "eval", "live", "live_labels"} & set(resolved.relative_to(derived).parts)
    )


def load_manifest(manifest: Path = TEST_MANIFEST, include_multi: bool = False) -> list[Query]:
    """Кадры по манифесту `scripts/import_live_photos.py` (`data/test` или `data/train`).

    В отличие от `load_own`, эталона рядом с кадрами нет и не нужно: он уже в каталоге
    платформы под `true_slug`. Пустой `true_slug` — вина в каталоге нет, `true_id` получает
    метку `unknown:<wine_id>`. Кадры с несколькими бутылками (`group == multi`) по умолчанию
    не берутся: что на них считать верным ответом, зависит от того, какую бутылку выберет
    детектор. Отсутствующие файлы — ошибка со списком, а не молчаливый пропуск: пропавший
    кадр меняет метрику, и лучше узнать об этом сразу.
    """
    table = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    queries: list[Query] = []
    missing: list[str] = []
    for row in table.itertuples(index=False):
        if row.group == "multi" and not include_multi:
            continue
        path = Path(row.path)
        if not path.exists():
            missing.append(str(path))
            continue
        queries.append(
            Query(
                path=path,
                true_id=row.true_slug or f"{UNKNOWN_PREFIX}{row.wine_id}",
                group=row.group,
                wine_id=row.wine_id,
                source=row.source,
                note=row.note,
            )
        )
    if missing:
        raise FileNotFoundError(
            f"в манифесте {manifest} нет на диске {len(missing)} кадров: "
            + ", ".join(missing[:5])
            + (" …" if len(missing) > 5 else "")
        )
    return queries


load_live = load_manifest  # прежнее имя, до раскладки 18.09 манифест был один — data/live


EQUIVALENCES = Path("data/splits/catalog_slug_equivalences.csv")
# Карточки с ошибкой контента, которые нельзя отдавать вовсе: например, фото чужой бутылки.
# Подменить их каноническим slug нельзя — картинка притягивает кадры другого вина.
EXCLUDED = Path("data/splits/catalog_excluded_slugs.csv")


@dataclass
class Equivalences:
    """Решения человека о дублях каталога (`data/splits/catalog_slug_equivalences.csv`).

    Сюда попадают только пары с `decision == equivalent` — одно вино, заведённое дважды.
    `canonical` — какой slug отдавать вместо дубля (у второго, например, страница 404);
    у пары без канонического slug сервис отвечает как есть, а метрика засчитывает любой из двух.
    """

    canonical: dict[str, str] = field(default_factory=dict)
    group_of: dict[str, frozenset[str]] = field(default_factory=dict)
    # Slug, которые сервис не отдаёт никогда (`data/splits/catalog_excluded_slugs.csv`).
    excluded: frozenset[str] = frozenset()

    def same(self, a: str, b: str) -> bool:
        return a == b or b in self.group_of.get(a, ())


def load_equivalences(path: Path = EQUIVALENCES, excluded: Path = EXCLUDED) -> Equivalences:
    result = Equivalences()
    if excluded.exists():
        table = pd.read_csv(excluded, dtype=str, keep_default_na=False)
        result.excluded = frozenset(table.slug) - {""}
    if not path.exists():
        return result
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for row in table.itertuples(index=False):
        if row.decision != "equivalent":
            continue
        group = frozenset({row.slug_a, row.slug_b} | set(result.group_of.get(row.slug_a, ()))
                          | set(result.group_of.get(row.slug_b, ())))
        for slug in group:
            result.group_of[slug] = group
        if row.canonical_slug:
            if row.canonical_slug not in group:
                raise ValueError(f"canonical_slug {row.canonical_slug} не из пары {row.slug_a}")
            for slug in group - {row.canonical_slug}:
                result.canonical[slug] = row.canonical_slug
    return result


def load_eval_queries(root: Path = EVAL_ROOT) -> list[Path]:
    """Кадры публичного набора организаторов в порядке queries.tsv.

    Правильных ответов у нас нет — ключ остаётся у кейсодержателя, — поэтому это просто пути,
    а не Query с true_id.
    """
    manifest = pd.read_csv(root / "queries.tsv", sep="\t")
    return [root / "queries" / name for name in manifest["image_path"]]
