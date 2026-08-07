"""Загрузка каталога вин.

Пока источник один — открытый X-Wines, на нём обкатываем пайплайн. Датасет платформы придёт
позже и подключится сюда же отдельной функцией: всё, что от каталога нужно остальному коду, —
это список CatalogItem. Так поздний датасет не потребует переписывать индекс, поиск и метрики.
"""

import json
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
    """Живое фото: путь, правильный ответ и группа сложности."""

    path: Path
    true_id: str
    group: str


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
            queries.append(Query(path=path, true_id=folder.name, group=path.stem.split("_")[0]))

    return catalog, queries
