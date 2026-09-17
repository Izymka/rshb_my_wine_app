"""Живые кадры вин платформы — из папки телефона в `data/live/` с манифестом.

    uv run python scripts/import_live_photos.py
    uv run python scripts/import_live_photos.py --source "data/Photos-1-001 (7)" --force

Зачем отдельный набор рядом с `data/own`. Свой набор строился до хакатона под свой же каталог:
одна папка — одно вино, эталон `catalog.*` внутри, и загрузчик `load_own` без эталона падает.
Для каталога платформы эталон уже есть — вырезка в `data/catalog/images/<slug>.png`, — а кадр
с телефона либо соответствует какому-то slug, либо не соответствует ничему. Поэтому здесь
единица разметки — кадр, а не папка, и правда лежит в одной таблице `manifest.csv`:

    path, wine_id, true_slug, group, source, note

`true_slug` пустой означает «этого вина в каталоге нет» — на таких кадрах проверяется отказ.
`group` для незнакомых — `unknown-hard` (российская полка, кириллица, в каталоге есть близнец
или та же винодельня) или `unknown-easy` (импорт, латиница). Кадр с несколькими бутылками
серии помечается `group=multi`: детектор режет по самой крупной, и что считать верным ответом,
зависит от того, какую он выберет, — в метрики такие кадры по умолчанию не идут.

Разметка 46 кадров 22–23.08.2026 сделана по одному, по увеличенным этикеткам: кадры серий
перемежаются (IMG_0270 — Розовый Алушта среди Красных), размечать диапазонами нельзя.

В тот же манифест попадают ссылками, без копирования файлов:
- свой набор `data/own` целиком, включая каталожные кадры: для платформы это такие же
  телефонные снимки; Chateau Tamagne есть в каталоге, всё остальное — незнакомцы;
- три публичных кадра организаторов из `data/eval`.

HEIC переписывается в JPEG полного разрешения: скрипт оценки шлёт JPEG/WebP, и нам нужно
мерить ровно то, что уйдёт по сети. MP4 от Live Photo остаются в исходной папке.

**Новые съёмки** кладутся папками в `data/live_raw/` — по папке на вино, без таблицы в коде:

    data/live_raw/<slug из каталога>/IMG_0310.HEIC ...      известное вино, группа live
    data/live_raw/unknown-<что это>/IMG_0340.HEIC ...        незнакомец с российской полки (unknown-hard)
    data/live_raw/import-<что это>/IMG_0350.HEIC ...         импорт, латиница (unknown-easy)

Имя папки известного вина — ровно slug из `data/catalog/catalog.csv` (он же в
`docs/shooting_list.md`); опечатка в slug — ошибка при импорте, а не молчаливый незнакомец.
Кадр с несколькими бутылками серии — в подпапку `multi/` внутри папки вина. Скрипт
конвертирует, складывает в `data/live/<папка>/` и дописывает манифест; повторный запуск
ничего не пересчитывает.
"""

import argparse
import csv
from pathlib import Path

from wine_scanner.catalog import EVAL_ROOT, IMAGE_SUFFIXES, NON_WINE_DIRS, OWN_ROOT
from wine_scanner.embed import load_image

LIVE_ROOT = Path("data/live")
RAW_ROOT = Path("data/live_raw")
CATALOG_CSV = Path("data/catalog/catalog.csv")
MANIFEST = LIVE_ROOT / "manifest.csv"
DEFAULT_SOURCE = Path("data/Photos-1-001 (7)")
RAW_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"}
FIELDS = ("path", "wine_id", "true_slug", "group", "source", "note")

# Вина на кадрах 22–23.08.2026. Ключ — wine_id (он же имя папки), значение — slug каталога
# платформы или None, если вина в каталоге нет.
WINES = {
    "massandra-portveyn-krasnyy-alushta": (
        "massandra-portveyn-krasnyy-alushta-krasnye-sorta-vinograda-krasnoe-sladkoe-17"
    ),
    "massandra-portveyn-belyy-krymskiy": "massandra-portveyn-belyy-krymskiy-kokur-beloe-sladkoe-18",
    # Та же этикетка, что у красного, отличается одним словом. В каталоге такого нет —
    # самый ценный незнакомец во всём наборе.
    "unknown-massandra-portveyn-rozovyy-alushta-2023": None,
    "unknown-portia-roble-2023": None,
    "unknown-ai-petri-krym-saperavi": None,
}

# Кадр -> (wine_id, group, note). Группы: `live` — обычный кадр с одной главной бутылкой;
# заметка описывает, чем кадр труден, если это видно глазами.
FRAMES = {
    "IMG_0254": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", ""),
    "IMG_0255": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", ""),
    "IMG_0256": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", "partial"),
    "IMG_0257": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", ""),
    "IMG_0258": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", ""),
    "IMG_0259": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", "far"),
    "IMG_0260": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", ""),
    "IMG_0261": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0262": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0263": ("massandra-portveyn-krasnyy-alushta", "live", "partial"),
    "IMG_0264": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0265": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0266": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0267": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0268": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0269": ("massandra-portveyn-krasnyy-alushta", "live", ""),
    "IMG_0270": ("unknown-massandra-portveyn-rozovyy-alushta-2023", "unknown-hard", "blur"),
    "IMG_0271": ("massandra-portveyn-belyy-krymskiy", "live", ""),
    "IMG_0272": ("massandra-portveyn-belyy-krymskiy", "live", "partial"),
    "IMG_0273": ("massandra-portveyn-belyy-krymskiy", "live", ""),
    "IMG_0274": ("massandra-portveyn-belyy-krymskiy", "live", ""),
    "IMG_0275": ("massandra-portveyn-belyy-krymskiy", "live", "blur"),
    "IMG_0276": ("massandra-portveyn-belyy-krymskiy", "live", ""),
    "IMG_0277": ("massandra-portveyn-belyy-krymskiy", "live", ""),
    "IMG_0278": ("massandra-portveyn-belyy-krymskiy", "live", "blur"),
    # Полка: слева Белый Крымский (обрезан), в центре Красный Алушта целиком.
    "IMG_0279": ("massandra-portveyn-krasnyy-alushta", "multi", "полка, две Массандры"),
    "IMG_0281": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0282": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0283": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0284": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0285": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0286": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0287": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0288": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0289": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0290": ("unknown-portia-roble-2023", "unknown-easy", ""),
    "IMG_0294": ("unknown-portia-roble-2023", "unknown-easy", "far"),
    "IMG_0296": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    "IMG_0297": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    "IMG_0298": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    "IMG_0299": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    "IMG_0300": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    "IMG_0301": ("unknown-ai-petri-krym-saperavi", "unknown-hard", "far"),
    "IMG_0302": ("unknown-ai-petri-krym-saperavi", "unknown-hard", ""),
    # Три бутылки серии в ряд, Саперави в центре; всё равно незнакомец, но бутылка не одна.
    "IMG_0303": ("unknown-ai-petri-krym-saperavi", "multi", "полка, три Ай-Петри"),
    "IMG_0304": ("unknown-ai-petri-krym-saperavi", "unknown-hard", "соседи в кадре"),
}

# Свой набор -> slug платформы. Единственное совпадение проверено глазами 16.09.2026: Chateau
# Tamagne красное 2025 — тот же дизайн, что у карточки 2023 года (в каталоге только она).
# Inkerman Ркацители и Лабра Ашамта в каталоге отсутствуют, хотя написаны кириллицей и
# винодельня Инкерман в каталоге есть — это трудные незнакомцы. Остальное — импорт.
OWN_TO_PLATFORM = {
    "Chateau_Tamagne_красное_2025": "kuban-vino-shato-tamane-ruzh-2023-tsvaygelt-krasnoe-suhoe-12",
}
OWN_HARD_UNKNOWN = {"Inkerman_Ркацители_белое_полусладкое_2025", "Лабра_Ашамта_white_wine"}

# Публичные кадры организаторов. Массандра известна, два других — знакомая винодельня,
# незнакомая позиция (Табия без Пино Нуар, Аристов без Donum).
EVAL_FRAMES = {
    "02eef911.webp": (
        "massandra-muskatel-belyy",
        "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16",
        "live",
        "полка, три Массандры, целевая в центре",
    ),
    "019c68d0.jpg": ("unknown-tabiya-pino-nuar-2025", None, "unknown-hard", ""),
    "096ca74e.jpg": ("unknown-aristov-donum-xxiv-2023", None, "unknown-hard", ""),
}


def convert(source: Path, force: bool) -> list[dict]:
    rows = []
    missing = [name for name in FRAMES if not (source / f"{name}.HEIC").exists()]
    if missing:
        raise FileNotFoundError(f"в {source} нет кадров: {', '.join(missing)}")
    for name, (wine_id, group, note) in sorted(FRAMES.items()):
        out = LIVE_ROOT / wine_id / f"{name}.jpg"
        if force or not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            load_image(source / f"{name}.HEIC").save(out, quality=92)
        rows.append(
            {
                "path": str(out),
                "wine_id": wine_id,
                "true_slug": WINES[wine_id] or "",
                "group": group,
                "source": "live",
                "note": note,
            }
        )
    return rows


def raw_rows(root: Path, catalog_csv: Path, force: bool) -> list[dict]:
    """Съёмки из data/live_raw: папка на вино, имя папки — slug каталога или unknown-/import-."""
    if not root.exists():
        return []
    import csv as _csv

    with catalog_csv.open(encoding="utf-8", newline="") as f:
        slugs = {row["slug"] for row in _csv.DictReader(f)}

    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        wine_id = folder.name
        if wine_id.startswith("unknown-"):
            slug, group = "", "unknown-hard"
        elif wine_id.startswith("import-"):
            slug, group = "", "unknown-easy"
        elif wine_id in slugs:
            slug, group = wine_id, "live"
        else:
            raise SystemExit(
                f"папка {folder}: это не slug каталога и не unknown-/import- — проверьте имя "
                f"по docs/shooting_list.md"
            )
        files = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in RAW_SUFFIXES
        )
        if not files:
            print(f"пусто: {folder}")
        for path in files:
            multi = path.parent.name == "multi"
            out = LIVE_ROOT / wine_id / f"{path.stem}.jpg"
            if force or not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                load_image(path).save(out, quality=92)
            rows.append(
                {
                    "path": str(out),
                    "wine_id": wine_id,
                    "true_slug": slug,
                    "group": "multi" if multi else group,
                    "source": "live",
                    "note": "несколько бутылок" if multi else "",
                }
            )
    return rows


def own_rows(root: Path) -> list[dict]:
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name in NON_WINE_DIRS:
            continue
        slug = OWN_TO_PLATFORM.get(folder.name)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if slug:
                group = path.stem.split("_")[0]
            else:
                group = "unknown-hard" if folder.name in OWN_HARD_UNKNOWN else "unknown-easy"
            rows.append(
                {
                    "path": str(path),
                    "wine_id": folder.name,
                    "true_slug": slug or "",
                    "group": group,
                    "source": "own",
                    "note": "",
                }
            )
    return rows


def eval_rows(root: Path) -> list[dict]:
    rows = []
    for name, (wine_id, slug, group, note) in EVAL_FRAMES.items():
        path = root / "queries" / name
        if not path.exists():
            raise FileNotFoundError(f"нет публичного кадра {path}")
        rows.append(
            {
                "path": str(path),
                "wine_id": wine_id,
                "true_slug": slug or "",
                "group": group,
                "source": "eval",
                "note": note,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="папка с HEIC 22–23.08")
    parser.add_argument("--raw", type=Path, default=RAW_ROOT, help="новые съёмки папками по винам")
    parser.add_argument("--catalog", type=Path, default=CATALOG_CSV)
    parser.add_argument("--own", type=Path, default=OWN_ROOT)
    parser.add_argument("--eval", type=Path, default=EVAL_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--force", action="store_true", help="перезаписать уже сконвертированные JPEG"
    )
    args = parser.parse_args()

    rows = (
        (convert(args.source, args.force) if args.source.exists() else [])
        + raw_rows(args.raw, args.catalog, args.force)
        + own_rows(args.own)
        + eval_rows(args.eval)
    )
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["path"], row)
    rows = list(seen.values())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    known = [r for r in rows if r["true_slug"] and r["group"] != "multi"]
    print(f"манифест: {args.manifest}, кадров {len(rows)}")
    print(f"  известных: {len(known)} ({len({r['true_slug'] for r in known})} вин)")
    live_new = [r for r in rows if r["path"].startswith(str(LIVE_ROOT)) and r["source"] == "live"]
    print(f"  живых кадров всего: {len(live_new)}")
    for group in ("unknown-hard", "unknown-easy", "multi"):
        part = [r for r in rows if r["group"] == group]
        print(f"  {group}: {len(part)} ({len({r['wine_id'] for r in part})} вин)")


if __name__ == "__main__":
    main()
