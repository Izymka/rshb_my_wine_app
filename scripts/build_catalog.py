"""Собрать каталог платформы из дампа Strapi: таблица карточек плюс эталон на каждый slug.

    uv run python scripts/build_catalog.py

Организаторы выдали CSV с карточками и папку uploads со всеми медиа сайта — 15 тысяч файлов,
из которых нам нужны две тысячи эталонов бутылок. Ключа между ними нет: в дампе записано
человеческое имя фото, в uploads лежит транслитерированное имя с хэшем. Скрипт восстанавливает
связь в два шага и складывает результат в data/catalog/, откуда его читает load_platform.

1. Sitemap сайта (подсказка организаторов): wines-sitemap.xml — image-sitemap, в нём у каждого
   опубликованного вина точное имя файла эталона. Один запрос закрывает 2037 slug из 2103.
   Ответ кэшируется в data/catalog/wines-sitemap.xml; без сети берётся кэш, без кэша — шаг 2.
2. Транслитерация по таблице Strapi (`wine_scanner.catalog.strapi_key`) — для вин, которых на
   сайте нет (не опубликованы). Если под одно имя загружали несколько файлов, сравниваем байты:
   одинаковые — всё равно какой, разные — берём первый и записываем в unresolved.csv.

Файла из sitemap может не оказаться в uploads (дамп старше сайта) — тогда он качается с сайта
в оригинальном размере. Это единственный случай, когда скрипт ходит в сеть дальше sitemap.

Эталоны — вырезанные бутылки на прозрачном фоне. PIL при convert("RGB") делает прозрачное
чёрным, и весь каталог превратился бы в бутылки на чёрном, чего на живых фото не бывает.
Поэтому картинка сохраняется уже композитной, на белом, в PNG без потерь.
"""

import argparse
import hashlib
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from wine_scanner.catalog import (
    PLATFORM_DUMP,
    PLATFORM_ROOT,
    UPLOADS_ROOT,
    index_uploads,
    load_platform_dump,
    strapi_key,
    vintage_from_name,
)

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


def match_by_sitemap(
    dump: pd.DataFrame, sitemap: pd.DataFrame, uploads_root: Path, downloads: Path
) -> pd.DataFrame:
    """Шаг 1: точное имя файла из sitemap; нет в uploads — скачать с сайта."""
    by_slug = dict(zip(sitemap["slug"], sitemap["file"], strict=True))
    rows = []
    for slug in dump["slug"]:
        name = by_slug.get(slug)
        if name is None:
            rows.append((slug, None, None))
            continue
        local = uploads_root / name
        if local.exists():
            rows.append((slug, "sitemap", local))
            continue
        target = downloads / name
        if not target.exists():
            payload = fetch(SITE_ORIGINAL.format(file=name))
            if not payload:
                rows.append((slug, None, None))
                continue
            downloads.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        rows.append((slug, "sitemap-download", target))
    return pd.DataFrame(rows, columns=["slug", "match", "source_file"])


def match_by_name(dump: pd.DataFrame, uploads: dict[str, list[Path]]) -> pd.DataFrame:
    """Шаг 2: сопоставить по имени, неоднозначности снять сравнением байтов."""
    rows = []
    for row in dump.itertuples(index=False):
        candidates = uploads.get(strapi_key(Path(row.photo_name).stem), [])
        if len(candidates) == 1:
            rows.append((row.slug, "exact", candidates[0]))
        elif candidates and len({digest(p) for p in candidates}) == 1:
            rows.append((row.slug, "bytes", candidates[0]))
        elif candidates:
            rows.append((row.slug, "ambiguous", candidates[0]))
        else:
            rows.append((row.slug, "missing", None))
    return pd.DataFrame(rows, columns=["slug", "match", "source_file"])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, default=PLATFORM_DUMP)
    parser.add_argument("--uploads", type=Path, default=UPLOADS_ROOT)
    parser.add_argument("--out", type=Path, default=PLATFORM_ROOT)
    parser.add_argument(
        "--refresh-sitemap", action="store_true", help="скачать sitemap заново, а не брать кэш"
    )
    parser.add_argument(
        "--no-sitemap", action="store_true", help="только транслитерация, без sitemap и сети"
    )
    parser.add_argument("--force", action="store_true", help="пересохранить уже готовые картинки")
    args = parser.parse_args()

    dump = load_platform_dump(args.dump)
    uploads = index_uploads(args.uploads)
    print(f"карточек в дампе: {len(dump)}, оригиналов в uploads: {sum(map(len, uploads.values()))}")

    by_name = match_by_name(dump, uploads)
    if args.no_sitemap:
        matches = by_name
        sitemap = pd.DataFrame(columns=["slug", "lastmod", "file"])
    else:
        sitemap = load_sitemap(args.out / "wines-sitemap.xml", args.refresh_sitemap)
        print(f"вин в sitemap: {len(sitemap)}")
        by_sitemap = match_by_sitemap(dump, sitemap, args.uploads, args.out / "downloaded")
        # Sitemap — первичный источник, транслитерация закрывает то, чего на сайте нет.
        matches = by_sitemap.where(by_sitemap["match"].notna(), by_name)
        new_on_site = sorted(set(sitemap["slug"]) - set(dump["slug"]))
        if new_on_site:
            print(f"на сайте есть, в дампе нет ({len(new_on_site)}): {', '.join(new_on_site)}")
    print("сопоставление:", matches["match"].value_counts().to_dict())

    table = dump.merge(matches, on="slug", validate="one_to_one")
    table["on_site"] = table["slug"].isin(sitemap["slug"])
    table["vintage"] = [
        vintage_from_name(n, s) for n, s in zip(table["name"], table["slug"], strict=True)
    ]
    table["vintage"] = table["vintage"].astype("Int64")
    table["shared_photo"] = table["photo_name"].duplicated(keep=False)
    # Один и тот же файл под разными именами — тоже бывает (в дампе таких больше, чем с общим
    # именем). По картинке такие slug неразличимы, и знать их надо поимённо.
    table["image_md5"] = [
        digest(Path(p)) if isinstance(p, Path | str) and not pd.isna(p) else None
        for p in table["source_file"]
    ]
    table["shared_image"] = table["image_md5"].duplicated(keep=False) & table["image_md5"].notna()

    images_dir = args.out / "originals"
    images_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, tuple[int, int, str]] = {}
    for row in tqdm(list(table.itertuples(index=False)), desc="эталоны"):
        if row.source_file is None or pd.isna(row.source_file):
            continue
        target = images_dir / f"{row.slug}.png"
        source = Path(row.source_file)
        if target.exists() and not args.force:
            with Image.open(source) as original:
                sizes[row.slug] = (*original.size, original.mode)
            continue
        sizes[row.slug] = flatten(source, target)

    table["width"] = [sizes.get(s, (None,))[0] for s in table["slug"]]
    table["height"] = [sizes.get(s, (None, None))[1] for s in table["slug"]]
    table["mode"] = [sizes.get(s, (None, None, None))[2] for s in table["slug"]]
    table["source_file"] = [
        str(Path(p).relative_to(Path.cwd())) if isinstance(p, Path) and p.is_absolute() else p
        for p in table["source_file"]
    ]
    table.to_csv(args.out / "catalog.csv", index=False)

    unresolved = table.loc[table["match"].isin(["ambiguous", "missing"])]
    unresolved[["slug", "name", "winery", "photo_name", "match", "source_file"]].to_csv(
        args.out / "unresolved.csv", index=False
    )
    print(
        f"готово: {args.out / 'catalog.csv'} ({len(table)} строк), картинок {len(sizes)}, "
        f"не разрешено {len(unresolved)} -> {args.out / 'unresolved.csv'}"
    )


if __name__ == "__main__":
    main()
