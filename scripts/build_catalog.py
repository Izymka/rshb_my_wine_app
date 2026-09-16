"""Собрать каталог платформы из дампа Strapi: таблица карточек плюс эталон на каждый slug.

    uv run python scripts/build_catalog.py --resolve-via-site

Организаторы выдали CSV с карточками и папку uploads со всеми медиа сайта — 15 тысяч файлов,
из которых нам нужны две тысячи эталонов бутылок. Ключа между ними нет: в дампе записано
человеческое имя фото, в uploads лежит транслитерированное имя с хэшем. Скрипт восстанавливает
связь в три шага и складывает результат в data/catalog/, откуда его читает load_platform.

1. Транслитерация по таблице Strapi (`wine_scanner.catalog.strapi_key`) — закрывает почти всё.
2. Если под одно имя загружали несколько файлов, сравниваем байты: одинаковые — всё равно какой.
3. Остаток — разные файлы под одним именем и имена, которых в uploads нет, — разрешается через
   страницу вина на vino-svoe.ru: её og:image называет файл точно. Это единственное место,
   где скрипт ходит в сеть, и оно включается флагом.

Эталоны — вырезанные бутылки на прозрачном фоне. PIL при convert("RGB") делает прозрачное
чёрным, и весь каталог превратился бы в бутылки на чёрном, чего на живых фото не бывает.
Поэтому картинка сохраняется уже композитной, на белом, в PNG без потерь.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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

SITE_PAGE = "https://vino-svoe.ru/wines/{slug}"
# Нулевые размеры — «не уменьшать»: отдаёт оригинал, проверено на нескольких карточках.
SITE_ORIGINAL = "https://api.vino-svoe.ru/v1/img/str-api/0/0/resize/uploads/{file}"
USER_AGENT = "Mozilla/5.0 (wine-scanner catalog builder)"
OG_IMAGE = re.compile(r'property="og:image" content="([^"]+)"')

BACKGROUND = (255, 255, 255)


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def match_by_name(dump: pd.DataFrame, uploads: dict[str, list[Path]]) -> pd.DataFrame:
    """Шаги 1 и 2: сопоставить по имени, неоднозначности снять сравнением байтов."""
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


def site_file_name(slug: str) -> str | None:
    """Имя Strapi-файла эталона со страницы вина, или None, если страницы нет."""
    html = fetch(SITE_PAGE.format(slug=slug))
    if not html:
        return None
    found = OG_IMAGE.search(html.decode("utf-8", "ignore"))
    return found.group(1).rsplit("/", 1)[-1] if found else None


def resolve_via_site(
    matches: pd.DataFrame, uploads_root: Path, downloads: Path, cache_path: Path
) -> pd.DataFrame:
    """Шаг 3: спросить сайт про всё, что не сошлось по имени.

    Ответы кэшируются в json рядом с каталогом: повторный запуск скрипта не должен снова
    ходить в сеть ради тех же пятидесяти slug.
    """
    cache: dict[str, str | None] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    todo = [s for s in matches.loc[matches["match"].isin(["ambiguous", "missing"]), "slug"]]
    pending = [s for s in todo if s not in cache]
    if pending:
        with ThreadPoolExecutor(max_workers=4) as pool:
            names = tqdm(pool.map(site_file_name, pending), total=len(pending), desc="сайт")
            for slug, name in zip(pending, names, strict=True):
                cache[slug] = name
                time.sleep(0.3)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    matches = matches.set_index("slug")
    downloads.mkdir(parents=True, exist_ok=True)
    for slug in todo:
        name = cache.get(slug)
        if not name:
            continue
        local = uploads_root / name
        if local.exists():
            matches.loc[slug, ["match", "source_file"]] = ["site", local]
            continue
        target = downloads / name
        if not target.exists():
            payload = fetch(SITE_ORIGINAL.format(file=name))
            if not payload:
                continue
            target.write_bytes(payload)
        matches.loc[slug, ["match", "source_file"]] = ["site-download", target]
    return matches.reset_index()


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
        "--resolve-via-site",
        action="store_true",
        help="спросить vino-svoe.ru про slug, не сошедшиеся по имени",
    )
    parser.add_argument("--force", action="store_true", help="пересохранить уже готовые картинки")
    args = parser.parse_args()

    dump = load_platform_dump(args.dump)
    uploads = index_uploads(args.uploads)
    print(f"карточек в дампе: {len(dump)}, оригиналов в uploads: {sum(map(len, uploads.values()))}")

    matches = match_by_name(dump, uploads)
    print("по имени:", matches["match"].value_counts().to_dict())

    if args.resolve_via_site:
        matches = resolve_via_site(
            matches, args.uploads, args.out / "downloaded", args.out / "site_resolved.json"
        )
        print("после сайта:", matches["match"].value_counts().to_dict())

    table = dump.merge(matches, on="slug", validate="one_to_one")
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

    images_dir = args.out / "images"
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
