"""Псевдофото из вырезок каталога: положительные примеры там, где живых снимков нет.

    uv run python scripts/synthesize_queries.py --n 300

Каталог платформы — вырезанные бутылки на прозрачном фоне, а запросы — фотографии с полки.
Живых снимков вин из этого каталога у нас единицы, и решающему слою учиться не на чем. Здесь
из вырезки делается снимок: бутылка кладётся на кадр реальной полки (`data/own/shelf_raw`,
219 кадров, снятых для детектора), уменьшается до телефонного масштаба, наклоняется, слегка
искажается перспективой, получает блик, смаз, шум и JPEG-сжатие.

Это не замена съёмке. Псевдофото сделано из той же картинки, что лежит в индексе, и локальные
признаки на нём сходятся легче, чем на настоящем снимке. Зато их можно сделать для всех 2103
карточек, и модель впервые увидит, как выглядит «верный ответ есть» и «верного ответа нет» на
этом каталоге. Разбор ограничений — в CLAUDE.md, «Данные хакатона».
"""

import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

from wine_scanner.catalog import PLATFORM_ROOT
from wine_scanner.embed import load_image

SHELVES = Path("data/own/shelf_raw")
OUT_ROOT = Path("data/synthetic")
CANVAS = (1200, 1600)  # портрет, как снимает телефон после уменьшения


def cutout(path: Path) -> Image.Image:
    """Бутылка с альфа-каналом, обрезанная по содержимому.

    У 155 карточек прозрачности нет — это студийный кадр на белом. Для них альфа строится
    из почти белых пикселей: грубо, но бутылка на белом отделяется порогом надёжно, а точность
    края здесь ничего не решает.
    """
    image = ImageOps.exif_transpose(Image.open(path))
    transparent = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if not transparent:
        rgb = np.asarray(image.convert("RGB"))
        alpha = np.where(rgb.min(axis=2) > 235, 0, 255).astype(np.uint8)
        image = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    image = image.convert("RGBA")
    box = image.getchannel("A").getbbox()
    return image.crop(box) if box else image


def perspective(image: Image.Image, rng: random.Random, strength: float = 0.06) -> Image.Image:
    """Слегка перекосить кадр, как при съёмке под углом."""
    w, h = image.size
    dx, dy = w * strength, h * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jx, jy = (lambda: rng.uniform(0, dx)), (lambda: rng.uniform(0, dy))
    dst = np.float32([[jx(), jy()], [w - jx(), jy()], [w - jx(), h - jy()], [jx(), h - jy()]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def glare(image: Image.Image, rng: random.Random) -> Image.Image:
    """Белёсый блик с мягким краем — то, что даёт лампа над полкой на стекле."""
    w, h = image.size
    layer = Image.new("L", image.size, 0)
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.25, 0.75) * h
    rx, ry = rng.uniform(0.05, 0.15) * w, rng.uniform(0.15, 0.4) * h
    ImageDraw.Draw(layer).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(15, 40)))
    alpha = rng.uniform(0.35, 0.75)
    white = Image.new("RGB", image.size, (255, 255, 250))
    mask = layer.point(lambda v: int(v * alpha))
    return Image.composite(white, image, mask)


def background(shelf: Image.Image, rng: random.Random) -> Image.Image:
    """Полка позади бутылки: отодвинута и не в фокусе.

    Кадры shelf_raw сняты вплотную, и бутылки на них крупнее любой вклеенной — первый вариант
    синтеза на этом и сломался: каскад детекторов честно выбирал самую большую бутылку в кадре,
    то есть чужую, и верная карточка в 73 % псевдофото не искалась вовсе. Поэтому полка
    вписывается в кадр целиком с запасом (её бутылки становятся мелкими) и размывается, как
    фон при съёмке с близкого расстояния.
    """
    zoom = rng.uniform(0.45, 0.75)
    small = ImageOps.fit(shelf, (int(CANVAS[0] * zoom), int(CANVAS[1] * zoom)), Image.BICUBIC)
    # Мозаика из уменьшенной полки, чтобы заполнить кадр без пустых полей.
    canvas = Image.new("RGB", CANVAS)
    for x in range(0, CANVAS[0], small.width):
        for y in range(0, CANVAS[1], small.height):
            canvas.paste(small, (x, y))
    return canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 4.0)))


def synthesize(bottle: Image.Image, shelf: Image.Image, rng: random.Random) -> Image.Image:
    canvas = background(shelf, rng)

    # Бутылка занимает от 55 до 90 % высоты кадра — она главная в кадре, как требует ТЗ,
    # но и «далеко» из групп сложности в этом диапазоне есть.
    height = int(CANVAS[1] * rng.uniform(0.55, 0.9))
    scale = height / bottle.height
    bottle = bottle.resize((max(1, int(bottle.width * scale)), height), Image.BICUBIC)
    bottle = bottle.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, expand=True)
    # Тон бутылки под свет полки.
    tone = ImageEnhance.Brightness(bottle.convert("RGB")).enhance(rng.uniform(0.75, 1.15))
    bottle = Image.merge("RGBA", (*tone.split(), bottle.getchannel("A")))

    x = rng.randint(0, max(0, CANVAS[0] - bottle.width))
    y = rng.randint(0, max(0, CANVAS[1] - bottle.height))
    canvas.paste(bottle, (x, y), bottle)

    canvas = perspective(canvas, rng)
    if rng.random() < 0.45:
        canvas = glare(canvas, rng)
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.8, 1.2))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.5)))

    array = np.asarray(canvas).astype(np.float32)
    array += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(1, 6), array.shape)
    canvas = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=rng.randint(55, 85))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="сколько карточек взять")
    parser.add_argument("--per-card", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--catalog", type=Path, default=PLATFORM_ROOT / "catalog.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    catalog = pd.read_csv(args.catalog)
    # Карточки с общей картинкой исключены: их верный ответ неопределён, и учить на них — учить
    # модель угадывать монетку.
    pool = catalog[~catalog["shared_image"]].sample(frac=1.0, random_state=args.seed)
    picked = pool.head(args.n)
    suffixes = {".jpg", ".jpeg", ".png", ".heic"}
    shelves = sorted(p for p in SHELVES.iterdir() if p.suffix.lower() in suffixes)
    if not shelves:
        raise SystemExit(f"нет кадров полок в {SHELVES}")

    rows = []
    for row in tqdm(list(picked.itertuples(index=False)), desc="псевдофото"):
        bottle = cutout(Path(row.source_file))
        for k in range(args.per_card):
            shelf = load_image(rng.choice(shelves))
            image = synthesize(bottle, shelf, rng)
            target = args.out / row.slug / f"synthetic_{k + 1:02d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=95)
            rows.append({"path": str(target), "true_id": row.slug, "group": "synthetic"})
    manifest = args.out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"псевдофото: {len(rows)} -> {manifest}")


if __name__ == "__main__":
    main()
