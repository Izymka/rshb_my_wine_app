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
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wine_scanner.catalog import PLATFORM_ROOT
from wine_scanner.embed import load_image
from wine_scanner.synthetic import SHELVES, cutout, shelf_paths, synthesize

OUT_ROOT = Path("data/synthetic")


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
    shelves = shelf_paths()
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
