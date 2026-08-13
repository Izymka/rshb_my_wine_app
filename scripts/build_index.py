"""Прогнать каталог через DINOv2 и сложить векторы в FAISS.

    uv run python scripts/build_index.py --catalog own+xwines --detect cascade --fit pad

Индекс, который поднимает сервис, обязан быть построен ровно той же конфигурацией, на которой
мерились метрики: та же обрезка, тот же способ приведения к квадрату, та же модель. Расхождение
здесь ничего не сломает явно — поиск просто станет хуже, и списать это будет не на что.
Поэтому конфигурация сохраняется рядом с векторами, а пайплайн при загрузке её проверяет.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import CatalogItem, load_own, load_xwines
from wine_scanner.detect import BottleDetector, CachedCropper, CascadeCropper
from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder, load_image, pick_device
from wine_scanner.index import VectorIndex
from wine_scanner.ocr import LabelOCR
from wine_scanner.rerank import DescriptorStore, XFeatMatcher
from wine_scanner.vintage import catalog_years, reference_region

CROP_CACHE = Path("models/crop_cache")


def load_catalog(name: str) -> list[CatalogItem]:
    """Собрать каталог. own — свои вина, xwines — открытый набор как отвлекающие карточки."""
    items: list[CatalogItem] = []
    if "own" in name:
        items += load_own()[0]
    if "xwines" in name:
        items += load_xwines()
    if not items:
        raise ValueError(f"пустой каталог для --catalog {name}")
    return items


def build_cropper(detect: str | None, weights: Path, device) -> object | None:
    if detect is None:
        return None
    if detect == "cascade":
        detector = CascadeCropper(
            BottleDetector(device=device, mode="bottle"),
            BottleDetector(device=device, weights_path=weights),
        )
    elif detect == "trained":
        detector = BottleDetector(device=device, weights_path=weights)
    else:
        detector = BottleDetector(device=device, mode=detect)
    return CachedCropper(detector, CROP_CACHE)


def year_boxes(items: list[CatalogItem], cropper) -> dict[str, tuple]:
    """Найти на каждой карточке участок с годом.

    Считается здесь, а не на запросе, по той же причине, что и дескрипторы XFeat: это свойство
    каталога, а не снимка. На запросе это был бы лишний вызов распознавания по каждому
    кандидату — те самые секунды, ради которых переписывался весь Э11.

    Год находится не у всех карточек, и это нормально: у вина без винтажа его на этикетке нет,
    а у части снимков он не читается. Такие карточки просто не получают поля, и увеличение
    по ним не работает — год для них берётся из общего текста этикетки или не берётся вовсе.
    """
    ocr = LabelOCR()
    found: dict[str, tuple] = {}
    for item in tqdm(items, desc="год на карточках"):
        image = load_image(item.image_path)
        crop = cropper(item.image_path, image) if cropper else image
        region = reference_region(ocr.read(crop, use_cache=True), catalog_years(item.payload))
        if region is not None:
            found[item.item_id] = region.box
    print(f"карточек с найденным годом: {len(found)} из {len(items)}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/index"))
    parser.add_argument("--catalog", default="own+xwines", choices=["own", "xwines", "own+xwines"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--detect", choices=["bottle", "label", "trained", "cascade"], default="cascade"
    )
    parser.add_argument("--fit", default="pad", choices=["center_crop", "squash", "pad"])
    parser.add_argument("--weights", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument(
        "--limit", type=int, default=None, help="взять первые N карточек, для проверки"
    )
    parser.add_argument(
        "--no-descriptors",
        action="store_true",
        help="не считать локальные признаки XFeat (индекс без ре-ранкинга)",
    )
    parser.add_argument(
        "--no-vintage-boxes",
        action="store_true",
        help="не искать год на карточках (сборка быстрее, увеличение по гомографии отключено)",
    )
    parser.add_argument("--rerank-max-side", type=int, default=640)
    parser.add_argument("--rerank-points", type=int, default=2048)
    args = parser.parse_args()

    items = load_catalog(args.catalog)
    if args.limit:
        items = items[: args.limit]
    print(f"карточек в каталоге: {len(items)}")

    device = pick_device()
    embedder = Dinov2Embedder(
        model_name=args.model,
        device=device,
        cropper=build_cropper(args.detect, args.weights, device),
        fit=args.fit,
    )
    print(f"модель: {args.model}, устройство: {device}, размерность: {embedder.dim}")

    vectors = embedder.encode_paths([it.image_path for it in items], batch_size=args.batch_size)

    boxes = year_boxes(items, embedder.cropper) if not args.no_vintage_boxes else {}

    index = VectorIndex(embedder.dim)
    index.add(
        vectors.numpy(),
        item_ids=[it.item_id for it in items],
        # image_path нужен не для показа, а для ре-ранкинга: XFeat сравнивает запрос с самой
        # картинкой кандидата, поэтому путь обязан пережить сборку индекса.
        # vintage_box — то же самое для Э8: где на карточке напечатан год.
        payloads=[
            {
                **it.payload,
                "image_path": str(it.image_path),
                **({"vintage_box": list(boxes[it.item_id])} if it.item_id in boxes else {}),
            }
            for it in items
        ],
    )
    index.save(args.out)

    # Локальные признаки карточек — тоже часть индекса. Считать их на запросе означает платить
    # за каждого кандидата открытием файла и прогоном каскада детекторов: замер сквозного
    # запроса показал на этом 10–15 секунд.
    if not args.no_descriptors:
        store = DescriptorStore(args.out / "descriptors")
        matcher = XFeatMatcher(max_side=args.rerank_max_side, top_k=args.rerank_points)
        for item in tqdm(items, desc="дескрипторы"):
            if store.exists(item.item_id):
                continue
            image = load_image(item.image_path)
            crop = embedder.cropper(item.image_path, image) if embedder.cropper else image
            store.save(item.item_id, matcher.describe(crop))
        print(f"дескрипторов: {len(store)}")

    (args.out / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "detect": args.detect,
                "fit": args.fit,
                "weights": str(args.weights),
                "catalog": args.catalog,
                "items": len(items),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"индекс сохранён: {args.out} ({index.index.ntotal} векторов)")


if __name__ == "__main__":
    main()
