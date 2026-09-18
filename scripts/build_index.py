"""Прогнать каталог через DINOv2 и сложить векторы в FAISS.

    uv run python scripts/build_index.py --catalog platform --detect cascade --fit pad

Индекс, который поднимает сервис, обязан быть построен ровно той же конфигурацией, на которой
мерились метрики: та же обрезка, тот же способ приведения к квадрату, та же модель. Расхождение
здесь ничего не сломает явно — поиск просто станет хуже, и списать это будет не на что.
Поэтому конфигурация сохраняется рядом с векторами, а пайплайн при загрузке её проверяет.
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import CatalogItem, load_own, load_platform, load_xwines
from wine_scanner.detect import COCO_BOTTLE_MODEL, CachedCropper, build_cropper, detector_kind
from wine_scanner.embed import DEFAULT_MODEL, build_embedder, load_image, pick_device
from wine_scanner.index import VectorIndex
from wine_scanner.ocr import LabelOCR
from wine_scanner.rerank import DescriptorStore, XFeatMatcher
from wine_scanner.vintage import catalog_years, reference_region

CROP_CACHE = Path("models/crop_cache")


def load_catalog(name: str) -> list[CatalogItem]:
    """Собрать каталог.

    platform — каталог «Своего Вина» из data/catalog (это боевой вариант), own — свои вина,
    xwines — открытый набор как отвлекающие карточки. Первый со вторыми не смешивается:
    у платформы ключ slug, и подмешивать к нему чужие карточки незачем — у неё свои близнецы.
    """
    items: list[CatalogItem] = []
    if name == "platform":
        return load_platform()
    if "own" in name:
        items += load_own()[0]
    if "xwines" in name:
        items += load_xwines()
    if not items:
        raise ValueError(f"пустой каталог для --catalog {name}")
    return items


def year_boxes(items: list[CatalogItem], cropper) -> dict[str, tuple]:
    """Найти на каждой карточке участок с годом.

    Считается здесь, а не на запросе, по той же причине, что и дескрипторы XFeat: это свойство
    каталога, а не снимка. На запросе это был бы лишний вызов распознавания по каждому
    кандидату — те самые секунды, ради которых переписывался весь Э11.

    Год находится не у всех карточек, и это нормально: у вина без винтажа его на этикетке нет,
    а у части снимков он не читается. Такие карточки просто не получают поля, и увеличение
    по ним не работает — год для них берётся из общего текста этикетки или не берётся вовсе.
    """
    # Боксы года на карточках каталога не должны зависеть от WINE_OCR: облако их не считает
    # так же, и индекс перестал бы воспроизводиться. Каталог читает всегда EasyOCR.
    ocr = LabelOCR(backend="easyocr")
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
    parser.add_argument(
        "--catalog", default="platform", choices=["platform", "own", "xwines", "own+xwines"]
    )
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
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="взять дескрипторы XFeat и боксы года из уже собранного индекса (другой эмбеддер)",
    )
    parser.add_argument("--rerank-max-side", type=int, default=640)
    parser.add_argument("--rerank-points", type=int, default=2048)
    args = parser.parse_args()

    items = load_catalog(args.catalog)
    if args.limit:
        items = items[: args.limit]
    print(f"карточек в каталоге: {len(items)}")

    device = pick_device()
    embedder = build_embedder(
        model_name=args.model,
        device=device,
        cropper=(
            CachedCropper(build_cropper(args.detect, args.weights, device), CROP_CACHE)
            if args.detect
            else None
        ),
        fit=args.fit,
    )
    print(f"модель: {args.model}, устройство: {device}, размерность: {embedder.dim}")

    vectors = embedder.encode_paths([it.image_path for it in items], batch_size=args.batch_size)

    # Дескрипторы XFeat и боксы года от эмбеддера не зависят: при сборке индекса другой
    # моделью их можно взять из уже собранного — это полтора часа против минут.
    reused_boxes: dict[str, tuple] = {}
    if args.reuse_from is not None:
        old_meta = json.loads((args.reuse_from / "meta.json").read_text(encoding="utf-8"))
        for item_id, payload in zip(old_meta["item_ids"], old_meta["payloads"], strict=True):
            if payload.get("vintage_box"):
                reused_boxes[item_id] = tuple(payload["vintage_box"])
        old_descriptors = args.reuse_from / "descriptors"
        if old_descriptors.exists() and not args.no_descriptors:
            target = args.out / "descriptors"
            if not target.exists():
                args.out.mkdir(parents=True, exist_ok=True)
                shutil.copytree(old_descriptors, target)
            print(
                f"дескрипторы и боксы года взяты из {args.reuse_from}: боксов {len(reused_boxes)}"
            )

    if args.no_vintage_boxes:
        boxes = {}
    elif reused_boxes:
        boxes = reused_boxes
    else:
        boxes = year_boxes(items, embedder.cropper)

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
                "detector": detector_kind(args.weights),
                "bottle_model": COCO_BOTTLE_MODEL if detector_kind(args.weights) == "rtdetr" else "torchvision-coco",
                "fit": args.fit,
                "weights": str(args.weights),
                "catalog": args.catalog,
                "items": len(items),
                "size": embedder.size,
                "descriptor": embedder.descriptor,
                "dim": embedder.dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"индекс сохранён: {args.out} ({index.index.ntotal} векторов)")


if __name__ == "__main__":
    main()
