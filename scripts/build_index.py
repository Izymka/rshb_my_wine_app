"""Прогнать каталог через DINOv2 и сложить векторы в FAISS.

    uv run python scripts/build_index.py --out models/index_xwines
"""

import argparse
from pathlib import Path

from wine_scanner.catalog import load_xwines
from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder
from wine_scanner.index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/index_xwines"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="взять первые N карточек, для проверки"
    )
    args = parser.parse_args()

    items = load_xwines()
    if args.limit:
        items = items[: args.limit]
    print(f"карточек в каталоге: {len(items)}")

    embedder = Dinov2Embedder(model_name=args.model)
    print(f"модель: {args.model}, устройство: {embedder.device}, размерность: {embedder.dim}")

    vectors = embedder.encode_paths([it.image_path for it in items], batch_size=args.batch_size)

    index = VectorIndex(embedder.dim)
    index.add(
        vectors.numpy(),
        item_ids=[it.item_id for it in items],
        payloads=[{**it.payload, "image_path": str(it.image_path)} for it in items],
    )
    index.save(args.out)
    print(f"индекс сохранён: {args.out} ({index.index.ntotal} векторов)")


if __name__ == "__main__":
    main()
