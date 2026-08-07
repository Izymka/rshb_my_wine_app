"""Найти вино по фотографии.

    uv run python scripts/search.py data/own/Photos-1-001/IMG_9840.HEIC
"""

import argparse
from pathlib import Path

from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder
from wine_scanner.index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("--index", type=Path, default=Path("models/index_xwines"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    index = VectorIndex.load(args.index)
    embedder = Dinov2Embedder(model_name=args.model)
    query = embedder.encode_one(args.photo).numpy()

    print(f"\nзапрос: {args.photo.name}\n")
    for rank, hit in enumerate(index.search(query, top_k=args.top_k), start=1):
        p = hit.payload
        print(f"{rank}. {hit.score:.3f}  {p['name']} — {p['winery']} ({p['country']}, {p['type']})")
        print(f"   {p['image_path']}")


if __name__ == "__main__":
    main()
