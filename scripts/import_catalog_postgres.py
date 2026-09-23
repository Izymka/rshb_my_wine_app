"""Idempotently import the legacy platform catalogue into PostgreSQL.

Run: uv run python scripts/import_catalog_postgres.py
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from wine_scanner.db import Base, GrapeVariety, Wine, WineImage, database_url, session_factory


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/catalog/originals"))
    parser.add_argument("--media", type=Path, default=Path("data/media/originals"))
    parser.add_argument(
        "--copy", action="store_true", help="copy legacy PNGs into the canonical media tree"
    )
    args = parser.parse_args()
    factory = session_factory(database_url())
    Base.metadata.create_all(factory.kw["bind"])
    table = pd.read_csv(args.catalog, dtype={"vintage": "Int64"})
    with factory.begin() as session:
        for row in table.itertuples(index=False):
            source = args.images / f"{row.slug}.png"
            if not source.exists():
                continue
            image_hash = digest(source)
            target = args.media / f"{image_hash}.png"
            if args.copy and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            storage_path = target if args.copy else source
            wine = session.query(Wine).filter_by(slug=row.slug).one_or_none()
            if wine is None:
                wine = Wine(slug=row.slug, name=row.name)
                session.add(wine)
            wine.name = row.name
            wine.producer = str(row.winery or "")
            wine.region = str(row.region or "")
            wine.vintage = int(row.vintage) if pd.notna(row.vintage) else None
            grapes = [part.strip() for part in str(row.grapes or "").split(",") if part.strip()]
            wine.grapes = []
            for name in grapes:
                grape = session.query(GrapeVariety).filter_by(name=name).one_or_none()
                if grape is None:
                    grape = GrapeVariety(name=name)
                    session.add(grape)
                wine.grapes.append(grape)
            with Image.open(source) as opened:
                width, height = opened.size
            record = (
                session.query(WineImage).filter_by(storage_path=str(storage_path)).one_or_none()
            )
            if record is None:
                record = WineImage(
                    wine=wine,
                    storage_path=str(storage_path),
                    sha256=image_hash,
                    width=width,
                    height=height,
                )
                session.add(record)
    print(f"imported catalogue rows from {args.catalog}")


if __name__ == "__main__":
    main()
