"""Shared wine-level validation selection for preprocessing experiments."""

import json
import random
from pathlib import Path

from wine_scanner.catalog import load_manifest


def own_validation_queries(ids, detector_meta: Path | None = None):
    queries = [
        q
        for q in load_manifest(Path("data/train/manifest.csv"))
        if q.source == "own" and q.wine_id in ids
    ]
    if detector_meta is not None:
        meta = json.loads(detector_meta.read_text(encoding="utf-8"))
        validation = {name.split("__")[0] for name in meta["own_valid_files"]}
        validation &= {q.wine_id for q in queries}
    else:
        wines = sorted({q.wine_id for q in queries})
        random.Random(2026).shuffle(wines)
        validation = set(wines[: max(1, round(len(wines) * 0.25))])
    if not validation:
        raise ValueError("No held-out known wines for the preprocessing experiment")
    return [q for q in queries if q.wine_id in validation], validation
