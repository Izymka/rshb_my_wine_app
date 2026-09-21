"""Geometry and annotation readers for an auditable, non-destructive crop export."""

import csv
import json
import unicodedata
from pathlib import Path


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace(":", "_")


def transfer_box(box, rect):
    """XYXY in frame -> clipped XYXY in crop, and fraction of GT area retained."""
    x1, y1, x2, y2 = box
    left, top, right, bottom = rect
    a, b, c, d = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    intersection = max(0, c - a) * max(0, d - b)
    return (
        [a - left, b - top, c - left, d - top] if intersection else None,
        intersection / area if area else 0.0,
    )


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def annotations(root: Path) -> dict:
    """Preserve label/bottle classes; never mistake a bottle for a label."""
    records = {}
    for path in sorted(root.rglob("_annotations*")):
        if "__MACOSX" in path.parts or "derived" in path.parts:
            continue
        rows = []
        if path.name == "_annotations.coco.json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            categories = {c["id"]: c["name"] for c in data["categories"]}
            images = {im["id"]: im for im in data["images"]}
            for im in images.values():
                records[nfc((path.parent / im["file_name"]).as_posix())] = {
                    "size": [im["width"], im["height"]],
                    "label": [],
                    "bottle": [],
                }
            for ann in data["annotations"]:
                im = images[ann["image_id"]]
                rows.append(
                    (
                        categories[ann["category_id"]],
                        *ann["bbox"],
                        im["file_name"],
                        im["width"],
                        im["height"],
                    )
                )
        elif path.name == "_annotations.csv":
            if (path.parent / "_annotations.coco.json").exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
        else:
            continue
        for row in rows:
            if len(row) < 8 or row[0] == "label_name":
                continue
            category, x, y, w, h, name, width, height = row[:8]
            category = {"wine-labels": "label"}.get(category.lower(), category.lower())
            if category not in {"label", "bottle"}:
                raise ValueError(f"Unknown category {category!r} in {path}")
            record = records.setdefault(
                nfc((path.parent / name).as_posix()),
                {"size": [int(float(width)), int(float(height))], "label": [], "bottle": []},
            )
            x, y, w, h = map(float, (x, y, w, h))
            if w > 0 and h > 0:
                record[category].append([x, y, x + w, y + h])
    return records
