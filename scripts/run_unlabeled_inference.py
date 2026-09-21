"""Run the production scanner on an unlabeled organizer set.

The official organizer archive contains images but deliberately no ground-truth slugs.  This
script therefore never calls its diagnostic output an accuracy.  It writes one row per image
with the returned item_id, five candidate cards and enough signals to inspect uncertainty.
If organizers later provide a separate ``image_path,slug`` CSV, pass it via ``--labels`` to
calculate top-1 without changing the model or its thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

from tqdm import tqdm

from wine_scanner.decide.guard import DEFAULT_MODE, GUARD_MODES
from wine_scanner.embed import load_image
from wine_scanner.pipeline import WineScanner

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        expected = {"image_path", "slug"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path}: required CSV columns are image_path,slug")
        labels = {str(row["image_path"]): str(row["slug"]) for row in reader if row["slug"]}
    return labels


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("models/index_platform"))
    parser.add_argument("--decider", type=Path, default=Path("models/decider_platform_paddle"))
    parser.add_argument("--guard", choices=GUARD_MODES, default=DEFAULT_MODE)
    parser.add_argument("--labels", type=Path, help="optional image_path,slug CSV from organizers")
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help=(
            "return the best catalogue candidate below the model threshold; "
            "use only for closed sets"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true", help="measure cold-path OCR/crops")
    args = parser.parse_args()

    images = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No supported images found in {args.images}")
    labels = load_labels(args.labels)
    scanner = WineScanner(
        index_dir=args.index,
        decider_dir=args.decider,
        crop_cache=None if args.no_cache else Path("models/crop_cache"),
        ocr_cache=not args.no_cache,
        guard=args.guard,
    )

    rows: list[dict] = []
    started = time.perf_counter()
    for image_path in tqdm(images, desc="organizer images"):
        relative = image_path.relative_to(args.images).as_posix()
        result = scanner.identify(
            load_image(image_path), image_key=None if args.no_cache else str(image_path), trace=True
        )
        best = result.best
        candidates = [
            {
                "item_id": candidate.item_id,
                "probability": round(candidate.probability, 6),
                "name": candidate.card.get("name"),
                "winery": candidate.card.get("winery"),
            }
            for candidate in result.candidates[:5]
        ]
        predicted_slug = best.item_id if best and (result.answered or args.force_answer) else None
        true_slug = labels.get(relative) or labels.get(image_path.name)
        rows.append(
            {
                "query_id": f"organizer-{len(rows) + 1:06d}",
                "image_path": relative,
                "path": str(image_path),
                "image_sha256": sha256(image_path),
                "predicted_slug": predicted_slug,
                "best_id": best.item_id if best else None,
                "true_slug": true_slug,
                "top1_correct": predicted_slug == true_slug if true_slug else None,
                "answered": result.answered,
                "force_answer": args.force_answer,
                "probability": round(best.probability, 6) if best else 0.0,
                "margin": round((result.confidence or {}).get("margin", 0.0), 6),
                "guard": result.guard,
                "recognized_text": result.text,
                "ocr_lines": len((result.trace or {}).get("ocr_lines", [])),
                "top5": candidates,
                "timings_ms": {
                    key: round(value * 1000, 1) for key, value in result.timings.items()
                },
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    answered = [row for row in rows if row["answered"]]
    margins = [row["margin"] for row in rows]
    total_ms = [row["timings_ms"].get("total", 0.0) for row in rows]
    summary = {
        "dataset": str(args.images),
        "n_images": len(rows),
        "ground_truth_labels": len(labels),
        "answered": len(answered),
        "refused": len(rows) - len(answered),
        "twin_warnings": sum(bool(row["guard"] and "twin" in row["guard"]) for row in rows),
        "ocr_empty": sum(not row["recognized_text"] for row in rows),
        "median_margin": statistics.median(margins),
        "p10_margin": percentile(margins, 0.1),
        "latency_ms": {"p50": percentile(total_ms, 0.5), "p95": percentile(total_ms, 0.95)},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "index": str(args.index),
        "decider": str(args.decider),
        "guard": args.guard,
        "force_answer": args.force_answer,
        "accuracy": None,
    }
    labeled = [row for row in rows if row["true_slug"]]
    if labeled:
        correct = sum(row["top1_correct"] for row in labeled)
        summary["accuracy"] = {
            "correct": correct,
            "total": len(labeled),
            "top1": correct / len(labeled),
        }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
