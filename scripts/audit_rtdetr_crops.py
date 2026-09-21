"""Export bottle/label crops for every image, transfer GT, record production selection metrics.

Run from repo root. Originals and annotations are never modified. Resume requires identical
model hashes/configuration. Old label crops are audited separately from full-frame annotations.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wine_scanner.detect import RTDetrDetector, bottle_detector
from wine_scanner.detect.audit import annotations, iou, nfc, transfer_box
from wine_scanner.embed import load_image

SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_relative(path: Path, root: Path) -> Path:
    """Short deterministic export name; source remains recorded in records.jsonl."""
    relative = path.relative_to(root)
    source_key = nfc(relative.as_posix()).encode("utf-8")
    filename = hashlib.sha256(source_key).hexdigest()[:20] + ".jpg"
    return relative.parent / filename


def group(path):
    parts = path.parts
    if "frames_annotated" in parts:
        return "/".join(parts[1:3])
    if "wine-labels" in parts:
        return "roboflow/" + parts[parts.index("wine-labels") + 1]
    if "labels" in parts:
        return "/".join(parts[1:3]) + "_legacy_crops"
    return parts[1]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    summary = {}
    for key, values in groups.items():
        valid = [r for r in values if "error" not in r]
        labelled = [r for r in valid if r.get("gt_labels")]
        ious = [r["selected_label_iou_frame"] for r in labelled]
        retention = [v for r in labelled for v in r["label_retention"]]
        output_retention = [
            max(r["label_crop_retention"], default=0)
            for r in labelled
            if "label_crop_retention" in r
        ]
        summary[key] = {
            "images": len(values),
            "errors": len(values) - len(valid),
            "bottle_fallbacks": sum(r["bottle_fallback"] for r in valid),
            "label_fallbacks": sum(r["label_fallback"] for r in valid),
            "annotated_images": len(labelled),
            "gt_labels": len(retention),
            "selected_mean_iou_frame": float(np.mean(ious)) if ious else None,
            "selected_iou_ge_50": float(np.mean(np.array(ious) >= 0.5)) if ious else None,
            "selected_iou_ge_75": float(np.mean(np.array(ious) >= 0.75)) if ious else None,
            "gt_retained_ge_99": float(np.mean(np.array(retention) >= 0.99)) if retention else None,
            "gt_lost": sum(v == 0 for v in retention),
            "gt_clipped": sum(0 < v < 0.99 for v in retention),
            "label_crop_any_gt_retained_ge_99": float(np.mean(np.array(output_retention) >= 0.99))
            if output_retention
            else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/derived/rtdetr_audit"))
    parser.add_argument("--weights", type=Path, default=Path("models/rtdetr_label"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reuse-bottles-from",
        type=Path,
        help="Reuse verified bottle boxes from an earlier audit, labels recomputed",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bottle = bottle_detector("rtdetr", device)
    label = RTDetrDetector(args.weights, target="label", device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "bottle": bottle.cache_tag,
        "label": label.cache_tag,
        "label_sha256": digest(args.weights / "model.safetensors"),
        "margin": bottle.margin,
        "device": str(device),
        "version": 1,
    }
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to a different configuration; use another --out")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    reused = {}
    if args.reuse_bottles_from is not None:
        old_config = json.loads((args.reuse_bottles_from / "config.json").read_text())
        if any(old_config[k] != config[k] for k in ("bottle", "margin")):
            raise ValueError("Bottle configuration differs from reuse source")
        reused = {
            r["source"]: r
            for r in (
                json.loads(s)
                for s in (args.reuse_bottles_from / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if "error" not in r
        }
    gt = annotations(args.root)
    paths = sorted(
        p
        for p in args.root.rglob("*")
        if p.suffix.lower() in SUFFIXES
        and not any(s in {"derived", "__MACOSX"} or s.startswith("._") for s in p.parts)
    )
    # User's annotated frames first, then all other images including distractors.
    paths.sort(
        key=lambda p: (
            "frames_annotated" not in p.parts,
            nfc(p.as_posix()) not in gt,
            not any(
                part in {"catalog", "own", "train", "test", "synthetic", "eval", "live"}
                for part in p.relative_to(args.root).parts
            ),
            p.as_posix(),
        )
    )
    total_available_images = len(paths)
    if args.limit:
        paths = paths[: args.limit]
    journal = args.out / "records.jsonl"
    previous = (
        [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if journal.exists()
        else []
    )
    done = {r["source"] for r in previous}
    paths = [p for p in paths if p.as_posix() not in done]
    print(f"Pending {len(paths)}, already {len(done)}, annotated {len(gt)}", flush=True)
    status_path = args.out / "status.json"
    status_path.write_text(
        json.dumps({"status": "running", "expected": len(paths) + len(done),
                    "available_images": total_available_images, "limit": args.limit}),
        encoding="utf-8"
    )
    rows = list(previous)
    with journal.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(paths), args.batch_size):
            images, batch_paths = [], []
            for path in paths[offset : offset + args.batch_size]:
                try:
                    image = load_image(path)
                    record = gt.get(nfc(path.as_posix()))
                    if record and list(image.size) != record["size"]:
                        raise ValueError(
                            f"Annotation size {record['size']} != oriented image {image.size}"
                        )
                    images.append(image)
                    batch_paths.append(path)
                except Exception as error:
                    row = {"source": path.as_posix(), "group": group(path), "error": str(error)}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            started = time.perf_counter()
            from wine_scanner.detect import Box

            boxes = [None] * len(images)
            missing = []
            for j, path in enumerate(batch_paths):
                old = reused.get(path.as_posix())
                if old is None:
                    missing.append(j)
                    continue
                if digest(path) != old["source_sha256"]:
                    raise ValueError(f"Source image changed since bottle audit: {path}")
                boxes[j] = [Box(**box) for box in old["bottle_boxes"]]
            computed = bottle.detect_batch([images[j] for j in missing])
            for j, result in zip(missing, computed, strict=True):
                boxes[j] = result
            rects, crops, picked = [], [], []
            for image, candidates in zip(images, boxes, strict=True):
                selected = bottle.pick(candidates, image.size)
                rect = bottle.crop_rect(selected, image.size) if selected else (0, 0, *image.size)
                rects.append(rect)
                crops.append(image.crop(rect))
                picked.append(selected)
            labels = label.detect_batch(crops)
            elapsed = time.perf_counter() - started
            for path, image, crop, rect, candidates, selected, predictions in zip(
                batch_paths, images, crops, rects, boxes, picked, labels, strict=True
            ):
                relative = artifact_relative(path, args.root)
                output = args.out / "bottles" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                crop.save(output, quality=95)
                best = label.pick(predictions, crop.size)
                label_rect = label.crop_rect(best, crop.size) if best else (0, 0, *crop.size)
                label_rect_frame = [
                    label_rect[0] + rect[0],
                    label_rect[1] + rect[1],
                    label_rect[2] + rect[0],
                    label_rect[3] + rect[1],
                ]
                label_out = args.out / "labels" / relative
                label_out.parent.mkdir(parents=True, exist_ok=True)
                crop.crop(label_rect).save(label_out, quality=95)
                record = gt.get(nfc(path.as_posix()), {})
                gt_boxes = record.get("label", [])
                transferred = [transfer_box(box, rect) for box in gt_boxes]
                prediction_frame = (
                    [best.x1 + rect[0], best.y1 + rect[1], best.x2 + rect[0], best.y2 + rect[1]]
                    if best
                    else None
                )
                row = {
                    "source": path.as_posix(),
                    "source_sha256": digest(path),
                    "group": group(path),
                    "size": list(image.size),
                    "crop_rect": rect,
                    "bottle_crop": output.as_posix(),
                    "label_crop": label_out.as_posix(),
                    "bottle_fallback": selected is None,
                    "label_fallback": best is None,
                    "bottle_boxes": [asdict(b) for b in candidates],
                    "label_boxes_crop": [asdict(b) for b in predictions],
                    "selected_label_frame": prediction_frame,
                    "gt_labels": gt_boxes,
                    "gt_bottles": record.get("bottle", []),
                    "transferred_labels": [box for box, _ in transferred],
                    "label_retention": [fraction for _, fraction in transferred],
                    "label_crop_rect_frame": label_rect_frame,
                    "label_crop_retention": [
                        transfer_box(b, label_rect_frame)[1] for b in gt_boxes
                    ],
                    "selected_label_iou_frame": max(
                        (iou(prediction_frame, b) for b in gt_boxes), default=0
                    )
                    if best
                    else 0,
                    "batch_inference_ms_per_image": elapsed * 1000 / max(len(images), 1),
                    "bottle_predictions_reused": path.as_posix() in reused,
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if offset % (args.batch_size * 10) == 0:
                print(f"Processed {len(rows)}; batch inference {elapsed:.2f}s", flush=True)
                (args.out / "summary.json").write_text(
                    json.dumps(summarize(rows), indent=2), encoding="utf-8"
                )
    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    # Separate COCO datasets for each original annotation directory: split boundaries preserved.
    exports = defaultdict(
        lambda: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "label"}]}
    )
    for row in rows:
        if "error" in row or nfc(row["source"]) not in gt:
            continue
        dataset = exports[str(Path(row["source"]).parent)]
        image_id = len(dataset["images"])
        rect = row["crop_rect"]
        dataset["images"].append(
            {
                "id": image_id,
                "file_name": Path(row["bottle_crop"]).name,
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
                "source": row["source"],
                "source_sha256": row["source_sha256"],
                "crop_rect": rect,
            }
        )
        for box, retention in zip(row["transferred_labels"], row["label_retention"], strict=True):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            dataset["annotations"].append(
                {
                    "id": len(dataset["annotations"]),
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "retained_fraction": retention,
                }
            )
    for source, dataset in exports.items():
        target = args.out / "bottles" / Path(source).relative_to(args.root)
        target = target / "_annotations.coco.json"
        target.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    missing = sorted(set(gt) - {nfc(r["source"]) for r in rows})
    (args.out / "missing_annotations.json").write_text(
        json.dumps(missing, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summarize(rows), indent=2), flush=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "processed": len(rows),
                "errors": sum("error" in row for row in rows),
                "available_images": total_available_images,
                "all_images_processed": len({r["source"] for r in rows}) == total_available_images,
                "limit": args.limit,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
