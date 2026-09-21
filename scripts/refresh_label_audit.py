"""Re-evaluate a label checkpoint on unchanged bottle crops, preserving frame transforms."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_rtdetr_crops import artifact_relative, digest, summarize

from wine_scanner.detect import RTDetrDetector
from wine_scanner.detect.audit import iou, transfer_box
from wine_scanner.embed import load_image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/derived/rtdetr_audit"))
    parser.add_argument("--out", type=Path, default=Path("data/derived/rtdetr_retrained_audit"))
    parser.add_argument("--weights", type=Path, default=Path("models/rtdetr_label_recrop"))
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = RTDetrDetector(args.weights, target="label", device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "source_audit": str(args.source),
        "source_config": json.loads((args.source / "config.json").read_text()),
        "label_sha256": digest(args.weights / "model.safetensors"),
        "label": detector.cache_tag,
    }
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Checkpoint/configuration changed; choose another output directory")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    journal = args.out / "records.jsonl"
    rows = (
        [json.loads(s) for s in journal.read_text(encoding="utf-8").splitlines()]
        if journal.exists()
        else []
    )
    done = {r["source"] for r in rows}
    pending = [
        json.loads(s)
        for s in (args.source / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pending = [r for r in pending if r["source"] not in done and "error" not in r]
    pending.sort(key=lambda r: (not bool(r["gt_labels"]), r["source"]))
    print(f"Pending {len(pending)}, completed {len(done)}", flush=True)
    with journal.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            images = [load_image(r["bottle_crop"]) for r in batch]
            predictions = detector.detect_batch(images)
            for original, image, boxes in zip(batch, images, predictions, strict=True):
                row = dict(original)
                best = detector.pick(boxes, image.size)
                label_rect = detector.crop_rect(best, image.size) if best else (0, 0, *image.size)
                relative = artifact_relative(Path(original["source"]), Path("data"))
                target = args.out / "labels" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                image.crop(label_rect).save(target, quality=95)
                rect = row["crop_rect"]
                output_rect = [
                    label_rect[0] + rect[0],
                    label_rect[1] + rect[1],
                    label_rect[2] + rect[0],
                    label_rect[3] + rect[1],
                ]
                frame_box = (
                    [best.x1 + rect[0], best.y1 + rect[1], best.x2 + rect[0], best.y2 + rect[1]]
                    if best
                    else None
                )
                row.update(
                    label_crop=target.as_posix(),
                    label_fallback=best is None,
                    label_boxes_crop=[asdict(b) for b in boxes],
                    selected_label_frame=frame_box,
                    label_crop_rect_frame=output_rect,
                    label_crop_retention=[
                        transfer_box(b, output_rect)[1] for b in row["gt_labels"]
                    ],
                    selected_label_iou_frame=max(
                        (iou(frame_box, gt) for gt in row["gt_labels"]), default=0
                    )
                    if best
                    else 0,
                )
                # Original combined inference timing does not describe the refreshed model.
                row.pop("batch_inference_ms_per_image", None)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if offset % (args.batch_size * 10) == 0:
                print(f"Processed {len(rows)}", flush=True)
                (args.out / "summary.json").write_text(
                    json.dumps(summarize(rows), indent=2), encoding="utf-8"
                )
    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
