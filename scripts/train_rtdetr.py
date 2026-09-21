"""Train RT-DETR on label annotations; select checkpoints on validation wines only.

Use scripts/audit_rtdetr_crops.py first to export annotations in bottle-crop coordinates.
Test data is evaluated only with --eval-only, never after each training epoch.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from wine_scanner.catalog import is_holdout
from wine_scanner.detect import Box, BoxCropper, CocoDetectionDataset, split_by_wine

PRETRAINED = "PekingU/rtdetr_r18vd"
LABEL_ID = 0
INPUT_SIZE = 640


def load_processor(source: str):
    from transformers import RTDetrImageProcessor

    # Разметка приходит в пикселях, resize 640×640 без сохранения пропорций — как учили RT-DETR.
    return RTDetrImageProcessor.from_pretrained(
        source, size={"height": INPUT_SIZE, "width": INPUT_SIZE}, do_pad=False
    )


def load_model(source: str):
    from transformers import RTDetrForObjectDetection

    return RTDetrForObjectDetection.from_pretrained(
        source,
        num_labels=1,
        id2label={LABEL_ID: "label"},
        label2id={"label": LABEL_ID},
        ignore_mismatched_sizes=True,
    )


def collate_for(processor):
    """Датасет отдаёт (тензор, разметка torchvision); процессор ждёт картинки и COCO-словари."""
    from torchvision.transforms import functional as TF

    def collate(batch):
        images, annotations, sizes = [], [], []
        for tensor, target in batch:
            image = TF.to_pil_image(tensor)
            sizes.append((image.height, image.width))
            boxes = target["boxes"].tolist()
            annotations.append(
                {
                    "image_id": int(target["image_id"][0]),
                    "annotations": [
                        {
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "category_id": LABEL_ID,
                            "area": (x1 - x0) * (y1 - y0),
                            "iscrowd": 0,
                        }
                        for x0, y0, x1, y1 in boxes
                    ],
                }
            )
            images.append(image)
        encoded = processor(images=images, annotations=annotations, return_tensors="pt")
        encoded["orig_sizes"] = torch.tensor(sizes)
        encoded["gt_boxes"] = [target["boxes"] for _, target in batch]
        return encoded

    return collate


def evaluate(model, processor, loader, device) -> dict:
    """IoU production-selected box at threshold 0.3; missing predictions count as zero."""
    model.eval()
    ious, found, total = [], 0, 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="валидация", leave=False):
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            results = processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=batch["orig_sizes"]
            )
            for result, gt, size in zip(
                results, batch["gt_boxes"], batch["orig_sizes"], strict=True
            ):
                total += 1
                if len(result["boxes"]) == 0:
                    ious.append(0.0)
                    continue
                found += 1
                boxes = [
                    Box(*b.tolist(), score=float(score))
                    for b, score in zip(result["boxes"], result["scores"], strict=True)
                ]
                selected = BoxCropper().pick(boxes, (int(size[1]), int(size[0])))
                best = torch.tensor([[selected.x1, selected.y1, selected.x2, selected.y2]])
                ious.append(float(box_iou(best, gt).max()))
    return {
        "detection_rate": found / max(total, 1),
        "mean_iou": sum(ious) / max(len(ious), 1),
        "iou@0.75": sum(1 for v in ious if v >= 0.75) / max(len(ious), 1),
    }


def report(title: str, metrics: dict) -> None:
    print(
        f"{title}: IoU {metrics['mean_iou']:.3f}, IoU>=0.75 {metrics['iou@0.75']:.3f}, "
        f"найдено {metrics['detection_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Roboflow wine-labels (train/, valid/)"
    )
    parser.add_argument("--out", type=Path, default=Path("models/rtdetr_label"))
    parser.add_argument(
        "--init", default=PRETRAINED, help="стартовые веса: имя на HF или своя папка"
    )
    parser.add_argument(
        "--own", type=Path, default=None, help="своя разметка для обучения (data/train/labels)"
    )
    parser.add_argument(
        "--test", type=Path, default=None, help="изолированный тест (data/test/labels)"
    )
    parser.add_argument("--own-repeat", type=int, default=20)
    parser.add_argument("--own-valid-share", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="ранняя остановка: столько эпох подряд без роста IoU на отборочной валидации "
        "(своей, если есть) — и обучение прекращается; 0 — выключить",
    )
    parser.add_argument(
        "--keep-all", action="store_true", help="сохранять модель каждой эпохи в <out>/epoch-NN/"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="взять N кадров, для проверки кода")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if is_holdout(args.data) or (args.own is not None and is_holdout(args.own)):
        raise SystemExit("Holdout data cannot be used for training/validation selection")
    if args.test is not None and not args.eval_only:
        raise SystemExit("Use --test only with --eval-only after selecting the checkpoint")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    processor = load_processor(args.init)
    model = load_model(args.init).to(device)
    collate = collate_for(processor)

    train_set = CocoDetectionDataset(args.data / "train")
    valid_set = CocoDetectionDataset(args.data / "valid")
    if args.limit:
        train_set.ids = train_set.ids[: args.limit]
        valid_set.ids = valid_set.ids[: max(4, args.limit // 4)]

    own_train = own_valid = None
    if args.own is not None:
        if is_holdout(args.own):
            raise SystemExit(
                f"{args.own} — изолированный тестовый набор, учить на нём нельзя; "
                "передайте его как --test, для обучения размечайте data/train/labels"
            )
        own_train, own_valid = split_by_wine(args.own, args.own_valid_share)
        print(
            f"своя разметка: train {len(own_train)}, valid {len(own_valid)} "
            "кадров (деление по винам)"
        )
    test_set = CocoDetectionDataset(args.test) if args.test is not None else None
    if test_set is not None:
        if args.limit:
            test_set.ids = test_set.ids[: max(4, args.limit // 4)]
        print(f"тестовый набор: {len(test_set)} кадров из {args.test} — только замер")
    print(
        f"train: {len(train_set)}, valid: {len(valid_set)}, "
        f"устройство: {device}, старт: {args.init}"
    )

    def loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0
        )

    train_source = train_set
    if own_train is not None and len(own_train):
        train_source = ConcatDataset([train_set] + [own_train] * max(1, args.own_repeat))
    train_loader = loader(train_source, True)
    valid_loader = loader(valid_set, False)
    own_loader = loader(own_valid, False) if own_valid is not None and len(own_valid) else None
    test_loader = loader(test_set, False) if test_set is not None and len(test_set) else None

    if args.eval_only:
        report("чужая валидация", evaluate(model, processor, valid_loader, device))
        if own_loader is not None:
            report("свои кадры (valid)", evaluate(model, processor, own_loader, device))
        if test_loader is not None:
            report("ТЕСТ", evaluate(model, processor, test_loader, device))
        return

    # Бэкбон уже видел COCO — ему шаг меньше, чем декодеру и голове.
    backbone_params = [
        p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=amp_enabled and args.precision == "fp16")

    best_iou, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running, skipped = time.time(), 0.0, 0
        for batch in tqdm(train_loader, desc=f"эпоха {epoch}", leave=False):
            labels = [{k: v.to(device) for k, v in item.items()} for item in batch["labels"]]
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(pixel_values=batch["pixel_values"].to(device), labels=labels)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            if not torch.isfinite(grad_norm) and not scaler.is_enabled():
                raise RuntimeError(f"Non-finite gradient in epoch {epoch}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < old_scale
            running += float(outputs.loss.detach())
        scheduler.step()

        metrics = evaluate(model, processor, valid_loader, device)
        print(
            f"эпоха {epoch}: loss {running / max(len(train_loader), 1):.3f}, "
            f"IoU {metrics['mean_iou']:.3f}, IoU>=0.75 {metrics['iou@0.75']:.3f}, "
            f"найдено {metrics['detection_rate']:.3f}, {time.time() - started:.0f} с"
        )
        external_metrics = dict(metrics)
        # Checkpoint selection uses validation wines only.
        if own_loader is not None:
            metrics = evaluate(model, processor, own_loader, device)
            report("  свои кадры (valid)", metrics)
        history.append(
            {
                "epoch": epoch,
                **metrics,
                "external_valid": external_metrics,
                "loss": running / max(len(train_loader), 1),
                "skipped_steps": skipped,
                "gradient_scale": scaler.get_scale(),
            }
        )
        print(f"  skipped optimizer steps: {skipped}, scale: {scaler.get_scale()}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        def save(where: Path, saved_epoch: int, selected_epoch: int) -> None:
            where.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(where)
            processor.save_pretrained(where)
            (where / "training.json").write_text(
                json.dumps(
                    {
                        "init": args.init,
                        "seed": args.seed,
                        "precision": args.precision,
                        "hyperparameters": {
                            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                        },
                        "epoch": saved_epoch,
                        "best_epoch": selected_epoch,
                        "data": str(args.data),
                        "own": str(args.own) if args.own else None,
                        "evaluation": "production-pick-threshold-0.3",
                        "own_train_files": [own_train.images[i]["file_name"] for i in own_train.ids]
                        if own_train
                        else [],
                        "own_valid_files": [own_valid.images[i]["file_name"] for i in own_valid.ids]
                        if own_valid
                        else [],
                        "history": history,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if args.keep_all:
            save(args.out / f"epoch-{epoch:02d}", epoch, best_epoch)
        if metrics["mean_iou"] > best_iou:
            best_iou, best_epoch, stale = metrics["mean_iou"], epoch, 0
            save(args.out, epoch, best_epoch)
            print(f"  сохранено: {args.out} (IoU {best_iou:.3f})")
        else:
            stale += 1
            if args.patience and stale >= args.patience and epoch >= args.min_epochs:
                print(f"  ранняя остановка: {stale} эпох без роста, лучшая — {best_epoch}")
                break

    print(f"\nлучший IoU на отборочной валидации: {best_iou:.3f} (эпоха {best_epoch})")


if __name__ == "__main__":
    main()
