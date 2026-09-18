"""Дообучение RT-DETR на один класс «этикетка» — соперник Faster R-CNN из каскада.

    uv run python scripts/train_rtdetr.py --data "data/third-party datasets/wine-labels" \\
        --own data/own_labels --test data/live_labels --device cuda --epochs 12 \\
        --out models/rtdetr_label
    uv run python scripts/train_rtdetr.py --data ... --test data/live_labels \\
        --init models/rtdetr_label --eval-only                 # измерить сохранённую модель
    uv run python scripts/train_rtdetr.py --data ... --limit 8 --epochs 1 --out /tmp/rtdetr-smoke
                                                               # проверка кода на процессоре

Зачем второй детектор. Faster R-CNN в каскаде режет края этикетки (замер 17.09 на своей
разметке: IoU 0.818 против 0.882 на чужой валидации). RT-DETR — детектор-трансформер без
якорей и без NMS, на COCO точнее и быстрее R-CNN того же размера; ТЗ хакатона скорость тоже
оценивает. Сравниваем по одной и той же метрике на одном и том же изолированном тесте
(`--test data/live_labels`): IoU лучшей рамки с разметкой и доля кадров с IoU ≥ 0.75 — ровно
то, что нужно для обрезки. Победитель встаёт в каскад вместо `label_detector.pt`.

Лицензия: реализация из HuggingFace transformers и веса `PekingU/rtdetr_r18vd` — Apache 2.0
(LICENSES.md). Ultralytics-версия RT-DETR не используется — AGPL.

Данные и правила те же, что у `train_label_detector.py`: Roboflow wine-labels как основа,
своя разметка `--own` (деление по винам, повтор `--own-repeat`), тест только измеряется
(`catalog.is_holdout` не даст передать его как `--own`). Обучение — на видеокарте: R18 на
640 px на процессоре идёт часами, `--limit` нужен только чтобы убедиться, что код проходит.
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
from wine_scanner.detect import CocoDetectionDataset, split_by_wine

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
    """Та же метрика, что у Faster R-CNN: IoU лучшей рамки и доля кадров с IoU ≥ 0.75."""
    model.eval()
    ious, found, total = [], 0, 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="валидация", leave=False):
            outputs = model(pixel_values=batch["pixel_values"].to(device))
            results = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=batch["orig_sizes"]
            )
            for result, gt in zip(results, batch["gt_boxes"], strict=True):
                total += 1
                if len(result["boxes"]) == 0:
                    ious.append(0.0)
                    continue
                found += 1
                best = result["boxes"][result["scores"].argmax()].unsqueeze(0).cpu()
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
    parser.add_argument("--data", type=Path, required=True, help="Roboflow wine-labels (train/, valid/)")
    parser.add_argument("--out", type=Path, default=Path("models/rtdetr_label"))
    parser.add_argument("--init", default=PRETRAINED, help="стартовые веса: имя на HF или своя папка")
    parser.add_argument("--own", type=Path, default=None, help="своя разметка для обучения (data/own_labels)")
    parser.add_argument("--test", type=Path, default=None, help="изолированный тест (data/live_labels)")
    parser.add_argument("--include-holdout", action="store_true", help="разрешить --own на тесте (нечестно)")
    parser.add_argument("--own-repeat", type=int, default=20)
    parser.add_argument("--own-valid-share", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=None, help="взять N кадров, для проверки кода")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
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
        if is_holdout(args.own) and not args.include_holdout:
            raise SystemExit(
                f"{args.own} — изолированный тестовый набор, учить на нём нельзя; "
                "передайте его как --test, для обучения размечайте data/own_labels"
            )
        own_train, own_valid = split_by_wine(args.own, args.own_valid_share)
        print(f"своя разметка: train {len(own_train)}, valid {len(own_valid)} кадров (деление по винам)")
    test_set = CocoDetectionDataset(args.test) if args.test is not None else None
    if test_set is not None:
        if args.limit:
            test_set.ids = test_set.ids[: max(4, args.limit // 4)]
        print(f"тестовый набор: {len(test_set)} кадров из {args.test} — только замер")
    print(f"train: {len(train_set)}, valid: {len(valid_set)}, устройство: {device}, старт: {args.init}")

    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)

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
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr}, {"params": other_params, "lr": args.lr}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_iou, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for batch in tqdm(train_loader, desc=f"эпоха {epoch}", leave=False):
            labels = [{k: v.to(device) for k, v in item.items()} for item in batch["labels"]]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(pixel_values=batch["pixel_values"].to(device), labels=labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            running += float(outputs.loss.detach())
        scheduler.step()

        metrics = evaluate(model, processor, valid_loader, device)
        print(
            f"эпоха {epoch}: loss {running / max(len(train_loader), 1):.3f}, "
            f"IoU {metrics['mean_iou']:.3f}, IoU>=0.75 {metrics['iou@0.75']:.3f}, "
            f"найдено {metrics['detection_rate']:.3f}, {time.time() - started:.0f} с"
        )
        # Отбор чекпойнта — по своей валидации, если она есть; тест только печатается.
        if own_loader is not None:
            metrics = evaluate(model, processor, own_loader, device)
            report("  свои кадры (valid)", metrics)
        if test_loader is not None:
            report("  ТЕСТ (не влияет на отбор)", evaluate(model, processor, test_loader, device))
        history.append({"epoch": epoch, **metrics})

        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            (args.out / "training.json").write_text(
                json.dumps({"init": args.init, "best_epoch": epoch, "history": history}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  сохранено: {args.out} (IoU {best_iou:.3f})")


if __name__ == "__main__":
    main()
