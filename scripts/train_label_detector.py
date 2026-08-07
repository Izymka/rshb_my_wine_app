"""Дообучение детектора этикетки на один класс.

    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels"

Ждёт экспорт в формате COCO: подпапки train/ и valid/, в каждой картинки и _annotations.coco.json.

Дообучаем только голову классификатора поверх COCO-весов: бэкбон уже умеет находить объекты,
доучиваем «что считать целью». Поэтому хватает тысяч кадров, а не сотен тысяч, и одной
видеокарты на полчаса.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from wine_scanner.detect import CocoDetectionDataset, build_label_detector, collate


def evaluate(model, loader, device) -> dict:
    """Доля кадров с найденной этикеткой и средний IoU лучшей рамки.

    Полноценный COCO mAP здесь избыточен: нам не нужен рейтинг детекторов, нужно понять,
    попадаем ли мы в этикетку достаточно точно, чтобы по этой рамке резать кадр. IoU 0.75+
    для обрезки уже вполне рабочий.
    """
    model.eval()
    ious, found = [], 0
    total = 0

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="валидация", leave=False):
            outputs = model([img.to(device) for img in images])
            for output, target in zip(outputs, targets, strict=True):
                total += 1
                if len(output["boxes"]) == 0:
                    ious.append(0.0)
                    continue
                found += 1
                # Берём рамку с наибольшей уверенностью — так же, как будет на инференсе.
                best = output["boxes"][output["scores"].argmax()].unsqueeze(0).cpu()
                ious.append(float(box_iou(best, target["boxes"]).max()))

    return {
        "detection_rate": found / max(total, 1),
        "mean_iou": sum(ious) / max(len(ious), 1),
        "iou@0.75": sum(1 for v in ious if v >= 0.75) / max(len(ious), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument(
        "--backbone", default="mobilenet320", choices=["mobilenet320", "mobilenet", "resnet50"]
    )
    parser.add_argument("--min-size", type=int, default=None, help="сторона входа детектора")
    parser.add_argument(
        "--trainable-layers", type=int, default=1, help="сколько верхних блоков бэкбона учим"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="обучение детектора идёт на cpu: на MPS часть операций Faster R-CNN "
        "не реализована, и с PYTORCH_ENABLE_MPS_FALLBACK шаг выходит в 13 раз медленнее",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--limit", type=int, default=None, help="взять N кадров, для проверки кода")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_set = CocoDetectionDataset(args.data / "train")
    valid_set = CocoDetectionDataset(args.data / "valid")
    if args.limit:
        train_set.ids = train_set.ids[: args.limit]
        valid_set.ids = valid_set.ids[: max(8, args.limit // 8)]
    print(f"train: {len(train_set)}, valid: {len(valid_set)}, устройство: {device}")

    # num_workers=0: на macOS дочерние процессы плохо уживаются с MPS.
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    model = build_label_detector(
        args.backbone,
        trainable_backbone_layers=args.trainable_layers,
        min_size=args.min_size,
    ).to(device)

    # Горизонтальное отражение здесь не используем: на этикетке текст, зеркальных этикеток
    # в природе не бывает, и такая аугментация только уводит модель от реальности.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), 0.0

        for images, targets in tqdm(train_loader, desc=f"эпоха {epoch}/{args.epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # В режиме train детектор возвращает не предсказания, а словарь функций потерь:
            # отдельно за рамки, за классификацию, за RPN. Складываем их и оптимизируем сумму.
            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)

        scheduler.step()
        metrics = evaluate(model, valid_loader, device)
        print(
            f"эпоха {epoch}: loss {running / max(len(train_loader), 1):.3f}, "
            f"IoU {metrics['mean_iou']:.3f}, IoU>=0.75 {metrics['iou@0.75']:.3f}, "
            f"найдено {metrics['detection_rate']:.3f}, {time.time() - started:.0f} с"
        )

        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backbone": args.backbone,
                    "min_size": args.min_size,
                    "state_dict": model.state_dict(),
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out,
            )
            print(f"  сохранено в {args.out}")

    print(f"\nлучший mean IoU: {best_iou:.3f}")


if __name__ == "__main__":
    main()
