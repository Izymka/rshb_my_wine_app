"""Дообучение детектора этикетки на один класс.

    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels"
    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels" \
        --own data/train/labels --test data/test/labels --init models/label_detector.pt \
        --own-repeat 20 --epochs 4 --device cuda --out models/label_detector_v2.pt
    uv run python scripts/train_label_detector.py --data ... --test data/test/labels \
        --init models/label_detector.pt --eval-only   # измерить текущий детектор на тесте

Ждёт экспорт в формате COCO: подпапки train/ и valid/, в каждой картинки и _annotations.coco.json.

Своя разметка для обучения (`--own`) — одна папка с кадрами и `_annotations.csv` (makesense.ai)
или `_annotations.coco.json` (Roboflow); делится на train/valid **по винам** 80/20 с
фиксированным зерном и в обучении повторяется `--own-repeat` раз: своих кадров сотня против
пяти тысяч чужих, и без повторов детектор их не заметит. Чекпойнт отбирается по своей
валидации. `--init` — старт с уже дообученных весов, а не с COCO: так эпох нужно меньше.

Тестовый набор (`--test`, `data/test/labels` — вырезки кадров российских вин) изолирован:
измеряется целиком после каждой эпохи, но в обучение и отбор чекпойнта не входит, и передать
его как `--own` скрипт не даст (`catalog.is_holdout`). Для обучения размечаются вырезки своего
набора — `data/train/labels`, их делает `prepare_label_annotation.py --manifest data/train/manifest.csv`.

Сохраняется лучший по своей валидации чекпойнт (`--out`); `--patience N` — ранняя остановка
после N эпох без роста, `--keep-all` — чекпойнт каждой эпохи рядом (`<out>.epoch-NN.pt`).

Дообучаем только голову классификатора поверх COCO-весов: бэкбон уже умеет находить объекты,
доучиваем «что считать целью». Поэтому хватает тысяч кадров, а не сотен тысяч, и одной
видеокарты на полчаса.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from wine_scanner.catalog import is_holdout
from wine_scanner.detect import CocoDetectionDataset, build_label_detector, collate, split_by_wine


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


def report(title: str, metrics: dict) -> None:
    print(
        f"{title}: IoU {metrics['mean_iou']:.3f}, IoU>=0.75 {metrics['iou@0.75']:.3f}, "
        f"найдено {metrics['detection_rate']:.3f}"
    )


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
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="ранняя остановка: столько эпох подряд без роста IoU на отборочной валидации "
        "(своей, если есть) — и обучение прекращается; 0 — выключить",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="сохранять чекпойнт каждой эпохи рядом с лучшим (<out>.epoch-NN.pt)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--limit", type=int, default=None, help="взять N кадров, для проверки кода")
    parser.add_argument(
        "--own",
        type=Path,
        default=None,
        help="своя разметка для обучения (одна папка, CSV makesense или COCO); "
        "data/train/labels — вырезки своего набора",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="изолированный тестовый набор (data/test/labels): только измеряется, целиком, "
        "в обучение и отбор чекпойнта не входит",
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="разрешить --own на тестовом наборе — только для сравнения, цифры нечестные",
    )
    parser.add_argument(
        "--own-repeat", type=int, default=20, help="во сколько раз повторить свои кадры"
    )
    parser.add_argument("--own-valid-share", type=float, default=0.2)
    parser.add_argument("--init", type=Path, default=None, help="стартовые веса (свой чекпойнт)")
    parser.add_argument("--eval-only", action="store_true", help="только измерить, не учить")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_set = CocoDetectionDataset(args.data / "train")
    valid_set = CocoDetectionDataset(args.data / "valid")
    if args.limit:
        train_set.ids = train_set.ids[: args.limit]
        valid_set.ids = valid_set.ids[: max(8, args.limit // 8)]

    own_train = own_valid = None
    if args.own is not None:
        if is_holdout(args.own) and not args.include_holdout:
            raise SystemExit(
                f"{args.own} — изолированный тестовый набор, учить на нём нельзя; "
                "передайте его как --test, для обучения размечайте data/train/labels"
            )
        own_train, own_valid = split_by_wine(args.own, args.own_valid_share)
        print(
            f"своя разметка: train {len(own_train)}, valid {len(own_valid)} кадров "
            f"(деление по винам, valid — {int(round(args.own_valid_share * 100))} % вин)"
        )
    test_set = CocoDetectionDataset(args.test) if args.test is not None else None
    if test_set is not None:
        print(f"тестовый набор: {len(test_set)} кадров из {args.test} — только замер")
    print(f"train: {len(train_set)}, valid: {len(valid_set)}, устройство: {device}")

    # num_workers=0: на macOS дочерние процессы плохо уживаются с MPS.
    train_source = train_set
    if own_train is not None and len(own_train):
        train_source = ConcatDataset([train_set] + [own_train] * max(1, args.own_repeat))
    train_loader = DataLoader(
        train_source, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    own_loader = (
        DataLoader(own_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        if own_valid is not None and len(own_valid)
        else None
    )
    test_loader = (
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        if test_set is not None and len(test_set)
        else None
    )

    if args.init is not None:
        checkpoint = torch.load(args.init, map_location="cpu")
        args.backbone = checkpoint["backbone"]
        args.min_size = args.min_size or checkpoint.get("min_size")
    model = build_label_detector(
        args.backbone,
        trainable_backbone_layers=args.trainable_layers,
        min_size=args.min_size,
    ).to(device)
    if args.init is not None:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"старт с весов {args.init}")

    if args.eval_only:
        report("чужая валидация", evaluate(model, valid_loader, device))
        if own_loader is not None:
            report("свои кадры (valid)", evaluate(model, own_loader, device))
        if test_loader is not None:
            report("ТЕСТ", evaluate(model, test_loader, device))
        return

    # Горизонтальное отражение здесь не используем: на этикетке текст, зеркальных этикеток
    # в природе не бывает, и такая аугментация только уводит модель от реальности.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou, best_epoch, stale = 0.0, 0, 0
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
        # Отбор чекпойнта — по своим кадрам, если они есть: чужая валидация уже хороша,
        # а чиним мы обрезку на своих.
        if own_loader is not None:
            own_metrics = evaluate(model, own_loader, device)
            report("  свои кадры", own_metrics)
            metrics = own_metrics

        if test_loader is not None:
            report("  ТЕСТ (не влияет на отбор)", evaluate(model, test_loader, device))

        checkpoint = {
            "backbone": args.backbone,
            "min_size": args.min_size,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.keep_all:
            torch.save(checkpoint, args.out.with_suffix(f".epoch-{epoch:02d}.pt"))
        if metrics["mean_iou"] > best_iou:
            best_iou, best_epoch, stale = metrics["mean_iou"], epoch, 0
            torch.save(checkpoint, args.out)
            print(f"  сохранено в {args.out}")
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"  ранняя остановка: {stale} эпох без роста, лучшая — {best_epoch}")
                break

    print(f"\nлучший mean IoU: {best_iou:.3f} (эпоха {best_epoch})")


if __name__ == "__main__":
    main()
