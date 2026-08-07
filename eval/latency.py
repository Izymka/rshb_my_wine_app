"""Разбивка задержки по блокам пайплайна.

    uv run python eval/latency.py

Цель этапа Э11 — p95 меньше 2 секунд. Чтобы её достигать осмысленно, надо знать, где время
тратится сейчас. Замеряем каждый блок отдельно на реальных кадрах: без разбивки оптимизация
превращается в угадывание.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from wine_scanner.catalog import load_own
from wine_scanner.detect import BottleDetector
from wine_scanner.embed import Dinov2Embedder, build_transform, load_image, pick_device
from wine_scanner.index import VectorIndex


class Timer:
    """Копит замеры по этапам и печатает p50/p95."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}

    def measure(self, stage: str, fn, *args):
        started = time.perf_counter()
        result = fn(*args)
        # MPS считает асинхронно: без синхронизации замер покажет время постановки в очередь,
        # а не время вычисления.
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.samples.setdefault(stage, []).append(time.perf_counter() - started)
        return result

    def report(self) -> None:
        header = f"{'этап':<28}{'p50, мс':>10}{'p95, мс':>10}"
        print("\n" + header)
        print("-" * len(header))
        total_p50 = total_p95 = 0.0
        for stage, values in self.samples.items():
            values = sorted(values)
            p50 = statistics.median(values) * 1000
            p95 = values[int(len(values) * 0.95) - 1] * 1000
            total_p50 += p50
            total_p95 += p95
            print(f"{stage:<28}{p50:>10.0f}{p95:>10.0f}")
        print("-" * len(header))
        print(f"{'ИТОГО':<28}{total_p50:>10.0f}{total_p95:>10.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="сколько кадров прогнать")
    parser.add_argument("--bottle-backbone", default="resnet50")
    parser.add_argument("--weights", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument("--index", type=Path, default=Path("models/index_xwines"))
    args = parser.parse_args()

    device = pick_device()
    _, queries = load_own()
    paths = [q.path for q in queries][: args.n]

    bottle = BottleDetector(device=device, backbone=args.bottle_backbone)
    label = BottleDetector(device=device, weights_path=args.weights)
    embedder = Dinov2Embedder(cropper=None)
    transform = build_transform(fit="pad")
    index = VectorIndex.load(args.index) if args.index.exists() else None

    print(f"кадров: {len(paths)}, устройство: {device}, детектор бутылки: {args.bottle_backbone}")
    if index is None:
        print(f"индекс {args.index} не найден — поиск в замер не войдёт")

    timer = Timer()
    for i, path in enumerate(paths):
        # Первый кадр прогревает ядра и в статистику не идёт.
        active = timer if i > 0 else Timer()

        image = active.measure("декодирование HEIC", load_image, path)
        bottle_crop = active.measure("детекция бутылки", bottle.crop, image)
        label_crop = active.measure("детекция этикетки", label.crop, bottle_crop)
        tensor = active.measure("препроцессинг", transform, label_crop)
        vector = active.measure("эмбеддинг DINOv2", embedder.encode_batch, tensor.unsqueeze(0))
        if index is not None:
            active.measure("поиск в FAISS", index.search, np.asarray(vector[0]), 50)

    timer.report()


if __name__ == "__main__":
    main()
