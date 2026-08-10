"""Разбивка задержки сквозного запроса по блокам.

    uv run python eval/latency.py --n 20

Цель этапа Э11 — p95 меньше 2 секунд.

Замер идёт через `WineScanner`, то есть через тот же код, который отвечает пользователю.
Прошлая версия этого файла собирала пайплайн заново из отдельных блоков и обрывалась на поиске
в FAISS — и именно так появилась цифра 0.92 с, в которую мы верили, пока сквозной запрос не
показал сорок секунд. Мерить надо ровно то, что работает, иначе замер описывает не систему,
а представление о ней.

Кэши здесь выключены намеренно (`ocr_cache=False`, `crop_cache=None`). С ними замер показывает
скорость чтения файлов, а пользователь на своём снимке платит полную цену.
"""

import argparse
import statistics
import time
from pathlib import Path

from wine_scanner.catalog import load_own
from wine_scanner.embed import load_image
from wine_scanner.pipeline import WineScanner

STAGE_ORDER = ("crop", "embed", "search", "ocr", "text_search", "rerank", "decide")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * q), len(values) - 1)
    return values[index]


def report(samples: dict[str, list[float]], wall: list[float]) -> None:
    header = f"{'этап':<16}{'p50, мс':>10}{'p95, мс':>10}{'доля':>8}"
    print("\n" + header)
    print("-" * len(header))

    total_p50 = sum(statistics.median(v) for v in samples.values())
    for stage in [s for s in STAGE_ORDER if s in samples] + [
        s for s in samples if s not in STAGE_ORDER
    ]:
        values = samples[stage]
        p50 = statistics.median(values)
        p95 = percentile(values, 0.95)
        share = p50 / total_p50 if total_p50 else 0.0
        print(f"{stage:<16}{p50 * 1000:>10.0f}{p95 * 1000:>10.0f}{share:>8.1%}")

    print("-" * len(header))
    print(
        f"{'ЗАПРОС ЦЕЛИКОМ':<16}{statistics.median(wall) * 1000:>10.0f}"
        f"{percentile(wall, 0.95) * 1000:>10.0f}{1.0:>8.1%}"
    )
    print(f"\nцель Э11: p95 < 2000 мс, сейчас {percentile(wall, 0.95) * 1000:.0f} мс")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="сколько кадров прогнать")
    parser.add_argument("--index", type=Path, default=Path("models/index"))
    parser.add_argument("--decider", type=Path, default=Path("models/decider"))
    parser.add_argument("--candidates", type=int, default=25, help="ширина окна ре-ранкинга")
    parser.add_argument("--ocr-max-side", type=int, default=None, help="0 — не ужимать вовсе")
    args = parser.parse_args()

    _, queries = load_own()
    paths = [q.path for q in queries][: args.n + 1]

    scanner = WineScanner(
        index_dir=args.index, decider_dir=args.decider, candidates=args.candidates
    )
    if args.ocr_max_side is not None:
        scanner.ocr.max_side = args.ocr_max_side or None

    print(
        f"кадров: {len(paths) - 1}, каталог: {len(scanner.index.item_ids)}, "
        f"окно ре-ранкинга: {args.candidates}, OCR до стороны: {scanner.ocr.max_side}, "
        f"дескрипторы каталога: {'есть' if scanner.descriptors else 'нет'}"
    )

    samples: dict[str, list[float]] = {}
    wall: list[float] = []
    for i, path in enumerate(paths):
        image = load_image(path)
        started = time.perf_counter()
        result = scanner.identify(image)
        elapsed = time.perf_counter() - started

        # Первый кадр прогревает ядра и в статистику не идёт.
        if i == 0:
            continue
        wall.append(elapsed)
        for stage, value in result.timings.items():
            if stage != "total":
                samples.setdefault(stage, []).append(value)

    report(samples, wall)
    if scanner.recomputed_descriptors:
        print(
            f"\nвнимание: признаки {scanner.recomputed_descriptors} кандидатов считались на лету "
            "— индекс и дескрипторы разошлись, пересоберите индекс"
        )


if __name__ == "__main__":
    main()
