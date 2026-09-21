"""Сравнение стратегий работы с серией кадров.

    uv run python eval/burst_benchmark.py

Серии берутся из Live Photo, снятых рядом с обычными кадрами. Сравниваем пять стратегий:

    один кадр   — усреднение по всем кадрам серии, то есть «повезло с моментом съёмки»
    резкий кадр — выбираем кадр с максимальной резкостью и ищем только по нему
    max / mean  — объединяем оценки поиска по всем кадрам
    RRF         — объединяем ранги
    оракул      — верхняя граница: засчитываем успех, если хоть один кадр дал верный ответ

Оракул нужен для интерпретации: он показывает, сколько вообще можно выжать из серии, и любая
стратегия оценивается расстоянием до него, а не до единицы.
"""

import argparse
import json
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wine_scanner.burst import fuse_max, fuse_mean, fuse_rrf, load_bursts, ranked, sharpness
from wine_scanner.catalog import load_own, load_xwines
from wine_scanner.detect import CachedCropper, build_cropper
from wine_scanner.embed import Dinov2Embedder, load_image, pick_device
from wine_scanner.index import VectorIndex

RESULTS_PATH = Path("eval/results/bursts.jsonl")
CROP_CACHE = Path("models/crop_cache")
TOP_K = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--weights", type=Path, default=Path("models/rtdetr_label"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device()
    cropper = CachedCropper(
        build_cropper("cascade", args.weights, device),
        CROP_CACHE,
    )
    embedder = Dinov2Embedder(cropper=cropper, fit="pad")

    own_catalog, _ = load_own()
    catalog = own_catalog + load_xwines()
    catalog_vectors = embedder.encode_paths(
        [it.image_path for it in catalog], batch_size=args.batch_size
    )
    index = VectorIndex(embedder.dim)
    index.add(
        catalog_vectors.numpy(),
        item_ids=[it.item_id for it in catalog],
        payloads=[{"image_path": str(it.image_path)} for it in catalog],
    )

    bursts = load_bursts(fps=args.fps)
    lengths = [len(b.frames) for b in bursts]
    print(
        f"серий: {len(bursts)}, вин: {len({b.true_id for b in bursts})}, "
        f"кадров в серии: {min(lengths)}–{max(lengths)}, всего кадров: {sum(lengths)}"
    )

    hits: dict[str, list[int]] = defaultdict(list)

    for burst in tqdm(bursts, desc="серии"):
        # Резкость считаем по обрезанной этикетке, а не по всему кадру: фон может быть
        # резким при смазанной бутылке, и тогда выбор кадра окажется бессмысленным.
        sharp = [sharpness(cropper(f, load_image(f))) for f in burst.frames]
        order_by_sharp = list(np.argsort(sharp)[::-1])
        sharpest = int(order_by_sharp[0])
        best3 = [int(i) for i in order_by_sharp[:3]]

        vectors = embedder.encode_paths(burst.frames, batch_size=args.batch_size, progress=False)

        per_frame_scores, per_frame_ranking, frame_top1 = [], [], []
        for vector in vectors:
            found = index.search(vector.numpy(), top_k=TOP_K)
            per_frame_scores.append({h.item_id: h.score for h in found})
            per_frame_ranking.append([h.item_id for h in found])
            frame_top1.append(bool(found) and found[0].item_id == burst.true_id)

        hits["один кадр"].append(sum(frame_top1) / len(frame_top1))
        hits["резкий кадр"].append(float(frame_top1[sharpest]))
        hits["оракул"].append(float(any(frame_top1)))
        for name, fused in (
            ("max по кадрам", fuse_max(per_frame_scores)),
            ("mean по кадрам", fuse_mean(per_frame_scores)),
            ("RRF по кадрам", fuse_rrf(per_frame_ranking)),
            # Объединение по всем кадрам тянет вниз мусор из смазанных. Пробуем объединять
            # только три самых резких: и отбор кадров, и объединение одновременно.
            ("max по 3 резким", fuse_max([per_frame_scores[i] for i in best3])),
            ("RRF по 3 резким", fuse_rrf([per_frame_ranking[i] for i in best3])),
        ):
            order = ranked(fused)
            hits[name].append(float(bool(order) and order[0] == burst.true_id))

    print(f"\n{'стратегия':<18}{'top-1':>9}")
    print("-" * 27)
    order = [
        "один кадр", "резкий кадр", "max по кадрам", "mean по кадрам",
        "RRF по кадрам", "max по 3 резким", "RRF по 3 резким", "оракул",
    ]
    summary = {}
    for name in order:
        value = float(np.mean(hits[name]))
        summary[name] = value
        print(f"{name:<18}{value:>9.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "host": platform.platform(),
                    "fps": args.fps,
                    "bursts": len(bursts),
                    "frames": sum(lengths),
                    "results": summary,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
