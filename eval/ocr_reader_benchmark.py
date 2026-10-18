"""Как читает PaddleOCR при разных детекторах строк и размерах кадра — только текстовая ветка.

    uv run python eval/ocr_reader_benchmark.py --configs v5-512,v6-512,v5-full960,v6-full960

Решающий слой обучен на чтении одного определённого OCR, поэтому сравнивать читателей по его
top-1 нечестно: новый читатель проигрывает уже тем, что признаки сдвинулись. Здесь мерится то,
что от OCR зависит напрямую, — на каком месте текстовая ветка ставит верную карточку
(R@1, R@5) и стоит ли она выше всех соседок по винодельне (`family_win`, только у карточек с
соседками). Вырезка этикетки берётся тем же каскадом, что в сервисе; время — только OCR.

Конфигурация `<детектор>-<кадр>[<сторона детектора>][+<распознаватель>]`:

- детектор: `v5` — PP-OCRv5_mobile_det, `v5s` — PP-OCRv5_server_det, `v6` — PP-OCRv6_medium_det,
  `v6s` — PP-OCRv6_small_det;
- кадр: `512` — квадрат эмбеддера, как в сервисе сейчас; `full` — вырезка этикетки в разрешении
  кадра (не больше 2048 по длинной стороне);
- сторона детектора (необязательно): `max` N — детектор работает не больше чем на N пикселях,
  строки распознаватель вырезает из присланного кадра (`WINE_OCR_DET_LIMIT=max`);
- распознаватель (необязательно): без суффикса — eslav_PP-OCRv5_mobile_rec, `+ru` — PP-OCRv6
  medium, дообученный на русском (`models/ocr_rec/PP-OCRv6_medium_rec_ru`, onnxruntime).

По умолчанию кадры — живые знакомые кадры `data/train`: выбор конфигурации делается на них,
тест (`--manifest data/test/manifest.csv`) — только итоговая проверка.
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from wine_scanner.catalog import TRAIN_MANIFEST, load_equivalences, load_manifest
from wine_scanner.embed import load_image
from wine_scanner.ocr import LabelOCR
from wine_scanner.pipeline import WineScanner

RESULTS = Path("eval/results/ocr_reader_runs.jsonl")
DETECTORS = {
    "v5": "PP-OCRv5_mobile_det",
    "v5s": "PP-OCRv5_server_det",
    "v6": "PP-OCRv6_medium_det",
    "v6s": "PP-OCRv6_small_det",
}
RECOGNIZERS = {
    "": "eslav_PP-OCRv5_mobile_rec",
    "ru": "models/ocr_rec/PP-OCRv6_medium_rec_ru",
}
FULL_MAX_SIDE = 2048
CONFIG = re.compile(
    r"^(?P<det>v5s?|v6s?)-(?P<frame>512|full)(?P<side>\d+)?(?:\+(?P<rec>ru))?$"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="v5-512,v6-512,v5-full960,v6-full960")
    parser.add_argument("--manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument(
        "--index", type=Path, default=Path("models/index_platform_sq_v3_site_1810")
    )
    parser.add_argument("--decider", type=Path, default=Path("models/decider_platform_sq_v3_v7"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--dump", type=Path, default=None, help="папка для строк OCR по кадрам")
    args = parser.parse_args()

    queries = [
        q
        for q in load_manifest(args.manifest)
        if q.true_id and not q.true_id.startswith("unknown:") and q.group != "multi"
    ][: args.limit]
    same = load_equivalences().same
    scanner = WineScanner(
        index_dir=args.index, decider_dir=args.decider, crop_cache=Path("models/crop_cache")
    )
    text_index = scanner.text_index
    family_of = text_index.family_of
    members: dict[str, list[str]] = {}
    for item_id, family in family_of.items():
        if family:
            members.setdefault(family, []).append(item_id)

    # Вырезки один раз: квадрат эмбеддера и та же этикетка в разрешении кадра.
    crops = []
    cascade = scanner.cropper
    while cascade is not None and not hasattr(cascade, "crop_with_metadata"):
        cascade = getattr(cascade, "detector", None)
    for q in queries:
        image = load_image(q.path)
        prepared, _ = scanner._crop(image, str(q.path))
        raw = cascade.crop(image) if cascade is not None else image
        crops.append((q, prepared, raw))
    print(f"кадров: {len(crops)} ({args.manifest})")

    for token in args.configs.split(","):
        match = CONFIG.match(token)
        if match is None:
            raise SystemExit(f"не понимаю конфигурацию {token!r}")
        os.environ["WINE_OCR_DET"] = DETECTORS[match["det"]]
        os.environ["WINE_OCR_REC"] = RECOGNIZERS[match["rec"] or ""]
        os.environ.pop("WINE_OCR_DET_SIDE", None)
        os.environ.pop("WINE_OCR_DET_LIMIT", None)
        if match["side"]:
            os.environ["WINE_OCR_DET_SIDE"] = match["side"]
            os.environ["WINE_OCR_DET_LIMIT"] = "max"
        full = match["frame"] == "full"
        ocr = LabelOCR(
            backend="paddle",
            min_confidence=scanner.ocr.min_confidence,
            max_side=FULL_MAX_SIDE if full else None,
        )
        ocr.read(crops[0][1])  # прогрев воркера: модели и CUDA не в счёт времени

        rows = []
        for q, prepared, raw in crops:
            source = raw if full else prepared
            started = time.perf_counter()
            lines = ocr.read(source)
            seconds = time.perf_counter() - started
            text = LabelOCR.joined(lines)
            hits = text_index.search(text, top_k=50) if text else []
            ranked = [h.item_id for h in hits]
            rank = next((i for i, item in enumerate(ranked) if same(q.true_id, item)), 999)
            family = family_of.get(q.true_id, "")
            siblings = [s for s in members.get(family, []) if not same(q.true_id, s)]
            win = None
            if siblings:
                score = {h.item_id: h.score for h in hits}
                true_score = max((v for k, v in score.items() if same(q.true_id, k)), default=-1.0)
                win = true_score > max(score.get(s, -1.0) for s in siblings)
            rows.append(
                {
                    "path": str(q.path),
                    "true_id": q.true_id,
                    "rank": rank,
                    "family_win": win,
                    "seconds": seconds,
                    "words": len(text.split()),
                    "lines": [line.text for line in lines],
                }
            )
        n = len(rows)
        with_family = [r for r in rows if r["family_win"] is not None]
        summary = {
            "tag": args.tag,
            "config": token,
            "detector": os.environ["WINE_OCR_DET"],
            "recognizer": os.environ["WINE_OCR_REC"],
            "frame": "full" if full else "512",
            "det_limit": match["side"],
            "manifest": str(args.manifest),
            "frames": n,
            "text_r1": sum(r["rank"] < 1 for r in rows) / n,
            "text_r5": sum(r["rank"] < 5 for r in rows) / n,
            "text_r50": sum(r["rank"] < 50 for r in rows) / n,
            "family_frames": len(with_family),
            "family_win": sum(r["family_win"] for r in with_family) / max(1, len(with_family)),
            "words_mean": statistics.mean(r["words"] for r in rows),
            "ocr_p50_ms": 1000 * statistics.median(r["seconds"] for r in rows),
            "ocr_p95_ms": 1000 * percentile([r["seconds"] for r in rows], 0.95),
        }
        print(json.dumps(summary, ensure_ascii=False))
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            with (args.dump / f"{token}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        local = ocr._local
        if local is not None and getattr(local, "_process", None) is not None:
            local._process.kill()


if __name__ == "__main__":
    main()
