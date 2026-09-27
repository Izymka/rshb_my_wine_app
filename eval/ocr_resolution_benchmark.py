"""До какой стороны ужимать этикетку перед OCR: чтение близнецов против задержки. Только train.

    uv run python eval/ocr_resolution_benchmark.py --sides legacy,640,1280,0

`0` — без ужатия, `legacy` — как до 27.09: OCR читает вырезку 512 px, приготовленную для
эмбеддера (из-за неё граница 640 на деле не работала). Граница 640 выбрана замером
10.08.2026 ещё на прошлом читателе и по общему top-1; здесь мерится то, ради чего OCR
нужен решающему слою сейчас: отличает ли прочитанное верную карточку от сильнейшей соседки
по винодельне (`family_attr_margin`, `family_disc_margin`, признаки v5).
Кадры — живые знакомые кадры `data/train`, тест не трогается. Кэш OCR выключен,
чтобы время было настоящим; VLM-судья не зовётся.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wine_scanner.catalog import TRAIN_MANIFEST, load_equivalences, load_manifest
from wine_scanner.embed import load_image
from wine_scanner.pipeline import WineScanner

RESULTS = Path("eval/results/ocr_resolution_runs.jsonl")


def sign(value: float) -> str:
    return "+" if value > 0 else "-" if value < 0 else "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", default="legacy,640,1280,0")
    parser.add_argument("--index", type=Path, default=Path("models/index_platform_sq_v3"))
    parser.add_argument("--decider", type=Path, default=Path("models/decider_platform_sq_v3_slug"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    queries = [
        q
        for q in load_manifest(TRAIN_MANIFEST)
        if q.true_id and not q.true_id.startswith("unknown:") and q.group != "multi"
    ]
    queries = queries[: args.limit] if args.limit else queries
    same = load_equivalences().same
    scanner = WineScanner(index_dir=args.index, decider_dir=args.decider)
    print(f"живых знакомых кадров train: {len(queries)}")

    for token in args.sides.split(","):
        legacy = token == "legacy"
        side = 640 if legacy else int(token)
        scanner.ocr_fullres = not legacy
        scanner.ocr.max_side = side or None
        rows = []
        for q in queries:
            image = load_image(q.path)
            started = time.perf_counter()
            result = scanner.identify(image, use_judge=False)
            total = time.perf_counter() - started
            true = next((c for c in result.candidates if c.item_id == q.true_id), None)
            f = true.features if true else {}
            rows.append({
                "path": str(q.path),
                "ocr_s": result.timings.get("ocr", 0.0),
                "total_s": total,
                "lines": len(result.text.split()) if result.text else 0,
                "top1": bool(result.best and same(q.true_id, result.best.item_id)),
                "answered": bool(result.answered),
                "has_sibling": f.get("family_size", 0) > 1,
                "attr": sign(f.get("family_attr_margin", 0)),
                "disc": sign(f.get("family_disc_margin", 0)),
            })
        sib = [r for r in rows if r["has_sibling"]]
        summary = {
            "side": token if legacy else (side or "full"),
            "frames": len(rows),
            "ocr_p50": float(np.percentile([r["ocr_s"] for r in rows], 50)),
            "ocr_p95": float(np.percentile([r["ocr_s"] for r in rows], 95)),
            "total_p50": float(np.percentile([r["total_s"] for r in rows], 50)),
            "total_p95": float(np.percentile([r["total_s"] for r in rows], 95)),
            "words_mean": float(np.mean([r["lines"] for r in rows])),
            "top1": sum(r["top1"] for r in rows),
            "answered_correct": sum(r["top1"] and r["answered"] for r in rows),
            "with_sibling": len(sib),
            "attr_for_true": sum(r["attr"] == "+" for r in sib),
            "attr_against_true": sum(r["attr"] == "-" for r in sib),
            "disc_for_true": sum(r["disc"] == "+" for r in sib),
            "disc_against_true": sum(r["disc"] == "-" for r in sib),
        }
        print(json.dumps(summary, ensure_ascii=False))
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": args.tag, **summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
