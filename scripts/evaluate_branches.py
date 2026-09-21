"""Wine-grouped train validation for OCR/XFeat RGB, grayscale and CLAHE branches."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from validation_split import own_validation_queries

from wine_scanner.catalog import load_own
from wine_scanner.embed import load_image
from wine_scanner.embed.branches import BRANCH_MODES, branch_image
from wine_scanner.ocr import LabelOCR, TextIndex, fold
from wine_scanner.rerank import XFeatMatcher


def paired_intervals(records, wines, repeats=2000):
    """Resample wines, not correlated frames; intervals describe uncertainty, not significance."""
    rng = np.random.default_rng(2026)
    wines = sorted(wines)
    result = {}
    for key, cutoff in (("text_rank", 5), ("geometry_rank", 1)):
        values = {}
        for mode in BRANCH_MODES:
            values[mode] = np.array(
                [
                    np.mean(
                        [
                            r[key] <= cutoff
                            for r in records
                            if r["mode"] == mode and r["wine"] == wine
                        ]
                    )
                    for wine in wines
                ]
            )
        sample = rng.integers(0, len(wines), size=(repeats, len(wines)))
        for mode in BRANCH_MODES:
            delta = values[mode] - values["rgb"]
            low, high = np.quantile(delta[sample].mean(axis=1), [0.025, 0.975])
            result[f"{mode}:{key}:r{cutoff}"] = {
                "delta_vs_rgb": float(delta.mean()),
                "ci95": [float(low), float(high)],
            }
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/derived/rtdetr_audit"))
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-meta", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = [
        json.loads(s)
        for s in (args.crop_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    crops = {Path(r["source"]).resolve(): r["label_crop"] for r in rows if "error" not in r}
    catalog, _ = load_own()
    ids = [item.item_id for item in catalog]
    queries, validation = own_validation_queries(ids, args.detector_meta)
    text_index = TextIndex.from_payloads(ids, [item.payload for item in catalog])
    matcher = XFeatMatcher(device=torch.device(args.device))
    ocr = LabelOCR(gpu=args.device == "cuda", backend="paddle")
    output = Path(f"data/derived/branches_{args.tag}.json")
    records, summaries = [], {}
    for mode in BRANCH_MODES:
        started = time.perf_counter()
        references = [
            matcher.describe(branch_image(load_image(crops[item.image_path.resolve()]), mode))
            for item in catalog
        ]
        mode_rows = []
        for query in queries:
            image = branch_image(load_image(crops[query.path.resolve()]), mode)
            lines = ocr.read(image, use_cache=True)
            text = ocr.joined(lines)
            textual = text_index.search(text, top_k=len(ids)) if text.strip() else []
            ranked = [hit.item_id for hit in textual]
            txt_rank = ranked.index(query.wine_id) + 1 if query.wine_id in ranked else len(ids) + 1
            features = matcher.describe(image)
            matches = [matcher.match(features, reference) for reference in references]
            scores = [match.score for match in matches]
            index = ids.index(query.wine_id)
            # Conservative tie handling: all-zero geometry is a failure, not rank 1.
            geom_rank = 1 + sum(
                score >= scores[index] for j, score in enumerate(scores) if j != index
            )
            target_text = fold(str(catalog[index].payload.get("label_text", "")))
            normalized = fold(text)
            cer = (
                Levenshtein.distance(normalized, target_text) / len(target_text)
                if target_text
                else None
            )
            record = {
                "mode": mode,
                "query": query.path.as_posix(),
                "wine": query.wine_id,
                "text": text,
                "text_rank": txt_rank,
                "geometry_rank": geom_rank,
                "cer_proxy": cer,
                "true_inliers": matches[index].inliers,
                "true_inlier_ratio": matches[index].inlier_ratio,
                "true_reprojection": matches[index].reproj_error,
                "best_wrong_inliers": max(score for j, score in enumerate(scores) if j != index),
            }
            mode_rows.append(record)
            print(mode, len(mode_rows), "/", len(queries), flush=True)

        def macro(key, cutoff, mode_rows=mode_rows):
            return float(
                np.mean(
                    [
                        np.mean([r[key] <= cutoff for r in mode_rows if r["wine"] == wine])
                        for wine in validation
                    ]
                )
            )

        summaries[mode] = {
            "text_macro_r1": macro("text_rank", 1),
            "text_macro_r5": macro("text_rank", 5),
            "geometry_macro_r1": macro("geometry_rank", 1),
            "geometry_macro_r5": macro("geometry_rank", 5),
            "mean_true_inliers": float(np.mean([r["true_inliers"] for r in mode_rows])),
            "seconds": time.perf_counter() - started,
        }
        records.extend(mode_rows)
        report = {
            "validation_wines": sorted(validation),
            "queries": len(queries),
            "catalog": len(ids),
            "crop_root": str(args.crop_root),
            "device": args.device,
            "summary": summaries,
            "records": records,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(mode, summaries[mode], flush=True)

    report["selection_rule"] = "OCR: macro R@5 then R@1; XFeat: macro R@1 then R@5; ties RGB first"
    report["selected_ocr"] = max(
        summaries, key=lambda m: (summaries[m]["text_macro_r5"], summaries[m]["text_macro_r1"])
    )
    report["selected_local"] = max(
        summaries,
        key=lambda m: (summaries[m]["geometry_macro_r1"], summaries[m]["geometry_macro_r5"]),
    )
    report["paired_wine_bootstrap"] = paired_intervals(records, validation)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
