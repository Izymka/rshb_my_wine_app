"""Compare padding colors on wine-grouped validation from train, never test.

Retrieval-only ablation, own catalog, no whitening/OCR/decider. Saved RT-DETR label crops
are held fixed. Results do not stand in for full platform recognition metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from validation_split import own_validation_queries

from wine_scanner.catalog import load_own
from wine_scanner.embed import load_image
from wine_scanner.embed.preprocess import PadToSquare, build_transform

COLORS = {
    "current": (124, 116, 104),
    "neutral": (128, 128, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/derived/rtdetr_audit"))
    parser.add_argument("--detector-meta", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/derived/padding_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.crop_root
    rows = [
        json.loads(line)
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    crops = {Path(r["source"]).resolve(): r["label_crop"] for r in rows if "error" not in r}
    catalog, _ = load_own()
    ids = [item.item_id for item in catalog]
    queries, validation = own_validation_queries(ids, args.detector_meta)
    if not queries:
        raise ValueError("No known train queries for own catalog")
    paths = [crops[item.image_path.resolve()] for item in catalog]
    paths += [crops[q.path.resolve()] for q in queries]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModel.from_pretrained(
            "models/siglip2",
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    results, predictions = {}, []
    for name, fill in COLORS.items():
        transform = build_transform(384, "pad", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transform.transforms[0] = PadToSquare(fill)
        vectors = []
        with torch.inference_mode():
            for path in paths:
                batch = transform(load_image(path)).unsqueeze(0).to(device, dtype=model.dtype)
                features = model.get_image_features(pixel_values=batch)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                vectors.append(F.normalize(features.float(), p=2, dim=-1).cpu().numpy()[0])
        vectors = np.asarray(vectors)
        scores = vectors[len(catalog) :] @ vectors[: len(catalog)].T
        ranking = np.argsort(-scores, axis=1)
        per_wine = {}
        for query, order in zip(queries, ranking, strict=True):
            rank = list(order).index(ids.index(query.wine_id)) + 1
            per_wine.setdefault(query.wine_id, []).append(rank)
            predictions.append(
                {
                    "color": name,
                    "path": query.path.as_posix(),
                    "wine_id": query.wine_id,
                    "rank": rank,
                }
            )
        ranks = [rank for values in per_wine.values() for rank in values]
        results[name] = {
            "fill": fill,
            "queries": len(queries),
            "catalog": len(catalog),
            "r1": float(np.mean(np.asarray(ranks) <= 1)),
            "r5": float(np.mean(np.asarray(ranks) <= 5)),
            "wine_macro_r1": float(
                np.mean([np.mean(np.asarray(v) <= 1) for v in per_wine.values()])
            ),
            "wine_macro_r5": float(
                np.mean([np.mean(np.asarray(v) <= 5) for v in per_wine.values()])
            ),
        }
        print(name, results[name], flush=True)
    # Stable insertion order resolves ties in favor of the unchanged production baseline.
    best = max(
        results, key=lambda key: (results[key]["wine_macro_r1"], results[key]["wine_macro_r5"])
    )
    with Path("models/siglip2/model.safetensors").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "selection": best,
        "validation_wines": sorted(validation),
        "seed": 2026,
        "model_sha256": model_hash,
        "model": "google/siglip2-so400m-patch16-384",
        "scope": "own-catalog retrieval, validation queries from train, no OCR/whitening",
        "dtype": str(model.dtype),
        "results": results,
        "predictions": predictions,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected:", best, flush=True)


if __name__ == "__main__":
    main()
