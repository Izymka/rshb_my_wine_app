"""Собрать признаки для решающего слоя. Дорогая часть, запускается один раз.

    uv run python scripts/build_decision_features.py

Прогоняет весь пайплайн — каскадная обрезка, эмбеддинг, поиск, OCR, ре-ранкинг — и складывает
признаки каждой пары «запрос — кандидат» в eval/results/features.jsonl. Дальше модель обучается
на этом файле за секунды, и перебирать её варианты можно сколько угодно, не трогая пайплайн.

Разделение сделано намеренно: смешивать долгий инференс с быстрым подбором модели — верный
способ потратить вечер на десять экспериментов вместо ста.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from wine_scanner.burst import fuse_rrf, ranked
from wine_scanner.catalog import load_own, load_xwines
from wine_scanner.decide import PairFeatures
from wine_scanner.detect import CachedCropper, build_cropper
from wine_scanner.embed import Dinov2Embedder, load_image, pick_device
from wine_scanner.index import VectorIndex
from wine_scanner.ocr import DEFAULT_MAX_SIDE, LabelOCR, TextIndex, catalog_document
from wine_scanner.rerank import XFeatMatcher
from wine_scanner.vintage import (
    Reading,
    VintageReader,
    YearRegion,
    catalog_years,
    combine,
    compare,
    from_text,
    project,
    reference_region,
)

OUT_PATH = Path("eval/results/features.jsonl")
CROP_CACHE = Path("models/crop_cache")
CANDIDATES = 25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--weights", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ocr-max-side", type=int, default=DEFAULT_MAX_SIDE)
    args = parser.parse_args()

    device = pick_device()
    cropper = CachedCropper(
        build_cropper("cascade", args.weights, device),
        CROP_CACHE,
    )
    embedder = Dinov2Embedder(cropper=cropper, fit="pad")

    own_catalog, queries = load_own()
    catalog = own_catalog + load_xwines()
    print(f"карточек: {len(catalog)}, запросов: {len(queries)}")

    catalog_vectors = embedder.encode_paths(
        [it.image_path for it in catalog], batch_size=args.batch_size
    )
    index = VectorIndex(embedder.dim)
    index.add(
        catalog_vectors.numpy(),
        item_ids=[it.item_id for it in catalog],
        payloads=[{**it.payload, "image_path": str(it.image_path)} for it in catalog],
    )
    query_vectors = embedder.encode_paths([q.path for q in queries], batch_size=args.batch_size)

    text_index = TextIndex(
        [it.item_id for it in catalog], [catalog_document(it.payload) for it in catalog]
    )
    path_by_id = dict(zip(index.item_ids, [p["image_path"] for p in index.payloads], strict=True))

    # OCR прогоняется целиком заранее — он дешевле в одном заходе и результат кладётся в кэш.
    # Раньше здесь стояло ещё и `del ocr`: считалось, что распознавание и XFeat не уживаются
    # в одном процессе. Настоящей причиной зависаний были четыре копии OpenMP, и она устранена
    # в wine_scanner/__init__.py — сквозной пайплайн держит оба блока одновременно и даже
    # в разных потоках. Поэтому распознаватель остаётся живым: он понадобится после
    # ре-ранкинга, чтобы прочитать увеличенный участок с годом.
    ocr = LabelOCR(max_side=args.ocr_max_side or None)
    ocr_by_query: dict[str, tuple[str, int, float]] = {}
    for query in tqdm(queries, desc="OCR"):
        lines = ocr.read(cropper(query.path, load_image(query.path)), use_cache=True)
        confidence = sum(x.confidence for x in lines) / len(lines) if lines else 0.0
        ocr_by_query[str(query.path)] = (LabelOCR.joined(lines), len(lines), confidence)

    matcher = XFeatMatcher(max_side=640, top_k=2048)
    reader = VintageReader(ocr)
    payload_by_id = dict(zip(index.item_ids, index.payloads, strict=True))
    regions: dict[str, YearRegion | None] = {}

    def region_for(item_id: str) -> YearRegion | None:
        """Где на карточке каталога напечатан год. Считается лениво и один раз.

        Лениво — потому что нужен только у того кандидата, чьей геометрией мы пользуемся,
        а карточек в каталоге тысяча. В сервисе эта величина считается при сборке индекса,
        здесь же прогон разовый и кэша OCR достаточно.
        """
        if item_id not in regions:
            path = Path(path_by_id[item_id])
            crop = cropper(path, load_image(path))
            lines = ocr.read(crop, use_cache=True)
            regions[item_id] = reference_region(
                lines, catalog_years(payload_by_id.get(item_id, {}))
            )
        return regions[item_id]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for query, vector in tqdm(
            list(zip(queries, query_vectors, strict=True)), desc="признаки"
        ):
            hits = index.search(vector.numpy(), top_k=50)
            vis_rank = {h.item_id: i for i, h in enumerate(hits)}
            vis_score = {h.item_id: h.score for h in hits}

            text, n_lines, confidence = ocr_by_query[str(query.path)]
            text_hits = text_index.search(text, top_k=50)
            txt_rank = {h.item_id: i for i, h in enumerate(text_hits)}
            txt_score = {h.item_id: h.score for h in text_hits}

            fused = fuse_rrf([[h.item_id for h in hits], [h.item_id for h in text_hits]])
            order = ranked(fused)[: args.candidates]

            query_crop = cropper(query.path, load_image(query.path))
            query_features = matcher.describe(query_crop, cache_key=str(query.path))

            matches = {}
            for item_id in order:
                candidate_path = Path(path_by_id[item_id])
                candidate = matcher.describe(
                    cropper(candidate_path, load_image(candidate_path)),
                    cache_key=str(candidate_path),
                )
                matches[item_id] = matcher.match(query_features, candidate)

            # Год урожая. Сначала общий текст этикетки — он уже прочитан и ничего не стоит.
            # Если года в нём нет, вырезаем участок по геометрии лучшего кандидата: где год
            # напечатан у него, там же он и у нас. Кандидат берётся именно тот, которого выбрал
            # бы сервис, а не правильный ответ — иначе замер показал бы недостижимое.
            vintage_text = from_text(text)
            vintage_zoom = Reading()
            leader = max(order, key=lambda item: matches[item].inliers, default=None)
            if not vintage_text and leader is not None:
                region = region_for(leader)
                if region is not None:
                    box = project(region.box, matches[leader])
                    if box is not None:
                        vintage_zoom = reader.zoom(query_crop, box)
            vintage = combine(vintage_text, vintage_zoom)

            for rrf_position, item_id in enumerate(order):
                match = matches[item_id]

                row = PairFeatures(
                    query=str(query.path),
                    true_id=query.true_id,
                    group=query.group,
                    item_id=item_id,
                    vis_score=vis_score.get(item_id, 0.0),
                    vis_rank=vis_rank.get(item_id, 999),
                    txt_score=txt_score.get(item_id, -1.0),
                    txt_rank=txt_rank.get(item_id, 999),
                    rrf_rank=rrf_position,
                    matches=match.matches,
                    inliers=match.inliers,
                    inlier_ratio=match.inlier_ratio,
                    reproj_error=match.reproj_error,
                    homography_ok=int(match.homography_ok),
                    ocr_lines=n_lines,
                    ocr_conf=confidence,
                    vintage_known=int(bool(vintage)),
                    vintage_match=compare(vintage.year, payload_by_id.get(item_id, {})),
                    vintage_year=vintage.year or 0,
                    vintage_source=vintage.source,
                    vintage_text=vintage_text.year or 0,
                    vintage_zoom=vintage_zoom.year or 0,
                    vintage_ref=leader if leader and vintage_zoom else "",
                )
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                written += 1

    print(f"\nзаписано пар: {written} в {args.out}")


if __name__ == "__main__":
    main()
