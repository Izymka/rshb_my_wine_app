"""Метрики поиска на своём тестовом наборе.

    uv run python eval/benchmark.py
    uv run python eval/benchmark.py --no-distractors --tag "без отвлекающих"

Собирает индекс из эталонных кадров своих вин плюс каталог X-Wines как отвлекающие карточки,
прогоняет живые фото и печатает top-1 / top-5 / Recall@50 по группам сложности.

Результат каждого прогона дописывается в eval/results/runs.jsonl. Это и есть будущая
ablation-таблица: строку в неё надо получать сразу после того, как блок заработал, а не
восстанавливать в последние дни хакатона.
"""

import argparse
import json
import platform
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wine_scanner.burst import fuse_rrf, ranked
from wine_scanner.catalog import Query, load_own, load_xwines
from wine_scanner.detect import BottleDetector, CachedCropper, CascadeCropper
from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder, load_image, pick_device
from wine_scanner.index import SearchHit, VectorIndex
from wine_scanner.ocr import DEFAULT_MAX_SIDE, LabelOCR, TextIndex, catalog_document
from wine_scanner.rerank import XFeatMatcher

RESULTS_PATH = Path("eval/results/runs.jsonl")
CROP_CACHE = Path("models/crop_cache")
RECALL_K = 50


def git_revision() -> str:
    """Под какой версией кода получены цифры. Без этого метрики нечем подтвердить."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate(ranks: list[int]) -> dict:
    """Метрики по списку рангов правильного ответа (0 = попал в первую позицию, -1 = не найден)."""
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    arr = np.array(ranks)
    found = arr >= 0
    return {
        "n": n,
        "top1": float(np.mean(found & (arr == 0))),
        "top5": float(np.mean(found & (arr < 5))),
        f"recall@{RECALL_K}": float(np.mean(found)),
        # MRR штрафует мягче, чем top-1: показывает, насколько близко ответ был к первой позиции.
        # maximum нужен, чтобы не делить на ноль на ранге -1 — np.where считает обе ветки.
        "mrr": float(np.mean(np.where(found, 1.0 / np.maximum(arr + 1, 1), 0.0))),
    }


def print_table(overall: dict, by_group: dict[str, dict]) -> None:
    header = f"{'группа':<12}{'кадров':>8}{'top-1':>9}{'top-5':>9}{'R@50':>9}{'MRR':>9}"
    print("\n" + header)
    print("-" * len(header))
    for group in sorted(by_group):
        m = by_group[group]
        print(
            f"{group:<12}{m['n']:>8}{m['top1']:>9.3f}{m['top5']:>9.3f}"
            f"{m[f'recall@{RECALL_K}']:>9.3f}{m['mrr']:>9.3f}"
        )
    print("-" * len(header))
    print(
        f"{'ВСЕ':<12}{overall['n']:>8}{overall['top1']:>9.3f}{overall['top5']:>9.3f}"
        f"{overall[f'recall@{RECALL_K}']:>9.3f}{overall['mrr']:>9.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--descriptor", default="cls_patchmean", choices=["cls", "patchmean", "cls_patchmean"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--detect",
        choices=["bottle", "label", "trained", "cascade"],
        default=None,
        help="обрезать кадр детектором: bottle — рамка бутылки, label — оценка этикетки (Э3)",
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="искать только среди своих вин, без карточек X-Wines",
    )
    parser.add_argument(
        "--tag", default="baseline", help="как называется эта конфигурация в ablation"
    )
    parser.add_argument(
        "--fit", default=None, choices=["center_crop", "squash", "pad"],
        help="как область приводится к квадрату; по умолчанию squash при детекции",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("models/label_detector.pt"),
        help="веса дообученного детектора этикетки для --detect trained",
    )
    parser.add_argument(
        "--text", action="store_true", help="добавить текстовую ветку и слить её через RRF (Э7)"
    )
    parser.add_argument(
        "--ocr-max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help="ужать кадр до этой стороны перед распознаванием; 0 — не ужимать вовсе (Э11)",
    )
    parser.add_argument("--rerank", action="store_true", help="ре-ранкинг top-K по XFeat (Э6)")
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--rerank-max-side", type=int, default=640)
    parser.add_argument("--rerank-points", type=int, default=2048)
    parser.add_argument("--matcher", default="lighterglue", choices=["lighterglue", "descriptors"])
    parser.add_argument(
        "--bottle-backbone", default="resnet50", choices=["resnet50", "mobilenet", "mobilenet320"],
        help="бэкбон первой ступени каскада",
    )
    parser.add_argument(
        "--detect-min-size", type=int, default=None,
        help="переопределить разрешение входа детектора на инференсе",
    )
    parser.add_argument("--errors", type=int, default=10, help="сколько худших случаев показать")
    args = parser.parse_args()

    own_catalog, queries = load_own()
    catalog = list(own_catalog)
    if not args.no_distractors:
        catalog += load_xwines()

    print(
        f"своих вин: {len(own_catalog)}, запросов: {len(queries)}, "
        f"карточек в индексе: {len(catalog)}"
    )

    cropper = None
    if args.detect == "cascade":
        bottle = BottleDetector(device=pick_device(), mode="bottle", backbone=args.bottle_backbone)
        label = BottleDetector(device=pick_device(), weights_path=args.weights)
        cropper = CachedCropper(CascadeCropper(bottle, label), CROP_CACHE)
    elif args.detect:
        weights = args.weights if args.detect == "trained" else None
        detector = BottleDetector(
            device=pick_device(),
            mode=args.detect,
            weights_path=weights,
            infer_min_size=args.detect_min_size,
        )
        cropper = CachedCropper(detector, CROP_CACHE)

    embedder = Dinov2Embedder(
        model_name=args.model, descriptor=args.descriptor, cropper=cropper, fit=args.fit
    )
    print(
        f"модель: {args.model} [{args.descriptor}], устройство: {embedder.device}, "
        f"детекция: {args.detect or 'выкл'}, fit: {embedder.fit}"
    )

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

    text_index = ocr = None
    if args.text:
        text_index = TextIndex(
            [it.item_id for it in catalog], [catalog_document(it.payload) for it in catalog]
        )
        ocr = LabelOCR(max_side=args.ocr_max_side or None)

    # Карточка по идентификатору: после слияния веток в списке могут оказаться кандидаты,
    # которых визуальный поиск не возвращал, и им нужен payload.
    by_id = {
        item_id: SearchHit(item_id, 0.0, payload)
        for item_id, payload in zip(index.item_ids, index.payloads, strict=True)
    }

    matcher = None
    if args.rerank:
        matcher = XFeatMatcher(
            matcher=args.matcher, max_side=args.rerank_max_side, top_k=args.rerank_points
        )

    def crop_of(path: Path):
        image = load_image(path)
        return cropper(path, image) if cropper is not None else image

    def rerank(hits, query_path: Path):
        """Переупорядочить top-K по числу инлаеров, остальных оставить как есть.

        Инлаеры — первичный ключ, косинус из индекса — вторичный. Так пары, которые локальный
        матчинг подтвердить не смог (0 инлаеров), сохраняют порядок глобального поиска,
        а не перемешиваются случайно. Осмысленные веса подберёт LightGBM на Э10.
        """
        head, tail = hits[: args.rerank_k], hits[args.rerank_k :]
        query_features = matcher.describe(crop_of(query_path), cache_key=str(query_path))

        scored = []
        for hit in head:
            path = Path(hit.payload["image_path"])
            candidate = matcher.describe(crop_of(path), cache_key=str(path))
            features = matcher.match(query_features, candidate)
            scored.append((features.score, hit.score, hit))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored] + tail

    ranks: list[int] = []
    text_ranks: list[int] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    mistakes: list[tuple[int, Query, str]] = []

    for query, vector in zip(queries, query_vectors, strict=True):
        hits = index.search(vector.numpy(), top_k=RECALL_K)

        if text_index is not None:
            lines = ocr.read(crop_of(query.path), use_cache=True)
            text_hits = text_index.search(LabelOCR.joined(lines), top_k=RECALL_K)
            text_ids = [h.item_id for h in text_hits]
            text_ranks.append(
                text_ids.index(query.true_id) if query.true_id in text_ids else -1
            )
            # RRF складывает позиции, а не оценки: шкалы косинуса и текстового счёта
            # несопоставимы, приводить их друг к другу пришлось бы подбором коэффициентов.
            fused = fuse_rrf([[h.item_id for h in hits], text_ids])
            hits = [by_id[item_id] for item_id in ranked(fused)][:RECALL_K]

        if matcher is not None:
            hits = rerank(hits, query.path)
        found_ids = [h.item_id for h in hits]
        rank = found_ids.index(query.true_id) if query.true_id in found_ids else -1
        ranks.append(rank)
        by_group[query.group].append(rank)
        if rank != 0:
            mistakes.append((rank, query, found_ids[0] if found_ids else "—"))

    overall = evaluate(ranks)
    groups = {g: evaluate(r) for g, r in by_group.items()}
    print_table(overall, groups)

    text_only = None
    if text_ranks:
        text_only = evaluate(text_ranks)
        print(
            f"\nтекстовая ветка отдельно: top-1 {text_only['top1']:.3f}, "
            f"top-5 {text_only['top5']:.3f}, R@50 {text_only[f'recall@{RECALL_K}']:.3f}"
        )

    if args.errors and mistakes:
        # Сначала те, где правильный ответ вообще не нашёлся, потом самые дальние ранги.
        mistakes.sort(key=lambda m: (m[0] != -1, -m[0]))
        print("\nхудшие случаи (ранг верного ответа, запрос, что выдано первым):")
        for rank, query, top1 in mistakes[: args.errors]:
            shown = "не найдено" if rank == -1 else f"ранг {rank}"
            print(f"  {shown:<12} {query.true_id}/{query.path.name}  ->  {top1}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "tag": args.tag,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_revision(),
        "host": platform.platform(),
        "model": args.model,
        "descriptor": args.descriptor,
        "detect": args.detect,
        "fit": embedder.fit,
        "text": args.text,
        "text_only": text_only,
        "rerank": args.rerank,
        "rerank_k": args.rerank_k if args.rerank else None,
        "matcher": args.matcher if args.rerank else None,
        "rerank_points": args.rerank_points if args.rerank else None,
        "distractors": not args.no_distractors,
        "catalog_size": len(catalog),
        "overall": overall,
        "by_group": groups,
    }
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nзапись добавлена в {RESULTS_PATH}")


if __name__ == "__main__":
    main()
