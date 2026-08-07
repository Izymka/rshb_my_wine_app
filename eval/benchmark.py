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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wine_scanner.catalog import load_own, load_xwines
from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder
from wine_scanner.index import VectorIndex

RESULTS_PATH = Path("eval/results/runs.jsonl")
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
    parser.add_argument("--descriptor", default="cls_patchmean", choices=["cls", "patchmean", "cls_patchmean"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="искать только среди своих вин, без карточек X-Wines",
    )
    parser.add_argument("--tag", default="baseline", help="как называется эта конфигурация в ablation")
    parser.add_argument("--errors", type=int, default=10, help="сколько худших случаев показать")
    args = parser.parse_args()

    own_catalog, queries = load_own()
    catalog = list(own_catalog)
    if not args.no_distractors:
        catalog += load_xwines()

    print(f"своих вин: {len(own_catalog)}, запросов: {len(queries)}, карточек в индексе: {len(catalog)}")

    embedder = Dinov2Embedder(model_name=args.model, descriptor=args.descriptor)
    print(f"модель: {args.model} [{args.descriptor}], устройство: {embedder.device}")

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

    ranks: list[int] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    mistakes: list[tuple[int, Query, str]] = []

    for query, vector in zip(queries, query_vectors, strict=True):
        hits = index.search(vector.numpy(), top_k=RECALL_K)
        found_ids = [h.item_id for h in hits]
        rank = found_ids.index(query.true_id) if query.true_id in found_ids else -1
        ranks.append(rank)
        by_group[query.group].append(rank)
        if rank != 0:
            mistakes.append((rank, query, found_ids[0] if found_ids else "—"))

    overall = evaluate(ranks)
    groups = {g: evaluate(r) for g, r in by_group.items()}
    print_table(overall, groups)

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
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "host": platform.platform(),
        "model": args.model,
        "descriptor": args.descriptor,
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
