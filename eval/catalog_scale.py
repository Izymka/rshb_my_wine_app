"""Как отбор кандидатов ведёт себя при росте каталога.

    uv run python scripts/fetch_distractors.py --shards 1
    uv run python eval/catalog_scale.py

Этим замером принимается решение по Э9 (дообучение ArcFace), и стоит он вечера вместо
20–40 GPU-часов вслепую.

Логика такая. ArcFace улучшает эмбеддинг, то есть **отбор кандидатов**: порядок внутри
отобранных чинит ре-ранкинг, а вот вернуть вино, не попавшее в top-50, не может уже ничто.
Значит, вопрос «нужен ли Э9» — это вопрос «падает ли R@50». Сейчас он 0.989, но измерен
на каталоге в 1024 карточки, а у платформы их будут десятки тысяч. Одно число на одном
размере каталога тут ничего не решает — нужна кривая.

Меряется визуальная ветка отдельно от текстовой, и это не упрощение, а суть. ArcFace меняет
только её. К тому же у отвлекающих карточек нет названий (в тар-архивах лежат одни картинки),
поэтому текстовая ветка отсеивала бы их даром, и общая цифра вышла бы завышенной.

Дорогая часть — векторы — считается один раз для всех карточек и кладётся на диск, а кривая
строится нарезкой: каталог в 3000 карточек это первые 3000 векторов, и пересчитывать ради него
нечего. Повторный запуск с готовыми векторами — секунды.

Две вещи сделаны не самым очевидным способом, и обе не от хорошей жизни.

Поиск здесь считается матричным умножением, а не через FAISS, хотя FAISS в проекте и есть.
Причина простая: пять индексов подряд в одном процессе кончились сегфолтом на пятом. Для
точного перебора FAISS и так делает ровно это умножение, приближения тут нет, а numpy
не роняет процесс.

И главное — считается диагностика «сколько чужих карточек попало в top-50». Без неё кривая
может оказаться плоской по двум противоположным причинам: либо отбор кандидатов действительно
устойчив, либо отвлекающие настолько не похожи на запросы, что не конкурируют вовсе, и тогда
замер ничего не проверяет.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wine_scanner.catalog import CatalogItem, load_own, load_xwines
from wine_scanner.detect import CachedCropper, build_cropper
from wine_scanner.embed import DEFAULT_MODEL, Dinov2Embedder, pick_device

DISTRACTORS = Path("data/third-party datasets/wine-images-126k/catalog.jsonl")
CROP_CACHE = Path("models/crop_cache")
RESULTS_PATH = Path("eval/results/scale.json")
VECTOR_CACHE = Path("models/scale_vectors.npz")
RECALL_K = 50
SIZES = (1_024, 2_000, 4_000, 8_000, 11_024)


def load_distractors(path: Path, limit: int | None = None) -> list[CatalogItem]:
    """Чужие вина из открытого набора. В каталоге они играют роль соседей по полке.

    Ни одно из них не является правильным ответом ни для одного запроса, поэтому любое
    попадание такой карточки выше правильной — это ровно та ошибка отбора, которую мы ищем.
    """
    if not path.exists():
        raise SystemExit(
            f"нет файла {path} — сначала `uv run python scripts/fetch_distractors.py --shards 1`"
        )
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            items.append(
                CatalogItem(
                    item_id=f"distractor:{row['item_id']}",
                    image_path=Path(row["image_path"]),
                    payload={"name": row.get("name", "")},
                )
            )
            if limit and len(items) >= limit:
                break
    return items


def evaluate(
    queries, query_vectors: np.ndarray, item_ids: list[str], vectors: np.ndarray
) -> dict:
    """Метрики отбора кандидатов и доля чужих карточек в выдаче.

    Векторы нормированы (за этим следит Dinov2Embedder), поэтому скалярное произведение —
    это косинус, а сортировка по нему и есть поиск ближайших соседей.
    """
    scores = query_vectors @ vectors.T
    top = np.argsort(-scores, axis=1)[:, :RECALL_K]

    ranks, intruders = [], []
    for row, query in zip(top, queries, strict=True):
        ids = [item_ids[i] for i in row]
        ranks.append(ids.index(query.true_id) if query.true_id in ids else -1)
        intruders.append(sum(1 for item_id in ids if item_id.startswith("distractor:")))

    arr = np.array(ranks)
    found = arr >= 0
    return {
        "n": len(ranks),
        "top1": float(np.mean(found & (arr == 0))),
        "top5": float(np.mean(found & (arr < 5))),
        f"recall@{RECALL_K}": float(np.mean(found)),
        # Сколько чужих карточек в среднем пробилось в top-50 и в top-1 хотя бы раз.
        # Это проверка не системы, а самого замера: если здесь ноль, отвлекающие
        # не конкурируют, и плоская кривая ничего не доказывает.
        "intruders_at_50": float(np.mean(intruders)),
        "queries_led_by_intruder": int(
            sum(
                1
                for row, query in zip(top, queries, strict=True)
                if item_ids[row[0]].startswith("distractor:")
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distractors", type=Path, default=DISTRACTORS)
    parser.add_argument("--limit", type=int, default=None, help="взять не весь набор, для отладки")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weights", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--vectors",
        type=Path,
        default=VECTOR_CACHE,
        help="куда сложить посчитанные векторы, чтобы кривую можно было пересчитать за секунды",
    )
    parser.add_argument("--recompute", action="store_true", help="считать векторы заново")
    args = parser.parse_args()

    own, queries = load_own()
    base = own + load_xwines()
    extra = load_distractors(args.distractors, args.limit)
    items = base + extra
    print(f"свои: {len(own)}, X-Wines: {len(base) - len(own)}, отвлекающих: {len(extra)}")
    print(f"запросов: {len(queries)}")

    cached = _load_vectors(args.vectors, items, queries) if not args.recompute else None
    if cached is not None:
        vectors, query_vectors = cached
        print(f"векторы взяты готовыми: {args.vectors}")
        return _report(args, queries, items, vectors, query_vectors)

    device = pick_device()
    # Обрезка ровно та же, что у своих карточек и у запросов. Соблазн пропустить её для
    # отвлекающих велик — их десять тысяч, и это часы, — но тогда они попадут в другое
    # пространство признаков, окажутся неестественно далеко от запросов, и замер покажет
    # благополучие, которого нет.
    cropper = CachedCropper(
        build_cropper("cascade", args.weights, device),
        CROP_CACHE,
    )
    embedder = Dinov2Embedder(
        model_name=DEFAULT_MODEL, device=device, cropper=cropper, fit="pad"
    )

    print("\nвекторы каталога (считаются один раз на все размеры)")
    vectors = embedder.encode_paths(
        [it.image_path for it in items], batch_size=args.batch_size
    ).numpy()
    query_vectors = embedder.encode_paths(
        [q.path for q in queries], batch_size=args.batch_size
    ).numpy()

    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.vectors,
        vectors=vectors,
        query_vectors=query_vectors,
        item_ids=np.array([it.item_id for it in items]),
        query_paths=np.array([str(q.path) for q in queries]),
    )
    print(f"векторы сохранены: {args.vectors}")

    _report(args, queries, items, vectors, query_vectors)


def _load_vectors(path: Path, items, queries):
    """Готовые векторы, если они посчитаны для ровно этого набора карточек и запросов."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    same_items = list(data["item_ids"]) == [it.item_id for it in items]
    same_queries = list(data["query_paths"]) == [str(q.path) for q in queries]
    if not (same_items and same_queries):
        print("готовые векторы не подходят: набор карточек или запросов изменился, считаем заново")
        return None
    return data["vectors"], data["query_vectors"]


def _report(args, queries, items, vectors: np.ndarray, query_vectors: np.ndarray) -> None:
    item_ids = [it.item_id for it in items]
    sizes = [s for s in SIZES if s <= len(items)]
    if len(items) not in sizes:
        sizes.append(len(items))

    table = []
    print(f"\n{'каталог':>8}{'top-1':>8}{'top-5':>8}{'R@50':>8}{'чужих в top-50':>16}")
    for size in sizes:
        row = {"catalog": size, **evaluate(queries, query_vectors, item_ids[:size], vectors[:size])}
        table.append(row)
        print(
            f"{size:>8}{row['top1']:>8.3f}{row['top5']:>8.3f}"
            f"{row[f'recall@{RECALL_K}']:>8.3f}{row['intruders_at_50']:>16.1f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "queries": len(queries),
                "wines": len({q.true_id for q in queries}),
                "branch": "visual",
                "distractors": str(args.distractors),
                "curve": table,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
