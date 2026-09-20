"""Бенчмарк на каталоге платформы по живым кадрам.

    uv run python eval/platform_benchmark.py --tag baseline
    uv run python eval/platform_benchmark.py --decider none --candidates 50 --tag retrieval-only
    uv run python eval/platform_benchmark.py --manifest data/train/manifest.csv --tag train   # обучающий набор, только для сравнения

Тестовый набор изолирован (`catalog.is_holdout`): `data/test` и `data/eval` не участвуют ни в
обучении решающего слоя, ни в подборе порога, ни в обучении детектора — поэтому цифры отсюда
честные ровно настолько, насколько решающий слой обучен без `--include-holdout`
(`decider_meta.holdout_isolated` в записи прогона).

Единственный измеритель, который отвечает на вопрос «что увидит скрипт организаторов»: гоняется
тот самый `WineScanner`, что стоит за `/v1/eval/predict`, с тем же индексом и решающим слоем,
по кадрам из `data/test/manifest.csv` (`scripts/import_live_photos.py`). До него метрики на
платформе считались либо на псевдофото из вырезок, либо на трёх публичных кадрах — и ни то,
ни другое не описывало живую полку.

Что считается и зачем именно это.

Для кадров, у которых верная карточка в каталоге есть, путь ответа разложен по ступеням —
и на каждой видно, где верная карточка потерялась:

- визуальный ранг по всему каталогу (R@1/5/10/25/50/100) — дошла ли она до визуальных
  кандидатов вообще;
- текстовый ранг (R@1/5/10/25/50) — спасает ли её текстовая ветка;
- попала ли в окно ре-ранкинга — иначе XFeat и решающий слой её уже не видят;
- лидер по инлаерам — верна ли геометрия;
- итоговый top-1 решающего слоя и исход: ответ верный / неверный той же винодельни (близнец) /
  неверный чужой / отказ.

Для незнакомых кадров считается одно — ответила система или отказала, — отдельно для трудных
(российская полка, кириллица, в каталоге есть близнец или та же винодельня) и лёгких (импорт).
Именно на трудных проверяется п. 5 ТЗ: честное «не найдено» вместо чужого slug.

Правило чтения цифр: известных кадров пока два с половиной десятка на четыре вина, и система
ошибается по вину целиком, а не по кадру. Разница меньше 15 пунктов между двумя прогонами —
шум, а не результат. Таблица по винам поэтому важнее сводной.

Каждый прогон дописывает строку в `eval/results/platform_runs.jsonl` с полной конфигурацией —
это и есть ablation-таблица.
"""

import argparse
import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import TEST_MANIFEST, Query, load_manifest
from wine_scanner.decide.guard import DEFAULT_MODE as DEFAULT_GUARD
from wine_scanner.decide.guard import GUARD_MODES
from wine_scanner.embed import load_image
from wine_scanner.index import VectorIndex
from wine_scanner.ocr import TextIndex
from wine_scanner.ocr.search import CATALOG_FIELDS
from wine_scanner.pipeline import (
    RERANK_CANDIDATES,
    TEXT_CANDIDATES,
    VISUAL_CANDIDATES,
    RetrievalOnlyDecider,
    ScanResult,
    WineScanner,
)

RESULTS = Path("eval/results/platform_runs.jsonl")
CROP_CACHE = Path("models/crop_cache")
VISUAL_K = (1, 5, 10, 25, 50, 100)
TEXT_K = (1, 5, 10, 25, 50)
MISSING_RANK = 10**6
STAGE_ORDER = (
    "crop",
    "embed",
    "search",
    "ocr",
    "text_search",
    "rerank",
    "vintage",
    "decide",
    "total",
)


@dataclass
class Outcome:
    """Что случилось с одним кадром."""

    query: Query
    known: bool
    vis_rank: int = MISSING_RANK
    txt_rank: int = MISSING_RANK
    in_long_list: bool = False
    in_window: bool = False
    rerank_top1: bool = False
    guard: str | None = None
    final_top1: bool = False
    answered: bool = False
    verdict: str = (
        ""  # correct / twin / other / refused (известные); accepted / refused (незнакомые)
    )
    best_id: str = ""
    probability: float = 0.0
    timings: dict = field(default_factory=dict)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    return values[min(int(len(values) * q), len(values) - 1)]


def rate(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def fmt(x: float) -> str:
    return "  —  " if x != x else f"{x:.3f}"


def fmt_rank(x: float) -> str:
    return "—" if x >= MISSING_RANK / 2 else f"{x:.0f}"


def full_visual_rank(scanner: WineScanner, vector, true_id: str) -> int:
    hits = scanner.index.search(vector, top_k=len(scanner.index.item_ids))
    for rank, hit in enumerate(hits):
        if hit.item_id == true_id:
            return rank
    return MISSING_RANK


def evaluate(scanner: WineScanner, query: Query, result: ScanResult) -> Outcome:
    trace = result.trace or {}
    out = Outcome(query=query, known=query.known, timings=dict(result.timings))
    best = result.best
    out.answered = result.answered
    out.guard = result.guard
    out.best_id = best.item_id if best else ""
    out.probability = best.probability if best else 0.0

    if not query.known:
        out.verdict = "accepted" if result.answered else "refused"
        return out

    true_id = query.true_id
    out.vis_rank = full_visual_rank(scanner, trace["vector"], true_id)
    textual = [item_id for item_id, _ in trace["textual"]]
    out.txt_rank = textual.index(true_id) if true_id in textual else MISSING_RANK
    out.in_long_list = true_id in trace.get("long_list", [])
    window = trace["window"]
    out.in_window = true_id in window
    inliers = trace["inliers"]
    if window:
        leader = max(window, key=lambda item_id: inliers.get(item_id, 0))
        out.rerank_top1 = leader == true_id and inliers.get(leader, 0) > 0
    out.final_top1 = bool(best and best.item_id == true_id)

    if not result.answered:
        out.verdict = "refused"
    elif out.final_top1:
        out.verdict = "correct"
    else:
        winery = scanner.payload_by_id.get(true_id, {}).get("winery")
        answered_winery = scanner.payload_by_id.get(best.item_id, {}).get("winery")
        out.verdict = "twin" if winery and winery == answered_winery else "other"
    return out


def summarize(outcomes: list[Outcome]) -> dict:
    known = [o for o in outcomes if o.known]
    unknown = [o for o in outcomes if not o.known]
    n = len(known)
    metrics = {
        "known": n,
        "unknown": len(unknown),
        "visual": {f"r@{k}": rate(sum(o.vis_rank < k for o in known), n) for k in VISUAL_K},
        "text": {f"r@{k}": rate(sum(o.txt_rank < k for o in known), n) for k in TEXT_K},
        "long_list_hit": rate(sum(o.in_long_list for o in known), n),
        "window_hit": rate(sum(o.in_window for o in known), n),
        "guard_fired_known": sum(bool(o.guard and "twin" in o.guard) for o in known),
        "guard_fired_unknown": sum(bool(o.guard and "twin" in o.guard) for o in unknown),
        "sibling_swaps_known": sum(bool(o.guard and "sibling" in o.guard) for o in known),
        "sibling_swaps_unknown": sum(bool(o.guard and "sibling" in o.guard) for o in unknown),
        "rerank_top1": rate(sum(o.rerank_top1 for o in known), n),
        "final_top1": rate(sum(o.final_top1 for o in known), n),
        "verdicts": dict(Counter(o.verdict for o in known)),
        "answered_precision": rate(
            sum(o.verdict == "correct" for o in known), sum(o.answered for o in known)
        ),
        "coverage": rate(sum(o.answered for o in known), n),
    }
    by_group: dict[str, list[Outcome]] = defaultdict(list)
    for o in unknown:
        by_group[o.query.group].append(o)
    metrics["false_accept"] = {
        group: {"n": len(items), "rate": rate(sum(o.answered for o in items), len(items))}
        for group, items in sorted(by_group.items())
    }
    metrics["false_accept"]["all"] = {
        "n": len(unknown),
        "rate": rate(sum(o.answered for o in unknown), len(unknown)),
    }
    return metrics


def per_wine(outcomes: list[Outcome]) -> list[dict]:
    by_wine: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_wine[o.query.wine_id].append(o)
    rows = []
    for wine_id, items in sorted(by_wine.items(), key=lambda kv: (not kv[1][0].known, kv[0])):
        known = items[0].known
        row = {
            "wine_id": wine_id,
            "known": known,
            "n": len(items),
            "source": items[0].query.source,
            "answered": sum(o.answered for o in items),
        }
        if known:
            row.update(
                vis_rank_median=statistics.median(o.vis_rank for o in items),
                txt_rank_median=statistics.median(o.txt_rank for o in items),
                window_hit=sum(o.in_window for o in items),
                rerank_top1=sum(o.rerank_top1 for o in items),
                final_top1=sum(o.final_top1 for o in items),
                correct=sum(o.verdict == "correct" for o in items),
                twin=sum(o.verdict == "twin" for o in items),
            )
        else:
            row["group"] = items[0].query.group
            row["best"] = Counter(o.best_id for o in items).most_common(1)[0][0]
        rows.append(row)
    return rows


def latency(outcomes: list[Outcome]) -> dict:
    samples: dict[str, list[float]] = defaultdict(list)
    for o in outcomes:
        for stage, seconds in o.timings.items():
            samples[stage].append(seconds)
    return {
        stage: {"p50_ms": statistics.median(v) * 1000, "p95_ms": percentile(v, 0.95) * 1000}
        for stage, v in samples.items()
    }


def print_report(metrics: dict, wines: list[dict], timing: dict, errors: list[Outcome]) -> None:
    print(f"\nизвестных кадров: {metrics['known']}, незнакомых: {metrics['unknown']}")
    print(
        "визуальный ранг верной   "
        + "  ".join(f"{k}:{fmt(v)}" for k, v in metrics["visual"].items())
    )
    print(
        "текстовый ранг верной    " + "  ".join(f"{k}:{fmt(v)}" for k, v in metrics["text"].items())
    )
    print(
        f"в длинном списке {fmt(metrics['long_list_hit'])}   в окне ре-ранкинга {fmt(metrics['window_hit'])}"
        f"   лидер по инлаерам {fmt(metrics['rerank_top1'])}   итоговый top-1 {fmt(metrics['final_top1'])}"
    )
    print(
        f"защита сработала: на известных {metrics['guard_fired_known']}, "
        f"на незнакомых {metrics['guard_fired_unknown']}; выбор внутри семьи по тексту: "
        f"{metrics['sibling_swaps_known']} / {metrics['sibling_swaps_unknown']}"
    )
    v = metrics["verdicts"]
    print(
        f"исходы: верно {v.get('correct', 0)}, близнец {v.get('twin', 0)}, чужое {v.get('other', 0)}, "
        f"отказ {v.get('refused', 0)}  →  точность среди ответов {fmt(metrics['answered_precision'])}, "
        f"покрытие {fmt(metrics['coverage'])}"
    )
    fa = metrics["false_accept"]
    print(
        "ложные приёмы незнакомых: "
        + ", ".join(f"{g} {d['rate']:.3f} ({d['n']})" for g, d in fa.items() if d["n"])
    )

    header = f"{'вино':<52}{'n':>3}{'vis':>6}{'txt':>6}{'окно':>5}{'геом':>5}{'top1':>5}{'ответ':>6}{'верно':>6}{'близн':>6}"
    print("\n" + header)
    print("-" * len(header))
    for w in wines:
        if not w["known"]:
            continue
        print(
            f"{w['wine_id'][:51]:<52}{w['n']:>3}{fmt_rank(w['vis_rank_median']):>6}{fmt_rank(w['txt_rank_median']):>6}"
            f"{w['window_hit']:>5}{w['rerank_top1']:>5}{w['final_top1']:>5}{w['answered']:>6}"
            f"{w['correct']:>6}{w['twin']:>6}"
        )
    header = f"{'незнакомец':<52}{'n':>3}{'группа':>14}{'ответил':>8}  чаще всего называет"
    print("\n" + header)
    print("-" * len(header))
    for w in wines:
        if w["known"]:
            continue
        print(
            f"{w['wine_id'][:51]:<52}{w['n']:>3}{w['group']:>14}{w['answered']:>8}  {w['best'][:40]}"
        )

    print(f"\n{'этап':<14}{'p50, мс':>10}{'p95, мс':>10}")
    for stage in [s for s in STAGE_ORDER if s in timing] + [
        s for s in timing if s not in STAGE_ORDER
    ]:
        print(f"{stage:<14}{timing[stage]['p50_ms']:>10.0f}{timing[stage]['p95_ms']:>10.0f}")

    if errors:
        print("\nхудшие известные кадры (визуальный ранг, текстовый ранг, что ответили):")
        for o in errors:
            print(
                f"  {o.query.path}  vis={o.vis_rank if o.vis_rank < MISSING_RANK else '—'}"
                f"  txt={o.txt_rank if o.txt_rank < MISSING_RANK else '—'}  {o.verdict}"
                f"  {o.best_id[:45]} p={o.probability:.2f}"
            )


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 — вне репозитория это просто не нужно
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--index", type=Path, default=Path("models/index_platform"))
    parser.add_argument(
        "--decider",
        default="models/decider_platform",
        help="папка решающего слоя или `none` — только порядок слияния веток",
    )
    parser.add_argument(
        "--candidates", type=int, default=RERANK_CANDIDATES, help="окно ре-ранкинга"
    )
    parser.add_argument("--visual-candidates", type=int, default=VISUAL_CANDIDATES)
    parser.add_argument("--text-candidates", type=int, default=TEXT_CANDIDATES)
    parser.add_argument(
        "--window-extra",
        type=int,
        default=None,
        help="сколько подтверждённых текстом добавить в окно",
    )
    parser.add_argument("--no-family", action="store_true", help="без расширения по винодельне")
    parser.add_argument(
        "--guard", default=DEFAULT_GUARD, choices=GUARD_MODES, help="защита от близнеца"
    )
    parser.add_argument("--no-sibling", action="store_true", help="без выбора внутри семьи по тексту")
    parser.add_argument(
        "--text-scorer", default=None, choices=["ngram", "bm25"], help="текстовый скорер"
    )
    parser.add_argument(
        "--text-fields",
        default=None,
        help="поля документа через запятую, например name,winery,category",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="порог отказа вместо порога модели"
    )
    parser.add_argument(
        "--sources",
        default="live,own,eval",
        help="источники кадров манифеста через запятую (live, own, eval)",
    )
    parser.add_argument(
        "--include-multi", action="store_true", help="брать кадры с несколькими бутылками"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="без кэшей кропов и OCR — честная задержка"
    )
    parser.add_argument("--sequential", action="store_true", help="текстовая ветка не в потоке")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--errors", type=int, default=8, help="сколько худших известных кадров показать"
    )
    parser.add_argument("--tag", default="", help="метка прогона для runs.jsonl")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument(
        "--dump", type=Path, default=None, help="куда сложить исходы по кадрам (jsonl)"
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = [
        q for q in load_manifest(args.manifest, include_multi=args.include_multi) if q.source in sources
    ]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit("кадров нет: проверьте --manifest и --sources")

    decider = RetrievalOnlyDecider() if args.decider == "none" else None
    text_index = None
    if args.text_scorer or args.text_fields:
        index = VectorIndex.load(args.index)
        fields = tuple(args.text_fields.split(",")) if args.text_fields else CATALOG_FIELDS
        text_index = TextIndex.from_payloads(
            index.item_ids, index.payloads, fields=fields, scorer=args.text_scorer
        )
    scanner = WineScanner(
        index_dir=args.index,
        decider_dir=Path(args.decider if decider is None else "models/_none"),
        candidates=args.candidates,
        threshold=args.threshold,
        crop_cache=None if args.no_cache else CROP_CACHE,
        ocr_cache=not args.no_cache,
        parallel=not args.sequential,
        decider=decider,
        text_index=text_index,
        visual_candidates=args.visual_candidates,
        text_candidates=args.text_candidates,
        family_expansion=not args.no_family,
        window_extra=args.window_extra,
        guard=args.guard,
        sibling=not args.no_sibling,
    )
    print(
        f"кадров: {len(queries)} (известных {sum(q.known for q in queries)}), каталог: "
        f"{len(scanner.index.item_ids)}, окно: {args.candidates}, порог: {scanner.threshold:.3f}, "
        f"решающий слой: {args.decider}, кэши: {'нет' if args.no_cache else 'да'}"
    )

    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for query in tqdm(queries, desc="кадры"):
        key = None if args.no_cache else str(query.path)
        result = scanner.identify(load_image(query.path), image_key=key, trace=True)
        outcomes.append(evaluate(scanner, query, result))
    elapsed = time.perf_counter() - started

    metrics = summarize(outcomes)
    wines = per_wine(outcomes)
    timing = latency(outcomes)
    worst = sorted(
        (o for o in outcomes if o.known and not o.final_top1),
        key=lambda o: (o.answered, -o.vis_rank),
    )[: args.errors]
    print_report(metrics, wines, timing, worst)
    print(
        f"\nвесь прогон: {elapsed:.0f} с, дескрипторов пересчитано: {scanner.recomputed_descriptors}"
    )

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_head(),
        "tag": args.tag,
        "config": {
            "index": str(args.index),
            "index_config": scanner.config,
            "version": scanner.version,
            "decider": args.decider,
            "decider_meta": getattr(scanner.decider, "meta", None),
            "threshold": scanner.threshold,
            "candidates": args.candidates,
            "visual_candidates": args.visual_candidates,
            "text_candidates": args.text_candidates,
            "window_extra": scanner.window_extra,
            "family_expansion": scanner.family_expansion,
            "guard": args.guard,
            "sibling": not args.no_sibling,
            "text_scorer": scanner.text_index.scorer,
            "text_fields": args.text_fields,
            "sources": sorted(sources),
            "manifest": str(args.manifest),
            "cached": not args.no_cache,
            "parallel": not args.sequential,
            "devices": scanner.devices(),
        },
        "metrics": metrics,
        "per_wine": wines,
        "latency": timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"записано в {args.out}")

    if args.dump:
        with args.dump.open("w", encoding="utf-8") as fh:
            for o in outcomes:
                fh.write(
                    json.dumps(
                        {
                            "path": str(o.query.path),
                            "wine_id": o.query.wine_id,
                            "true_id": o.query.true_id,
                            "group": o.query.group,
                            "vis_rank": o.vis_rank,
                            "txt_rank": o.txt_rank,
                            "in_long_list": o.in_long_list,
                            "in_window": o.in_window,
                            "guard": o.guard,
                            "rerank_top1": o.rerank_top1,
                            "final_top1": o.final_top1,
                            "verdict": o.verdict,
                            "best_id": o.best_id,
                            "probability": o.probability,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
