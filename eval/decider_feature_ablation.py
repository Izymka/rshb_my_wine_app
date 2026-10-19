"""Сравнить наборы признаков решающего слоя по OOF на train — без теста.

    uv run python eval/decider_feature_ablation.py \\
        --features eval/results/features_platform_sq_v3_v5.jsonl

Протокол как во внешней CV ноутбука 06: `GroupKFold` по винам, CatBoost с параметрами
действующего решающего слоя (`--params-from`), живые кадры весят `--live-weight`, каждый
знакомый запрос дополнительно подаётся без верного кандидата. Меняется только набор колонок,
поэтому разница — это вклад признаков, а не подбора. Несколько сидов CatBoost показывают шум.

Верным считается и выбор второй карточки того же вина из
`data/splits/catalog_slug_equivalences.csv` (как `final_top1` бенчмарка); строгое совпадение —
колонка `*_strict`. Метки обучения не меняются.

Незнакомые вина (живые кадры, которых нет в каталоге) оцениваются по той же OOF: `auroc_unknown`
— насколько максимальная оценка отделяет живые знакомые кадры от незнакомых, `fa_at_cov` —
доля незнакомых, получивших ответ, при пороге, на котором ответ получают `--coverage` живых
знакомых, `right_at_cov` — доля живых знакомых, получивших верный ответ при том же пороге.
`--extra` задаёт, какие признаки добавить ко второму набору (по умолчанию «прочитанное» v5).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from wine_scanner.catalog import is_holdout, load_equivalences
from wine_scanner.decide import FEATURE_NAMES, PairFeatures, derive
from wine_scanner.decide.features import RAW_FEATURE_VERSIONS

UNKNOWN_PREFIX = "unknown:"
READING = (
    "grape_match",
    "attr_agree",
    "family_disc_margin",
    "family_color_margin",
    "family_style_margin",
    "family_grape_margin",
    "family_attr_margin",
)
# Признаки текста, с ростом которых уверенность не должна падать (--monotone text).
TEXT_MONOTONE = (
    "color_match", "style_match", "grape_match", "attr_agree", "name_cover", "disc_hit",
    "winery_hit", "family_color_margin", "family_style_margin", "family_grape_margin",
    "family_attr_margin", "family_disc_margin",
)
RESULTS = Path("eval/results/decider_feature_ablation.jsonl")


def load(path: Path, versions=RAW_FEATURE_VERSIONS) -> dict[str, list[PairFeatures]]:
    grouped: dict[str, list[PairFeatures]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            if payload.get("feature_version") not in versions:
                raise SystemExit(f"{path}: сырые пары старой версии, пересоберите признаки")
            row = PairFeatures.from_dict(payload)
            if not is_holdout(row.query):
                grouped[row.query].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--params-from", type=Path, default=Path("models/decider_platform_sq_v3_slug")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--live-weight", type=float, default=3.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--extra", default=",".join(READING), help="признаки второго набора")
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument(
        "--raw-version", type=int, default=None,
        help="принять сырые пары этой версии (сравнение со старым файлом признаков)",
    )
    parser.add_argument(
        "--monotone", default="",
        help="монотонные ограничения: имя:+1 или имя:-1 через запятую; "
        "«text» — набор признаков текста по умолчанию",
    )
    parser.add_argument(
        "--veto", default="",
        help="оси запрета после модели (color,style,grape): оценка карточки, которой "
        "противоречит читаемый текст, умножается на --veto-factor",
    )
    parser.add_argument("--veto-factor", type=float, default=0.0)
    parser.add_argument(
        "--veto-mode", choices=("exclude", "refuse"), default="exclude",
        help="exclude — понизить карточку (наверх выйдет следующая); refuse — выбор прежний, "
        "понижается только уверенность выбранной",
    )
    parser.add_argument(
        "--gap-rivals", choices=("all", "text"), default="all",
        help="визуальные разрывы против всех соперников или только не опровергнутых текстом",
    )
    parser.add_argument("--dump", type=Path, default=None, help="выборы сида 2026 в jsonl")
    args = parser.parse_args()

    meta = json.loads((args.params_from / "meta.json").read_text(encoding="utf-8"))
    base = list(meta["feature_names"])
    extra = [n for n in args.extra.split(",") if n and n not in base]
    if any(n not in FEATURE_NAMES for n in extra):
        parser.error(f"неизвестные признаки: {[n for n in extra if n not in FEATURE_NAMES]}")
    sets = {meta.get("feature_set", "deployed"): base}
    if extra:
        sets["+" + ",".join(extra)] = base + extra
    params = dict(meta["params"])

    by_query = load(args.features, {args.raw_version} if args.raw_version else RAW_FEATURE_VERSIONS)
    queries = sorted(by_query)
    wine = {q: by_query[q][0].true_id for q in queries}
    live = {q: by_query[q][0].group.startswith("live-") for q in queries}
    known = {q: not wine[q].startswith(UNKNOWN_PREFIX) for q in queries}
    same_wine = load_equivalences().group_of

    full, labels, no_true = {}, {}, {}
    for q in queries:
        rows = derive(by_query[q], rivals=args.gap_rivals)
        full[q] = {n: np.asarray([float(r[n]) for r in rows]) for n in FEATURE_NAMES}
        labels[q] = np.asarray([r["label"] for r in rows])
        rest = [c for c in by_query[q] if not c.label]
        if len(rest) < len(by_query[q]):
            no_true[q] = {
                n: np.asarray([float(r[n]) for r in derive(rest, rivals=args.gap_rivals)])
                for n in FEATURE_NAMES
            }

    monotone = {}
    for item in filter(None, args.monotone.split(",")):
        if item == "text":
            monotone.update({n: 1 for n in TEXT_MONOTONE})
            continue
        name, _, sign = item.partition(":")
        monotone[name] = int(sign)
    if any(n not in FEATURE_NAMES for n in monotone):
        parser.error(f"неизвестные признаки: {[n for n in monotone if n not in FEATURE_NAMES]}")
    veto_axes = [f"{a}_match" for a in filter(None, args.veto.split(","))]

    def vetoed(q: str, scores: np.ndarray) -> np.ndarray:
        """Запрет по тексту: карточке, которой противоречит читаемая этикетка, — оценка ниже."""
        if not veto_axes:
            return scores
        block = full[q]
        hit = (block["ocr_lines"] >= 3) & np.any([block[a] == -1 for a in veto_axes], axis=0)
        return np.where(hit, scores * args.veto_factor, scores)

    def columns(block: dict, names: list[str]) -> np.ndarray:
        return np.column_stack([block[n] for n in names])

    def fit(train_q: list[str], names: list[str], seed: int) -> CatBoostClassifier:
        x, y, w = [], [], []
        for q in train_q:
            weight = args.live_weight if live[q] else 1.0
            x.append(columns(full[q], names))
            y.append(labels[q])
            w.append(np.full(len(labels[q]), weight))
            if q in no_true:
                x.append(columns(no_true[q], names))
                y.append(np.zeros(len(no_true[q][names[0]]), int))
                w.append(np.full(len(no_true[q][names[0]]), weight))
        extra = {}
        if monotone:
            extra["monotone_constraints"] = [monotone.get(n, 0) for n in names]
        model = CatBoostClassifier(
            **params, **extra, loss_function="Logloss", random_seed=seed, thread_count=-1,
            allow_writing_files=False, verbose=False,
        )
        return model.fit(np.vstack(x), np.concatenate(y), sample_weight=np.concatenate(w))

    known_q = [q for q in queries if known[q]]
    folds = list(GroupKFold(args.folds).split(queries, groups=[wine[q] for q in queries]))
    print(
        f"запросов {len(queries)}, знакомых {len(known_q)} "
        f"(живых {sum(live[q] for q in known_q)}), сидов {args.seeds}"
    )

    summary, picks = {}, {}
    for name, names in sets.items():
        per_seed = []
        for seed in range(2026, 2026 + args.seeds):
            chosen = {}
            for train_idx, test_idx in folds:
                model = fit([queries[i] for i in train_idx], names, seed)
                for i in test_idx:
                    q = queries[i]
                    if known[q] or live[q]:
                        raw = model.predict_proba(columns(full[q], names))[:, 1]
                        scores = vetoed(q, raw)
                        best = int(np.argmax(scores))
                        if args.veto_mode == "refuse":
                            best = int(np.argmax(raw))
                            # Уверенность ответа — у выбранной; остальные не выше неё.
                            scores = np.minimum(raw, vetoed(q, raw)[best])
                        chosen[q] = (by_query[q][best].item_id, scores, labels[q])
            per_seed.append(chosen)

        def metric(chosen: dict, subset: list[str], strict: bool) -> float:
            ok = []
            for q in subset:
                best = chosen[q][0]
                group = same_wine.get(wine[q], frozenset({wine[q]}))
                ok.append(best == wine[q] if strict else best in group)
            return float(np.mean(ok))

        def mrr(chosen: dict, subset: list[str]) -> float:
            out = []
            for q in subset:
                _, scores, y = chosen[q]
                out.append(1 / (1 + int((scores > scores[y == 1].max()).sum())) if y.any() else 0)
            return float(np.mean(out))

        live_q = [q for q in known_q if live[q]]
        syn_q = [q for q in known_q if not live[q]]
        unknown_q = [q for q in queries if live[q] and not known[q]]

        def unknown_metrics(chosen: dict, live_q=live_q, unknown_q=unknown_q) -> dict:
            from sklearn.metrics import roc_auc_score

            top_known = np.asarray([chosen[q][1].max() for q in live_q])
            top_unknown = np.asarray([chosen[q][1].max() for q in unknown_q])
            auroc = roc_auc_score(
                np.r_[np.ones(len(top_known)), np.zeros(len(top_unknown))],
                np.r_[top_known, top_unknown],
            )
            cut = np.quantile(top_known, 1 - args.coverage)
            right = [
                chosen[q][1].max() >= cut
                and chosen[q][0] in same_wine.get(wine[q], frozenset({wine[q]}))
                for q in live_q
            ]
            return {
                "auroc_unknown": float(auroc),
                "fa_at_cov": float(np.mean(top_unknown >= cut)),
                "right_at_cov": float(np.mean(right)),
            }

        rows = [
            {
                "live": metric(c, live_q, False),
                "live_strict": metric(c, live_q, True),
                "synthetic": metric(c, syn_q, True),
                "mrr_live": mrr(c, live_q),
                "mrr": mrr(c, known_q),
                **unknown_metrics(c),
            }
            for c in per_seed
        ]
        summary[name] = {k: [r[k] for r in rows] for k in rows[0]}
        picks[name] = per_seed[0]
        counts = [round(r["live"] * len(live_q)) for r in rows]
        print(f"\n{name} ({len(names)} признаков): живые верно {counts} из {len(live_q)}")
        for k, values in summary[name].items():
            print(f"  {k:12s} {np.mean(values):.3f} ± {np.std(values):.3f}")

    first, second = (list(sets) * 2)[:2]
    print(f"\nживые кадры, где наборы расходятся (сид 2026): {first} -> {second}")
    for q in sorted(q for q in known_q if live[q]):
        a, b = picks[first][q][0], picks[second][q][0]
        if a != b:
            group = same_wine.get(wine[q], frozenset({wine[q]}))
            mark_a, mark_b = ("ok" if x in group else "x" for x in (a, b))
            print(f"  {mark_a}->{mark_b} {wine[q]}: {a} -> {b}")

    if args.dump:
        with args.dump.open("w", encoding="utf-8") as fh:
            for name in sets:
                for q, (best, scores, _) in picks[name].items():
                    fh.write(json.dumps({
                        "set": name, "query": q, "wine": wine[q], "known": known[q],
                        "live": live[q], "best": best, "p": float(scores.max()),
                    }, ensure_ascii=False) + "\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "features": str(args.features), "params_from": str(args.params_from),
            "folds": args.folds, "seeds": args.seeds, "live_weight": args.live_weight,
            "monotone": args.monotone, "veto": args.veto, "veto_factor": args.veto_factor,
            "gap_rivals": args.gap_rivals,
            "summary": summary,
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
