"""Обучить решающий слой и положить его на диск для пайплайна.

    uv run python scripts/train_decider.py

Работает на готовом eval/results/features.jsonl, считается за секунды. От eval/decide_benchmark.py
отличается назначением: тот отвечает на вопрос «работает ли идея», этот делает артефакт, который
поднимет сервис. Метрики здесь печатаются те же, но короче — подробный разбор остаётся в бенчмарке.

Три вещи, которые обязаны считаться именно так, а не проще:

1. Калибровка обучается на отложенных предсказаниях (out-of-fold), а не на тех же данных,
   что и бустер. На своих же обучающих ответах модель почти всегда права, и калибровка по ним
   выдала бы уверенность 0.99 на всё подряд — то есть ровно уничтожила бы возможность отказа.
2. Разбиение на фолды идёт по винам, а не по кадрам: у одного вина несколько снимков, и,
   разъехавшись между обучением и контролем, они превратят задачу «узнай вино» в «вспомни вино».
3. Сам бустер в конце переобучается на всех данных. Фолды нужны были для честной оценки,
   в продукт идёт модель, видевшая весь материал.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from wine_scanner.catalog import is_holdout
from wine_scanner.decide import FEATURE_NAMES, Decider, PairFeatures, derive, logit, matrix
from wine_scanner.decide.features import RAW_FEATURE_VERSIONS

FEATURES_PATH = Path("eval/results/features.jsonl")
OUT_DIR = Path("models/decider")


FAMILIES: dict[str, str] = {}
UNKNOWN_PREFIX = "unknown:"


def family_key(item_id: str, row: PairFeatures | None = None) -> str:
    """Группировка по производителю.

    Для каталога платформы винодельня едет в самой строке признаков (`family`, с версии 2)
    или берётся из таблицы (`--family-map`). Для своего набора и X-Wines — первые два слова
    идентификатора, как и раньше.
    """
    if row is not None and row.family:
        return row.family
    return FAMILIES.get(item_id) or "_".join(item_id.split("_")[:2])


def is_unknown(true_id: str) -> bool:
    """Живой незнакомый запрос: вина нет в каталоге, ни один кандидат не верен."""
    return true_id.startswith(UNKNOWN_PREFIX)


def load_queries(path: Path) -> dict[str, list[PairFeatures]]:
    grouped: dict[str, list[PairFeatures]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            if payload.get("feature_version") not in RAW_FEATURE_VERSIONS:
                raise ValueError(
                    "Raw candidate features are incompatible with the current pipeline; "
                    "rebuild them"
                )
            row = PairFeatures.from_dict(payload)
            grouped[row.query].append(row)
    return grouped


def training_rows(
    groups: list[list[PairFeatures]], augment_unknown: bool, live_weight: float = 1.0
) -> tuple[list, list, list]:
    """Матрица и метки для обучения.

    `augment_unknown` добавляет к каждому запросу его же копию без правильного ответа. Смысл
    в том, что порог отказа обучается на данных, где отказываться не от чего: в обычной
    выборке верный кандидат всегда присутствует, и признаки вида «здесь нет верного ответа»
    ничего не улучшают, поэтому модель их не берёт. Замер: AUROC отказа 0.895 -> 0.945.
    """
    rows, labels, weights = [], [], []
    for candidates in groups:
        # Живые кадры весят больше псевдофото: их в разы меньше, а описывают они именно то,
        # что придёт в сервис, — телефонную съёмку с полки, а не вырезку на размытом фоне.
        weight = live_weight if candidates[0].group.startswith("live-") else 1.0
        derived = derive(candidates)
        rows += matrix(derived)
        labels += [r["label"] for r in derived]
        weights += [weight] * len(derived)

        if not augment_unknown:
            continue
        without_true = [c for c in candidates if not c.label]
        if len(without_true) < len(candidates):
            unknown = derive(without_true)
            rows += matrix(unknown)
            labels += [0] * len(unknown)
            weights += [weight] * len(unknown)
    return rows, labels, weights


def make_confidence_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        loss_function="Logloss",
        random_seed=2026,
        thread_count=4,
        allow_writing_files=False,
        # Положительных примеров на порядки меньше — по одному верному кандидату на запрос
        # при длинном списке в сотню с лишним строк.
        auto_class_weights="Balanced",
        verbose=False,
    )


def make_ranker() -> CatBoostRanker:
    """Модель порядка внутри одного запроса, а не независимой вероятности пары."""
    return CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6,
        loss_function="PairLogitPairwise",
        random_seed=2026,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )


def ranking_pool(groups: list[list[PairFeatures]], live_weight: float = 1.0) -> tuple[Pool, dict]:
    """Сформировать пары «истинная карточка лучше трудного соседа» для ranker.

    Случайные далёкие вина не участвуют: в каждом запросе уже лежат сильные FAISS/OCR-соседи
    и расширение той же винодельни. Пары с ними получают тройной вес, чтобы ranker учился
    именно различать близнецов, а не повторять лёгкое отделение чужой бутылки.
    """
    rows: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    pairs: list[tuple[int, int]] = []
    pair_weights: list[float] = []
    included = 0
    for query_id, candidates in enumerate(groups):
        if not candidates or is_unknown(candidates[0].true_id):
            continue
        derived = derive(candidates)
        positive = [index for index, row in enumerate(derived) if row["label"]]
        if not positive:
            # Верный slug не попал в long-list: это задача retrieval, ranker его не создаст.
            continue
        offset = len(rows)
        rows.extend(matrix(derived))
        labels.extend(row["label"] for row in derived)
        group_ids.extend([query_id] * len(derived))
        live = live_weight if candidates[0].group.startswith("live-") else 1.0
        true_family = family_key(candidates[positive[0]].true_id, candidates[positive[0]])
        for winner in positive:
            for loser, candidate in enumerate(candidates):
                if loser == winner:
                    continue
                hard = (
                    family_key(candidate.item_id, candidate) == true_family
                    or candidate.vis_rank < 5
                    or candidate.txt_rank < 5
                    or candidate.rrf_rank < 5
                )
                pairs.append((offset + winner, offset + loser))
                pair_weights.append(live * (3.0 if hard else 1.0))
        included += 1
    if not pairs:
        raise ValueError("No positive candidate pairs for CatBoostRanker")
    return (
        Pool(
            np.asarray(rows),
            label=np.asarray(labels),
            group_id=np.asarray(group_ids),
            pairs=pairs,
            pairs_weight=np.asarray(pair_weights),
        ),
        {"queries": included, "pairs": len(pairs), "hard_pair_weight": 3.0},
    )


def pick_threshold(
    known: np.ndarray,
    known_correct: np.ndarray,
    unknown: np.ndarray,
    budget: float,
) -> dict:
    """Наименьший порог, при котором доля ответов на незнакомые вина укладывается в бюджет.

    Напрашивается более простой критерий — «точность среди отвеченных не ниже 0.95». Он не
    работает, и понять почему важнее, чем взять готовое число. Точность на смеси зависит от
    доли незнакомых вин в потоке, а у нас она задана способом проверки: правильный ответ
    выбрасывается из кандидатов ровно один раз на запрос, значит незнакомых ровно половина.
    В реальном потоке их будет заметно меньше — и любая цифра точности, посчитанная на смеси
    50/50, к продукту отношения не имеет.

    Поэтому порог выбирается по двум величинам, которые от доли незнакомцев не зависят вовсе:
    сколько незнакомых вин мы ошибочно опознали (это и есть бюджет) и какое покрытие остаётся
    на знакомых. Точность на любой интересующей доле восстанавливается из них арифметикой —
    см. precision_at_prior.
    """
    for threshold in np.arange(0.0, 1.0, 0.01):
        answered_unknown = float((unknown >= threshold).mean())
        if answered_unknown > budget:
            continue
        answered = known >= threshold
        return {
            "threshold": round(float(threshold), 2),
            "coverage": float(answered.mean()),
            "precision": float(known_correct[answered].mean()) if answered.any() else 0.0,
            "false_answer_rate": answered_unknown,
        }
    # Бюджет недостижим ни при каком пороге, кроме «не отвечать никогда». Такое бывает, когда
    # незнакомое вино набирает столько же, сколько верное, — у нас так ведёт себя сценарий
    # с близнецом в каталоге. Возвращаем честный отказ вместо порога, который ничего не значит.
    return {"threshold": 1.0, "coverage": 0.0, "precision": 0.0, "false_answer_rate": 0.0}


def precision_at_prior(picked: dict, unknown_share: float) -> float:
    """Какой окажется точность среди отвеченных, если незнакомых вин в потоке доля p.

    Из отвеченных запросов доля (1-p)*coverage приходится на знакомые вина, и среди них верны
    precision; доля p*false_answer_rate — на незнакомые, и там верных нет по определению.
    """
    good = (1 - unknown_share) * picked["coverage"] * picked["precision"]
    total = (1 - unknown_share) * picked["coverage"] + unknown_share * picked["false_answer_rate"]
    return good / total if total else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--budget",
        type=float,
        default=0.10,
        help="какую долю незнакомых вин допустимо ошибочно опознать; по ней и выбирается порог",
    )
    parser.add_argument(
        "--scenario",
        choices=["family", "nofamily", "real"],
        default="nofamily",
        help="как моделируется незнакомое вино: с роднёй в каталоге, без (по умолчанию) "
        "или по живым незнакомым запросам из признаков (true_id = unknown:...)",
    )
    parser.add_argument(
        "--family-map",
        type=Path,
        default=None,
        help="csv со столбцами slug и winery — родня по винодельне (каталог платформы)",
    )
    parser.add_argument(
        "--only-groups",
        default=None,
        help="учить на запросах выбранных групп (префиксы через запятую), "
        "например live- — без псевдофото",
    )
    parser.add_argument(
        "--live-weight",
        type=float,
        default=3.0,
        help="вес живых кадров (group live-*) относительно псевдофото при обучении",
    )
    parser.add_argument(
        "--no-augment-unknown",
        dest="augment_unknown",
        action="store_false",
        help="учить только на запросах, где верный ответ есть в каталоге (как было до Э8)",
    )
    args = parser.parse_args()

    if args.family_map is not None:
        import pandas as pd

        table = pd.read_csv(args.family_map)
        FAMILIES.update(dict(zip(table["slug"], table["winery"], strict=True)))

    by_query = load_queries(args.features)
    holdout = [q for q in by_query if is_holdout(q)]
    for q in holdout:
        del by_query[q]
    print(f"Тестовый набор изолирован: исключено {len(holdout)} запросов")
    if args.only_groups:
        prefixes = tuple(args.only_groups.split(","))
        by_query = {q: rows for q, rows in by_query.items() if rows[0].group.startswith(prefixes)}
    queries = sorted(by_query)
    wines = [by_query[q][0].true_id for q in queries]
    groups_of = {q: by_query[q][0].group for q in queries}
    known = np.array([not is_unknown(w) for w in wines])
    pairs = sum(map(len, by_query.values()))
    print(
        f"запросов: {len(queries)} (знакомых {int(known.sum())}, живых незнакомых "
        f"{int((~known).sum())}), вин: {len(set(wines))}, пар: {pairs}"
    )

    splitter = GroupKFold(n_splits=args.folds)
    oof_raw = np.zeros(len(queries))
    oof_rank = np.zeros(len(queries))
    oof_correct = np.zeros(len(queries), dtype=bool)
    oof_item_ids = [""] * len(queries)
    oof_unknown = np.full((2, len(queries)), np.nan)  # 0 — с роднёй, 1 — без родни

    for train_idx, test_idx in splitter.split(queries, groups=wines):
        train_groups = [by_query[queries[i]] for i in train_idx]
        confidence_rows, labels, weights = training_rows(
            train_groups, args.augment_unknown, args.live_weight
        )
        confidence = make_confidence_model().fit(
            np.asarray(confidence_rows), np.asarray(labels), sample_weight=np.asarray(weights)
        )
        rank_pool, _ = ranking_pool(train_groups, args.live_weight)
        ranker = make_ranker().fit(rank_pool)

        def select(candidates: list[PairFeatures], ranker=ranker, confidence=confidence) -> tuple:
            values = np.asarray(matrix(derive(candidates)))
            ranking = np.asarray(ranker.predict(values), dtype=float).ravel()
            confidence_raw = np.asarray(confidence.predict_proba(values)[:, 1], dtype=float)
            best = int(np.argmax(ranking))
            return best, confidence_raw[best], ranking[best]

        for i in test_idx:
            candidates = by_query[queries[i]]
            best, confidence_raw, rank_score = select(candidates)
            oof_raw[i] = confidence_raw
            oof_rank[i] = rank_score
            oof_correct[i] = bool(candidates[best].label)
            oof_item_ids[i] = candidates[best].item_id
            if not known[i]:
                # Живой незнакомец: его лучший кандидат — то, что сервис ответил бы мимо.
                continue

            # Незнакомое вино моделируем удалением правильного ответа. Важно сначала
            # переупорядочить список ranker'ом и только потом брать confidence победителя:
            # именно так работает production-пайплайн.
            without_true = [c for c in candidates if not c.label]
            if without_true:
                _, unknown_confidence, _ = select(without_true)
                oof_unknown[0, i] = unknown_confidence

            true_row = next((c for c in candidates if c.label), None)
            family = family_key(candidates[0].true_id, true_row)
            without_family = [c for c in candidates if family_key(c.item_id, c) != family]
            if without_family:
                _, unknown_confidence, _ = select(without_family)
                oof_unknown[1, i] = unknown_confidence

        # ``confidence`` остаётся classifier'ом для порога. Ranker обучен на парах и его
        # числа несопоставимы между запросами, поэтому калибровать их как вероятность нельзя.
        del confidence, ranker

    # Калибровка Платта: сырой выход бустера монотонно связан с правильностью, но числом
    # вероятности не является. Логистическая регрессия по одному признаку делает его таковым.
    #
    # Два обязательных условия, каждое из которых мы уже нарушали. Признак — логит сырой
    # оценки, а не она сама (иначе верх шкалы схлопывается, см. Decider.calibrate).
    # Регуляризация выключена большим C: штраф за величину коэффициента здесь ничему не мешает
    # переобучаться, зато прижимает наклон к нулю и сплющивает ту же шкалу ещё раз.
    # Живые незнакомцы участвуют в калибровке как отрицательные примеры: их лучший кандидат
    # неверен по построению, и это ровно тот случай, который калибровка должна прижать к нулю.
    calibrator = LogisticRegression(C=1e6).fit(
        logit(oof_raw).reshape(-1, 1), oof_correct.astype(int)
    )
    weight = float(calibrator.coef_[0][0])
    bias = float(calibrator.intercept_[0])

    # Сценарии незнакомого вина: два смоделированных (как раньше) и, если в признаках есть
    # живые незнакомые запросы, третий — по ним. Он единственный, где незнакомец настоящий:
    # чужая бутылка, снятая телефоном, а не каталог с выброшенной строкой.
    scenarios = {
        "family": oof_unknown[0][~np.isnan(oof_unknown[0])],
        "nofamily": oof_unknown[1][~np.isnan(oof_unknown[1])],
    }
    if (~known).any():
        scenarios["real"] = oof_raw[~known]
    elif args.scenario == "real":
        raise SystemExit("в признаках нет живых незнакомых запросов (true_id = unknown:...)")

    auroc = {
        name: float(
            roc_auc_score(
                np.r_[np.ones(int(known.sum())), np.zeros(len(row))], np.r_[oof_raw[known], row]
            )
        )
        for name, row in scenarios.items()
    }

    # Порог считаем той же формулой, которой его потом применит сервис: пустышка без бустера
    # умеет калибровать, а больше здесь ничего и не нужно.
    probe = Decider(booster=None, calib_weight=weight, calib_bias=bias, threshold=0.0)
    calibrated_known = probe.calibrate(oof_raw[known])
    known_correct = oof_correct[known]

    print(f"\ntop-1 (out-of-fold), знакомые: {known_correct.mean():.3f}")
    for group in sorted(set(groups_of.values())):
        mask = known & np.array([groups_of[q] == group for q in queries])
        if mask.any():
            print(f"  {group:12s} {oof_correct[mask].mean():.3f}  (запросов {int(mask.sum())})")
    for name, value in auroc.items():
        print(f"AUROC отказа, {name:9s} {value:.3f}")

    picked = None
    for name, row in scenarios.items():
        current = pick_threshold(calibrated_known, known_correct, probe.calibrate(row), args.budget)
        if name == args.scenario:
            picked = current
        print(f"\nсценарий «{name}», бюджет ошибок на незнакомых {args.budget:.2f}:")
        if current["coverage"] == 0.0:
            print(
                "  рабочего порога нет: уложиться в бюджет удаётся только отказом от всех "
                "ответов, то есть незнакомое вино набирает столько же, сколько верное"
            )
            continue
        print(
            f"  порог {current['threshold']:.2f} -> отвечаем на {current['coverage']:.3f} "
            f"знакомых вин, верно из них {current['precision']:.3f}"
        )
        print(f"  ошибочно отвечаем на {current['false_answer_rate']:.3f} незнакомых")
        for share in (0.05, 0.2, 0.5):
            print(
                f"  точность среди отвеченных при доле незнакомых {share:.0%}: "
                f"{precision_at_prior(current, share):.3f}"
            )

    final_confidence = make_confidence_model()
    rows, labels, weights = training_rows(
        [by_query[q] for q in queries], args.augment_unknown, args.live_weight
    )
    final_confidence.fit(np.asarray(rows), np.asarray(labels), sample_weight=np.asarray(weights))
    final_rank_pool, ranker_stats = ranking_pool(
        [by_query[q] for q in queries], args.live_weight
    )
    final_ranker = make_ranker().fit(final_rank_pool)

    decider = Decider(
        booster=final_confidence,
        calib_weight=weight,
        calib_bias=bias,
        threshold=picked["threshold"],
        feature_names=FEATURE_NAMES,
        ranker=final_ranker,
        meta={
            "trained_on": str(args.features),
            "holdout_isolated": True,
            "queries": len(queries),
            "unknown_queries": int((~known).sum()),
            "wines": len(set(wines)),
            "oof_top1": float(known_correct.mean()),
            "family_map": str(args.family_map) if args.family_map else None,
            "auroc_unknown": auroc,
            "scenario": args.scenario,
            "budget": args.budget,
            "augment_unknown": args.augment_unknown,
            "live_weight": args.live_weight,
            "only_groups": args.only_groups,
            "coverage": picked["coverage"],
            "precision": picked["precision"],
            "false_answer_rate": picked["false_answer_rate"],
            "ranking": {
                "objective": "PairLogitPairwise",
                "hard_negatives": "same_family_or_top5_visual_text_rrf",
                **ranker_stats,
            },
            "ranker_feature_importance": dict(
                zip(
                    FEATURE_NAMES,
                    map(
                        float,
                        final_ranker.get_feature_importance(type="PredictionValuesChange"),
                    ),
                    strict=True,
                )
            ),
            "confidence_feature_importance": dict(
                zip(
                    FEATURE_NAMES,
                    map(float, final_confidence.get_feature_importance()),
                    strict=True,
                )
            ),
        },
    )
    decider.save(args.out)
    probabilities = decider.calibrate(oof_raw)
    with (args.out / "oof_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, query in enumerate(queries):
            stream.write(
                json.dumps(
                    {
                        "query": query,
                        "wine": wines[i],
                        "known": bool(known[i]),
                        "best_id": oof_item_ids[i],
                        "correct": bool(oof_correct[i]),
                        "confidence_raw": float(oof_raw[i]),
                        "rank_score": float(oof_rank[i]),
                        "calibrated": float(probabilities[i]),
                        "note": "OOF base model; calibration/threshold fitted on these OOF scores",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nмодель сохранена: {args.out}")


if __name__ == "__main__":
    main()
