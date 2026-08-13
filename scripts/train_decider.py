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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from wine_scanner.decide import FEATURE_NAMES, Decider, PairFeatures, derive, logit, matrix

FEATURES_PATH = Path("eval/results/features.jsonl")
OUT_DIR = Path("models/decider")


def family_key(item_id: str) -> str:
    """Грубая группировка по производителю: первые два слова идентификатора."""
    return "_".join(item_id.split("_")[:2])


def load_queries(path: Path) -> dict[str, list[PairFeatures]]:
    grouped: dict[str, list[PairFeatures]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = PairFeatures.from_dict(json.loads(line))
            grouped[row.query].append(row)
    return grouped


def training_rows(
    groups: list[list[PairFeatures]], augment_unknown: bool
) -> tuple[list, list]:
    """Матрица и метки для обучения.

    `augment_unknown` добавляет к каждому запросу его же копию без правильного ответа. Смысл
    в том, что порог отказа обучается на данных, где отказываться не от чего: в обычной
    выборке верный кандидат всегда присутствует, и признаки вида «здесь нет верного ответа»
    ничего не улучшают, поэтому модель их не берёт. Замер: AUROC отказа 0.895 -> 0.945.
    """
    rows, labels = [], []
    for candidates in groups:
        derived = derive(candidates)
        rows += matrix(derived)
        labels += [r["label"] for r in derived]

        if not augment_unknown:
            continue
        without_true = [c for c in candidates if not c.label]
        if len(without_true) < len(candidates):
            unknown = derive(without_true)
            rows += matrix(unknown)
            labels += [0] * len(unknown)
    return rows, labels


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        # Положительных примеров в 25 раз меньше — по одному верному кандидату на запрос.
        class_weight="balanced",
        verbose=-1,
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
        choices=["family", "nofamily"],
        default="nofamily",
        help="как моделируется незнакомое вино: с роднёй в каталоге или без (по умолчанию)",
    )
    parser.add_argument(
        "--no-augment-unknown",
        dest="augment_unknown",
        action="store_false",
        help="учить только на запросах, где верный ответ есть в каталоге (как было до Э8)",
    )
    args = parser.parse_args()

    by_query = load_queries(args.features)
    queries = sorted(by_query)
    wines = [by_query[q][0].true_id for q in queries]
    pairs = sum(map(len, by_query.values()))
    print(f"запросов: {len(queries)}, вин: {len(set(wines))}, пар: {pairs}")

    splitter = GroupKFold(n_splits=args.folds)
    oof_raw = np.zeros(len(queries))
    oof_correct = np.zeros(len(queries), dtype=bool)
    oof_unknown = np.full((2, len(queries)), np.nan)  # 0 — с роднёй, 1 — без родни

    for train_idx, test_idx in splitter.split(queries, groups=wines):
        rows, labels = training_rows(
            [by_query[queries[i]] for i in train_idx], args.augment_unknown
        )
        model = make_model().fit(np.asarray(rows), np.asarray(labels))

        def predict(candidates: list[PairFeatures], model=model) -> np.ndarray:
            return model.predict_proba(np.asarray(matrix(derive(candidates))))[:, 1]

        for i in test_idx:
            candidates = by_query[queries[i]]
            probs = predict(candidates)
            best = int(np.argmax(probs))
            oof_raw[i] = probs[best]
            oof_correct[i] = bool(candidates[best].label)

            # Незнакомое вино моделируем удалением правильного ответа из кандидатов: для
            # решающего слоя это в точности ситуация «вина нет в каталоге».
            without_true = [c for c in candidates if not c.label]
            if without_true:
                oof_unknown[0, i] = predict(without_true).max()

            family = family_key(candidates[0].true_id)
            without_family = [c for c in candidates if family_key(c.item_id) != family]
            if without_family:
                oof_unknown[1, i] = predict(without_family).max()

    # Калибровка Платта: сырой выход бустера монотонно связан с правильностью, но числом
    # вероятности не является. Логистическая регрессия по одному признаку делает его таковым.
    #
    # Два обязательных условия, каждое из которых мы уже нарушали. Признак — логит сырой
    # оценки, а не она сама (иначе верх шкалы схлопывается, см. Decider.calibrate).
    # Регуляризация выключена большим C: штраф за величину коэффициента здесь ничему не мешает
    # переобучаться, зато прижимает наклон к нулю и сплющивает ту же шкалу ещё раз.
    calibrator = LogisticRegression(C=1e6).fit(
        logit(oof_raw).reshape(-1, 1), oof_correct.astype(int)
    )
    weight = float(calibrator.coef_[0][0])
    bias = float(calibrator.intercept_[0])

    scenario_index = 0 if args.scenario == "family" else 1
    unknown = oof_unknown[scenario_index]
    unknown = unknown[~np.isnan(unknown)]

    auroc = {
        name: float(
            roc_auc_score(
                np.r_[np.ones(len(queries)), np.zeros(len(row[~np.isnan(row)]))],
                np.r_[oof_raw, row[~np.isnan(row)]],
            )
        )
        for name, row in zip(("family", "nofamily"), oof_unknown, strict=True)
    }

    # Порог считаем той же формулой, которой его потом применит сервис: пустышка без бустера
    # умеет калибровать, а больше здесь ничего и не нужно.
    probe = Decider(booster=None, calib_weight=weight, calib_bias=bias, threshold=0.0)
    calibrated_known = probe.calibrate(oof_raw)

    print(f"\ntop-1 (out-of-fold):      {oof_correct.mean():.3f}")
    print(f"AUROC отказа, с роднёй:   {auroc['family']:.3f}")
    print(f"AUROC отказа, без родни:  {auroc['nofamily']:.3f}")

    picked = None
    for name, row in zip(("family", "nofamily"), oof_unknown, strict=True):
        row = row[~np.isnan(row)]
        current = pick_threshold(calibrated_known, oof_correct, probe.calibrate(row), args.budget)
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

    final = make_model()
    rows, labels = training_rows([by_query[q] for q in queries], args.augment_unknown)
    final.fit(np.asarray(rows), np.asarray(labels))

    decider = Decider(
        booster=final.booster_,
        calib_weight=weight,
        calib_bias=bias,
        threshold=picked["threshold"],
        feature_names=FEATURE_NAMES,
        meta={
            "trained_on": str(args.features),
            "queries": len(queries),
            "wines": len(set(wines)),
            "oof_top1": float(oof_correct.mean()),
            "auroc_unknown": auroc,
            "scenario": args.scenario,
            "budget": args.budget,
            "augment_unknown": args.augment_unknown,
            "coverage": picked["coverage"],
            "precision": picked["precision"],
            "false_answer_rate": picked["false_answer_rate"],
        },
    )
    decider.save(args.out)
    print(f"\nмодель сохранена: {args.out}")


if __name__ == "__main__":
    main()
