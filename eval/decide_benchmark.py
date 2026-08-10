"""Решающий слой: обучение, калибровка, кривая precision/coverage.

    uv run python eval/decide_benchmark.py

Работает на готовом eval/results/features.jsonl, поэтому считается за секунды.

Отвечает на два вопроса. Первый: даёт ли обученное взвешивание сигналов больше, чем нынешнее
фиксированное. Второй, более важный для продукта: умеем ли мы отказываться от ответа, когда
вина нет в каталоге.

Случаи «вина нет в базе» получаем без досъёмки — выбрасывая из кандидатов правильный ответ.
Для решающего слоя это в точности та же ситуация, что и отсутствие вина в индексе: все
кандидаты неверны, и модель обязана это заметить по слабости сигналов.
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

from wine_scanner.decide import FEATURE_NAMES, PairFeatures, derive, matrix

FEATURES_PATH = Path("eval/results/features.jsonl")
RESULTS_PATH = Path("eval/results/decide.json")


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


def train_model(x: np.ndarray, y: np.ndarray) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        # Положительных примеров в 25 раз меньше — по одному верному кандидату на запрос.
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(x, y)
    return model


def precision_coverage(scores: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Что будет, если отвечать только при уверенности выше порога.

    Покрытие — доля запросов, на которые мы вообще ответили. Точность — доля верных среди
    отвеченных. Ровно эта пара чисел нужна продукту: «когда мы отвечаем, мы правы в 95%
    случаев, и отвечаем в 80% случаев» звучит принципиально иначе, чем «accuracy 0.84».
    """
    table = []
    for threshold in np.arange(0.0, 1.0, 0.05):
        answered = scores >= threshold
        if answered.sum() == 0:
            continue
        table.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": float(answered.mean()),
                "precision": float(correct[answered].mean()),
            }
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    by_query = load_queries(args.features)
    queries = sorted(by_query)
    wines = [by_query[q][0].true_id for q in queries]
    groups_of = {q: by_query[q][0].group for q in queries}
    print(
        f"запросов: {len(queries)}, вин: {len(set(wines))}, "
        f"пар: {sum(map(len, by_query.values()))}"
    )

    # Разбиение по винам, а не по запросам: у одного вина шесть кадров, и если они разъедутся
    # между обучением и контролем, модель просто узнает вино, а метрика окажется завышенной.
    splitter = GroupKFold(n_splits=args.folds)
    oof_best_prob = np.zeros(len(queries))
    oof_best_correct = np.zeros(len(queries), dtype=bool)
    oof_unknown_prob = np.zeros(len(queries))
    oof_unknown_nofamily = np.full(len(queries), np.nan)
    baseline_correct = np.zeros(len(queries), dtype=bool)
    importance = np.zeros(len(FEATURE_NAMES))

    for train_idx, test_idx in splitter.split(queries, groups=wines):
        rows, labels = [], []
        for i in train_idx:
            derived = derive(by_query[queries[i]])
            rows += matrix(derived)
            labels += [r["label"] for r in derived]
        model = train_model(np.asarray(rows), np.asarray(labels))
        importance += model.feature_importances_

        for i in test_idx:
            candidates = by_query[queries[i]]
            derived = derive(candidates)
            probs = model.predict_proba(np.asarray(matrix(derived)))[:, 1]
            best = int(np.argmax(probs))
            oof_best_prob[i] = probs[best]
            oof_best_correct[i] = candidates[best].item_id == candidates[best].true_id
            # Нынешняя система отвечает первым кандидатом после слияния веток.
            baseline_correct[i] = any(c.rrf_rank == 0 and c.label for c in candidates)

            # Тот же запрос, но правильного ответа в каталоге нет.
            without_true = [c for c in candidates if not c.label]
            if without_true:
                unknown = derive(without_true)
                oof_unknown_prob[i] = model.predict_proba(np.asarray(matrix(unknown)))[:, 1].max()

            # Второй сценарий: нет не только самого вина, но и его родни — других вин того же
            # производителя. В нашем наборе 11 вин из 17 имеют почти близнеца, и это делает
            # отказ неестественно трудным: система обязана усомниться там, где визуальные
            # свидетельства честно сильные. В настоящем каталоге у случайного незнакомого
            # вина близнеца обычно нет.
            family = family_key(candidates[0].true_id)
            without_family = [c for c in candidates if family_key(c.item_id) != family]
            if without_family:
                unknown = derive(without_family)
                oof_unknown_nofamily[i] = model.predict_proba(
                    np.asarray(matrix(unknown))
                )[:, 1].max()

    print(f"\ntop-1 сейчас (RRF без обучения): {baseline_correct.mean():.3f}")
    print(f"top-1 с решающим слоем:          {oof_best_correct.mean():.3f}")

    print("\nпо группам сложности:")
    for group in sorted(set(groups_of.values())):
        mask = np.array([groups_of[q] == group for q in queries])
        print(
            f"  {group:<10} было {baseline_correct[mask].mean():.3f}  "
            f"стало {oof_best_correct[mask].mean():.3f}  (кадров {mask.sum()})"
        )

    # Отказ: вероятность верного ответа должна быть выше, когда вино в каталоге есть.
    auroc = roc_auc_score(
        np.r_[np.ones(len(queries)), np.zeros(len(queries))],
        np.r_[oof_best_prob, oof_unknown_prob],
    )
    print(f"\nAUROC отказа, близнец остался в каталоге: {auroc:.3f}")

    mask = ~np.isnan(oof_unknown_nofamily)
    auroc_nofamily = roc_auc_score(
        np.r_[np.ones(len(queries)), np.zeros(int(mask.sum()))],
        np.r_[oof_best_prob, oof_unknown_nofamily[mask]],
    )
    print(f"AUROC отказа, родни в каталоге нет:      {auroc_nofamily:.3f}")

    # Калибровка Платта: сырой выход модели — не вероятность, а монотонная ей величина.
    # Логистическая регрессия поверх него делает число, которое можно показывать пользователю.
    calibrator = LogisticRegression()
    calibrator.fit(oof_best_prob.reshape(-1, 1), oof_best_correct.astype(int))
    calibrated = calibrator.predict_proba(oof_best_prob.reshape(-1, 1))[:, 1]

    def curve(unknown_probs: np.ndarray) -> list[dict]:
        unknown_calibrated = calibrator.predict_proba(unknown_probs.reshape(-1, 1))[:, 1]
        return precision_coverage(
            np.r_[calibrated, unknown_calibrated],
            np.r_[oof_best_correct, np.zeros(len(unknown_probs), dtype=bool)],
        )

    table = curve(oof_unknown_prob)
    table_nofamily = curve(oof_unknown_nofamily[mask])

    # Доля незнакомых вин здесь 50% — столько же, сколько знакомых. В продукте она будет
    # заметно ниже, и точность при том же пороге окажется выше. Кривая нужна не как готовое
    # обещание, а как инструмент выбора порога.
    print(f"\n{'порог':>7}{'покрытие':>11}{'точность':>11}{'покрытие*':>12}{'точность*':>12}")
    print("-" * 53)
    by_threshold = {row["threshold"]: row for row in table_nofamily}
    for row in table:
        alt = by_threshold.get(row["threshold"])
        extra = (
            f"{alt['coverage']:>12.3f}{alt['precision']:>12.3f}" if alt else f"{'—':>12}{'—':>12}"
        )
        print(f"{row['threshold']:>7.2f}{row['coverage']:>11.3f}{row['precision']:>11.3f}{extra}")
    print("* — сценарий, где у незнакомого вина нет родни в каталоге")

    order = np.argsort(importance)[::-1]
    print("\nважность признаков:")
    for i in order[:8]:
        print(f"  {FEATURE_NAMES[i]:<16}{importance[i] / args.folds:>7.0f}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "baseline_top1": float(baseline_correct.mean()),
                "model_top1": float(oof_best_correct.mean()),
                "auroc_unknown": float(auroc),
                "auroc_unknown_nofamily": float(auroc_nofamily),
                "precision_coverage": table,
                "precision_coverage_nofamily": table_nofamily,
                "importance": dict(
                    zip(FEATURE_NAMES, (importance / args.folds).tolist(), strict=True)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
