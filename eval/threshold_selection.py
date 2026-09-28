"""Как выбирать порог отказа устойчиво — сравнение правил на OOF живых кадров train.

    uv run python eval/threshold_selection.py --decider models/decider_platform_sq_v3_v7

Порог решающего слоя выбирался как наименьший, при котором ложные приёмы незнакомых на половине
P не выше бюджета 10 %, после калибровки Платта на половине K. По 50 разбиениям K/P он гулял от
0.12 до 0.77. Причины две: калибровка переподбирается на каждом разбиении, а кривая ложных
приёмов плоская между сырыми 0.90 и 0.99 — хвост дают ~14 вин-близнецов, и порядковая статистика
на границе бюджета прыгает через всё плато.

Здесь каждое правило прогоняется по бутстрепу винодельнями (единица — винодельня, как в K/P):
порог подбирается на выборке, а покрытие и ложные приёмы меряются на винодельнях, не попавших в
неё. Сравниваются:

* `budget-split` — действующее правило (Платт на K, бюджет на P);
* `budget-raw` — тот же бюджет, но по сырому выходу и по всем живым кадрам выборки;
* `utility` — максимум ожидаемой пользы: верный ответ +1, отказ 0, неверный ответ на знакомом
  −c_wrong, ответ на незнакомом −c_fa, доля незнакомых в потоке π.

Итоговый порог — медиана по бутстрепу (bagging): она устойчивее одной порядковой статистики.
Тест не используется; калибровка для перевода порога в вероятность — один Платт по всем живым.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_decider as TD  # noqa: E402

from wine_scanner.decide import Decider  # noqa: E402

GRID = np.round(np.concatenate([np.arange(0.0, 0.9, 0.01), np.arange(0.9, 1.0, 0.001)]), 4)


def load(decider: Path, features: Path) -> dict:
    """OOF живых кадров: сырой выход, верность, знакомость и ключ кластера (винодельня)."""
    group, family = {}, {}
    with features.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            group.setdefault(row["query"], row["group"])
            if row.get("family") and row["item_id"] == row["true_id"]:
                family[row["query"]] = row["family"]
    rows = [
        json.loads(line)
        for line in (decider / "oof_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows = [r for r in rows if group.get(r["query"], "").startswith("live-")]
    keys = []
    for r in rows:
        if r["known"]:
            keys.append("known:" + family.get(r["query"], r["wine"]))
        else:
            words = [
                w
                for w in re.split(r"[-_]", r["wine"].removeprefix(TD.UNKNOWN_PREFIX).lower())
                if w and w not in {"unknown", "org", "import"}
            ]
            keys.append("unknown:" + (words[0] if words else r["wine"]))
    return {
        "raw": np.array([r["confidence_raw"] for r in rows]),
        "correct": np.array([r["correct"] for r in rows]),
        "known": np.array([r["known"] for r in rows]),
        "cluster": np.array(keys),
        "wine": [r["wine"] for r in rows],
        "group": [group[r["query"]] for r in rows],
    }


def curves(raw, correct, known) -> dict:
    """Покрытие, верные ответы и ложные приёмы на сетке сырых порогов."""
    answered = raw[None, :] >= GRID[:, None]
    k, u = known, ~known
    n_k, n_u = max(k.sum(), 1), max(u.sum(), 1)
    return {
        "coverage": answered[:, k].sum(1) / n_k,
        "right": (answered[:, k] & correct[k]).sum(1) / n_k,
        "wrong": (answered[:, k] & ~correct[k]).sum(1) / n_k,
        "fa": answered[:, u].sum(1) / n_u,
    }


def budget_raw(raw, correct, known, budget: float) -> float:
    fa = curves(raw, correct, known)["fa"]
    ok = np.flatnonzero(fa <= budget)
    return float(GRID[ok[0]]) if len(ok) else 1.0


def utility(c: dict, share: float, c_wrong: float, c_fa: float) -> np.ndarray:
    return (1 - share) * (c["right"] - c_wrong * c["wrong"]) - share * c_fa * c["fa"]


def best_utility(raw, correct, known, share, c_wrong, c_fa) -> float:
    u = utility(curves(raw, correct, known), share, c_wrong, c_fa)
    # Среди равных по пользе берём середину плато, а не его левый край.
    best = np.flatnonzero(u >= u.max() - 1e-9)
    return float(GRID[best[len(best) // 2]])


def budget_split(raw, correct, known, cluster, groups, budget, seed) -> float:
    """Действующее правило: Платт на K, бюджет на P; порог переводится в сырую шкалу."""
    wines = [c.split(":", 1)[1] for c in cluster]
    roles = TD.split_live(wines, known, groups, seed)
    split = TD.split_calibration(raw, correct, known, roles, budget)
    probe = Decider(
        booster=None, calib_weight=split["calib_weight"], calib_bias=split["calib_bias"],
        threshold=0.0,
    )
    above = np.flatnonzero(probe.calibrate(GRID) >= split["threshold"])
    return float(GRID[above[0]]) if len(above) else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decider", type=Path, default=Path("models/decider_platform_sq_v3_v7"))
    parser.add_argument(
        "--features", type=Path, default=Path("eval/results/features_platform_sq_v3_v7.jsonl")
    )
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--share", type=float, default=0.2, help="доля незнакомых вин в потоке")
    parser.add_argument("--c-wrong", type=float, default=2.0, help="цена неверного ответа")
    parser.add_argument("--c-fa", type=float, default=2.0, help="цена ответа на незнакомое")
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("eval/results/threshold_selection.json"))
    parser.add_argument(
        "--apply",
        choices=["budget-split", "budget-raw", "utility"],
        default=None,
        help="записать порог этого правила (медиану по бутстрепу) и калибровку по всем живым "
        "кадрам в meta.json решающего слоя",
    )
    args = parser.parse_args()

    data = load(args.decider, args.features)
    raw, correct, known, cluster = data["raw"], data["correct"], data["known"], data["cluster"]
    groups = np.array(data["group"])
    clusters = np.unique(cluster)
    print(
        f"живых кадров {len(raw)}: знакомых {known.sum()} (верных {correct[known].sum()}), "
        f"незнакомых {(~known).sum()}; винодельни {len(clusters)} "
        f"(знакомых {sum(c.startswith('known:') for c in clusters)}, "
        f"незнакомых {sum(c.startswith('unknown:') for c in clusters)})"
    )

    weight, bias = TD.calibrate_on(raw, correct)
    probe = Decider(booster=None, calib_weight=weight, calib_bias=bias, threshold=0.0)

    rules = {
        "budget-split": lambda r, c, k, cl, g, i: budget_split(
            r, c, k, cl, g, args.budget, args.seed + i
        ),
        "budget-raw": lambda r, c, k, cl, g, i: budget_raw(r, c, k, args.budget),
        "utility": lambda r, c, k, cl, g, i: best_utility(
            r, c, k, args.share, args.c_wrong, args.c_fa
        ),
    }
    rng = np.random.default_rng(args.seed)
    index_of = defaultdict(list)
    for i, c in enumerate(cluster):
        index_of[c].append(i)
    report = {}
    for name, rule in rules.items():
        picks, oob = [], defaultdict(list)
        for b in range(args.boot):
            drawn = rng.choice(clusters, len(clusters), replace=True)
            idx = np.concatenate([index_of[c] for c in drawn])
            rest = np.concatenate([index_of[c] for c in set(clusters) - set(drawn)])
            t = rule(raw[idx], correct[idx], known[idx], cluster[idx], list(groups[idx]), b)
            picks.append(t)
            r, c, k = raw[rest], correct[rest], known[rest]
            if k.any() and (~k).any():
                answered = r >= t
                oob["coverage"].append(float(answered[k].mean()))
                oob["right"].append(float((answered & c)[k].mean()))
                oob["fa"].append(float(answered[~k].mean()))
        picks = np.array(picks)
        bagged = float(np.median(picks))
        full = curves(raw, correct, known)
        at = int(np.searchsorted(GRID, bagged))
        report[name] = {
            "raw_threshold": {
                "median": bagged,
                "p05": float(np.quantile(picks, 0.05)),
                "p95": float(np.quantile(picks, 0.95)),
            },
            "probability_threshold": {
                "median": float(probe.calibrate(np.array([bagged]))[0]),
                "p05": float(probe.calibrate(np.array([np.quantile(picks, 0.05)]))[0]),
                "p95": float(probe.calibrate(np.array([np.quantile(picks, 0.95)]))[0]),
            },
            "oob": {
                k: {"median": float(np.median(v)), "p05": float(np.quantile(v, 0.05)),
                    "p95": float(np.quantile(v, 0.95))}
                for k, v in oob.items()
            },
            "all_live_at_median": {k: float(v[at]) for k, v in full.items()},
        }
        q = report[name]
        prob = q["probability_threshold"]
        print(
            f"\n{name}: сырой порог {bagged:.3f} [{q['raw_threshold']['p05']:.3f}; "
            f"{q['raw_threshold']['p95']:.3f}], вероятность {prob['median']:.2f} "
            f"[{prob['p05']:.2f}; {prob['p95']:.2f}]"
        )
        for k in ("coverage", "right", "fa"):
            v = q["oob"][k]
            print(f"  вне выборки {k:9s} {v['median']:.3f} [{v['p05']:.3f}; {v['p95']:.3f}]")
        a = q["all_live_at_median"]
        print(
            f"  на всех живых при медиане: покрытие {a['coverage']:.3f}, верно {a['right']:.3f}, "
            f"неверно {a['wrong']:.3f}, ложные приёмы {a['fa']:.3f}"
        )

    full = curves(raw, correct, known)
    args.out.write_text(
        json.dumps(
            {
                "decider": str(args.decider), "features": str(args.features),
                "budget": args.budget, "share": args.share, "c_wrong": args.c_wrong,
                "c_fa": args.c_fa, "boot": args.boot, "calib_weight": weight, "calib_bias": bias,
                "rules": report,
                "curve": {"grid": GRID.tolist(), **{k: v.tolist() for k, v in full.items()}},
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nзаписано: {args.out}")

    if args.apply:
        # Калибровка — один Платт по всем живым OOF, а не по половине K: на ней же порог
        # переведён в вероятность, и так он не зависит от случайного разбиения.
        meta_path = args.decider / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chosen = report[args.apply]
        meta.update(
            calib_weight=weight,
            calib_bias=bias,
            threshold=round(chosen["probability_threshold"]["median"], 3),
            calibration="live-all",
            threshold_rule={
                "rule": args.apply, "script": "eval/threshold_selection.py",
                "raw_threshold": chosen["raw_threshold"],
                "probability_threshold": chosen["probability_threshold"],
                "oob": chosen["oob"], "budget": args.budget, "share": args.share,
                "c_wrong": args.c_wrong, "c_fa": args.c_fa, "boot": args.boot,
            },
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"порог {meta['threshold']} ({args.apply}) записан в {meta_path}")


if __name__ == "__main__":
    main()
