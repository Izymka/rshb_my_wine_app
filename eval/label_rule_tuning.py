"""Подобрать границы правила «этикетка подтверждена» на train и записать их в meta решающего слоя.

    uv run python eval/label_rule_tuning.py \\
        --features eval/results/features_platform_sq_v3_v6.jsonl \\
        --decider models/decider_platform_sq_v3_v6 --write

Правило (decide/guard.py `label_confirmed`) отвечает ниже порога, когда текст однозначно
подтверждает карточку: так вино той же этикетки другого года или без года в каталоге не
теряется из-за неуверенности решающего слоя (правило продукта 27.09.2026). Цена — ложные
ответы на незнакомых винах. Границы выбираются по живым кадрам train и OOF-вероятностям
решающего слоя (`oof_predictions.jsonl`): больше всего добавленных верных ответов на знакомых
при добавленных ложных ответах на незнакомых не выше `--extra-false` (доля). Тест не трогается.
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from wine_scanner.catalog import is_holdout
from wine_scanner.decide import PairFeatures, derive
from wine_scanner.decide.guard import label_confirmed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--decider", type=Path, required=True)
    parser.add_argument("--extra-false", type=float, default=0.02)
    parser.add_argument("--write", action="store_true", help="записать label_rule в meta.json")
    args = parser.parse_args()

    meta_path = args.decider / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = meta["threshold"]
    oof = {}
    for line in (args.decider / "oof_predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        oof[row["query"]] = row

    grouped: dict[str, list[PairFeatures]] = defaultdict(list)
    with args.features.open(encoding="utf-8") as fh:
        for line in fh:
            row = PairFeatures.from_dict(json.loads(line))
            if row.group.startswith("live-") and not is_holdout(row.query) and row.query in oof:
                grouped[row.query].append(row)

    frames = []
    for query, rows in grouped.items():
        pred = oof[query]
        top = next(r for r in derive(rows) if r["item_id"] == pred["best_id"])
        frames.append({
            "known": pred["known"], "correct": pred["correct"],
            "answered": pred["calibrated"] >= threshold, "features": top,
        })
    known = [f for f in frames if f["known"]]
    unknown = [f for f in frames if not f["known"]]
    base_ok = sum(f["correct"] and f["answered"] for f in known)
    base_false = sum(f["answered"] for f in unknown)
    print(f"живых кадров train: знакомых {len(known)}, незнакомых {len(unknown)}; порог "
          f"{threshold:.2f}: верных ответов {base_ok}, ложных на незнакомых {base_false}")

    best = None
    grid = itertools.product((0, 1), (0.34, 0.5, 0.67, 0.8, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9, 1.01))
    for max_rank, disc, cover in grid:
        rule = {"max_txt_rank": max_rank, "min_disc_hit": disc, "min_name_cover": cover}
        fired = [f for f in frames if not f["answered"] and label_confirmed(f["features"], rule)]
        gain = sum(f["known"] and f["correct"] for f in fired)
        wrong = sum(f["known"] and not f["correct"] for f in fired)
        false = sum(not f["known"] for f in fired)
        ok = false <= args.extra_false * len(unknown)
        key = (ok, gain - wrong, -false)
        if best is None or key > best[0]:
            best = (key, rule, gain, wrong, false)
    _, rule, gain, wrong, false = best
    print(f"правило {rule}: +{gain} верных, +{wrong} неверных на знакомых, "
          f"+{false} ложных на незнакомых (из {len(unknown)})")
    if args.write:
        meta["label_rule"] = {**rule, "tuned_on": str(args.features), "extra_false": false,
                              "gain_known": gain, "wrong_known": wrong}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"записано в {meta_path}")


if __name__ == "__main__":
    main()
