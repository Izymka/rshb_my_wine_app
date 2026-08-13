"""Э8: насколько хорошо читается год урожая.

    uv run python eval/vintage_benchmark.py

Считает по готовому eval/results/features.jsonl, потому что год собирается там же, где и
остальные признаки, — в одном дорогом прогоне пайплайна (scripts/build_decision_features.py).

Метрика блока — не одна доля, а три, и делить их обязательно:

* **верно** — год прочитан и совпал с каталогом;
* **молчим** — год не прочитан, показываем карточку и спрашиваем год у пользователя;
* **неверно** — прочитан чужой год.

Третья строка важнее первых двух вместе взятых. Молчание пользователь видит как честный
вопрос, а неверный год — как знание, и проверять его он не станет. Поэтому блок настроен
на отказ при любом сомнении, и следить надо именно за тем, чтобы третья строка не росла.

Отдельно считаются два источника: общий текст этикетки (бесплатен, он уже прочитан текстовой
веткой) и увеличенный по гомографии участок (стоит ещё одного вызова распознавания). Только
так видно, платим ли мы за второй проход не зря.

Вина без года (NV) — не балласт, а контрольная группа: на них правильный ответ «молчим»,
и любое прочитанное число здесь ложное срабатывание.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wine_scanner.catalog import load_own
from wine_scanner.decide import PairFeatures
from wine_scanner.vintage import catalog_years

FEATURES_PATH = Path("eval/results/features.jsonl")
RESULTS_PATH = Path("eval/results/vintage.json")


def truth() -> dict[str, set[int]]:
    """Годы, записанные в карточках своего набора. Это и есть разметка."""
    catalog, _ = load_own()
    return {item.item_id: catalog_years(item.payload) for item in catalog}


def first_rows(path: Path) -> list[PairFeatures]:
    """По одной строке на запрос: год — свойство кадра, а не пары «запрос — кандидат»."""
    seen: dict[str, PairFeatures] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = PairFeatures.from_dict(json.loads(line))
            seen.setdefault(row.query, row)
    return list(seen.values())


def tally(rows: list[PairFeatures], years_of: dict[str, set[int]], field: str) -> dict:
    """Разложить кадры на «верно / молчим / неверно» по одному источнику года."""
    counts = {"верно": 0, "молчим": 0, "неверно": 0}
    for row in rows:
        read = getattr(row, field)
        known = years_of.get(row.true_id, set())
        if not read:
            counts["молчим"] += 1
        elif read in known:
            counts["верно"] += 1
        else:
            counts["неверно"] += 1
    return counts


def show(title: str, counts: dict, total: int) -> None:
    print(
        f"  {title:<22}"
        + "  ".join(f"{k} {v:>3} ({v / total:.3f})" for k, v in counts.items() if total)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    years_of = truth()
    rows = first_rows(args.features)
    dated = [r for r in rows if years_of.get(r.true_id)]
    undated = [r for r in rows if not years_of.get(r.true_id)]
    print(f"кадров: {len(rows)}, из них с годом в каталоге: {len(dated)}, без года: {len(undated)}")

    print("\nвина с годом на этикетке:")
    sources = {
        "только текст": "vintage_text",
        "только увеличение": "vintage_zoom",
        "как в пайплайне": "vintage_year",
    }
    result = {}
    for title, field in sources.items():
        counts = tally(dated, years_of, field)
        show(title, counts, len(dated))
        result[field] = counts

    print("\nвина без года (NV) — здесь верен только отказ:")
    false_alarms = sum(1 for r in undated if r.vintage_year)
    print(
        f"  прочитан год у {false_alarms} из {len(undated)} кадров "
        f"({false_alarms / len(undated):.3f})"
        if undated
        else "  таких вин в наборе нет"
    )

    print("\nпо группам сложности (вина с годом):")
    by_group: dict[str, list[PairFeatures]] = defaultdict(list)
    for row in dated:
        by_group[row.group].append(row)
    for group in sorted(by_group):
        counts = tally(by_group[group], years_of, "vintage_year")
        show(group, counts, len(by_group[group]))

    # Что дало увеличение сверх текста: кадры, где текст промолчал, а участок прочитался.
    rescued = [r for r in dated if not r.vintage_text and r.vintage_zoom]
    good = sum(1 for r in rescued if r.vintage_zoom in years_of.get(r.true_id, set()))
    print(f"\nувеличение спасло кадров: {len(rescued)}, из них верно: {good}")
    for row in rescued:
        mark = "верно" if row.vintage_zoom in years_of.get(row.true_id, set()) else "НЕВЕРНО"
        print(f"  {mark:<8} {row.vintage_zoom}  {row.group:<8} {Path(row.query).parent.name[:40]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "frames": len(rows),
                "dated": len(dated),
                "undated": len(undated),
                "sources": result,
                "false_alarms_nv": false_alarms,
                "rescued_by_zoom": len(rescued),
                "rescued_correct": good,
                "by_group": {
                    group: tally(items, years_of, "vintage_year")
                    for group, items in by_group.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
