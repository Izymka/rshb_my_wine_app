"""Признаки решающего слоя на каталоге платформы — через сам пайплайн.

    uv run python scripts/build_platform_features.py

От scripts/build_decision_features.py отличается двумя вещами. Во-первых, каталог не
пересчитывается: берётся готовый индекс платформы с whitening и дескрипторами, и признаки
считает тот же WineScanner, что отвечает в сервисе, — обучение видит ровно то, что видит
пользователь, включая окно ре-ранкинга и порядок слияния веток. Во-вторых, запросы приходят
из трёх источников, и у части из них верного ответа нет по-настоящему:

- псевдофото из вырезок (`data/synthetic/manifest.csv`) — положительные примеры;
- свой набор (`data/own`): вина, которых нет в каталоге платформы, идут как живые незнакомые
  (`true_id` = `unknown:<папка>`), а те, что есть, — как положительные по таблице соответствий;
- публичные кадры организаторов (`data/eval`): Массандра известна, два других незнакомы.

Незнакомый запрос — это тот, у которого `true_id` начинается с `unknown:`. Метка у всех его
кандидатов нулевая, и train_decider.py использует его для порога вместо симуляции.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wine_scanner.catalog import EVAL_ROOT, OWN_ROOT, load_own
from wine_scanner.decide import PairFeatures
from wine_scanner.embed import load_image
from wine_scanner.pipeline import WineScanner

OUT_PATH = Path("eval/results/features_platform.jsonl")
SYNTHETIC_MANIFEST = Path("data/synthetic/manifest.csv")

# Свой набор -> slug платформы. Проверено глазами 16.09.2026: Chateau Tamagne красное 2025 —
# тот же дизайн, что у карточки 2023 года (в каталоге только она); Inkerman Ркацители в каталоге
# отсутствует (под «Инкерман полусладкое» лежит игристое), Лабра Ашамта — тоже. Всё остальное
# в своём наборе — импорт, которого на платформе нет по определению.
OWN_TO_PLATFORM = {
    "Chateau_Tamagne_красное_2025": "kuban-vino-shato-tamane-ruzh-2023-tsvaygelt-krasnoe-suhoe-12",
}
EVAL_TRUTH = {
    "096ca74e.jpg": None,  # Aristov Donum XXIV — нет в дампе
    "019c68d0.jpg": None,  # Табия Пино Нуар 2025 — нет в дампе
    "02eef911.webp": "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16",
}


def collect_queries(
    synthetic: Path, own_root: Path, eval_root: Path, sources: set[str]
) -> list[dict]:
    rows: list[dict] = []
    if "synthetic" in sources and synthetic.exists():
        for r in pd.read_csv(synthetic).itertuples(index=False):
            rows.append({"path": r.path, "true_id": r.true_id, "group": r.group})
    _, own_queries = load_own(own_root) if "own" in sources else (None, [])
    for q in own_queries:
        wine = q.path.parent.name
        true_id = OWN_TO_PLATFORM.get(wine, f"unknown:{wine}")
        rows.append({"path": str(q.path), "true_id": true_id, "group": f"own-{q.group}"})
    for name, slug in EVAL_TRUTH.items() if "eval" in sources else ():
        rows.append(
            {
                "path": str(eval_root / "queries" / name),
                "true_id": slug or f"unknown:eval-{Path(name).stem}",
                "group": "eval",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("models/index_platform"))
    parser.add_argument("--decider", type=Path, default=Path("models/decider"))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sources",
        default="synthetic,own,eval",
        help="какие источники запросов брать, через запятую",
    )
    parser.add_argument(
        "--keep-from",
        type=Path,
        default=None,
        help="перенести из этого файла признаков запросы источников, которые не пересчитываются",
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = collect_queries(args.synthetic, OWN_ROOT, EVAL_ROOT, sources)
    if args.limit:
        queries = queries[: args.limit]
    unknown = sum(q["true_id"].startswith("unknown:") for q in queries)
    print(f"запросов: {len(queries)}, из них незнакомых: {unknown}")

    # Решающий слой здесь нужен только чтобы пайплайн собрался: его оценки не пишутся,
    # в файл идут сырые признаки пар.
    scanner = WineScanner(index_dir=args.index, decider_dir=args.decider, ocr_cache=True)

    kept: list[str] = []
    if args.keep_from is not None and args.keep_from.exists():
        prefixes = {"synthetic": "synthetic", "own": "own-", "eval": "eval"}
        skip = tuple(prefixes[s] for s in sources)
        for line in args.keep_from.read_text(encoding="utf-8").splitlines():
            if not json.loads(line)["group"].startswith(skip):
                kept.append(line)
        print(f"перенесено пар из {args.keep_from}: {len(kept)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        for query in tqdm(queries, desc="признаки"):
            result = scanner.identify(load_image(query["path"]), image_key=query["path"])
            for candidate in result.candidates:
                row = PairFeatures.from_dict(candidate.features)
                row.query, row.true_id, row.group = query["path"], query["true_id"], query["group"]
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                written += 1
    print(f"записано пар: {written} в {args.out}")


if __name__ == "__main__":
    main()
