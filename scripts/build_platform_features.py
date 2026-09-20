"""Признаки решающего слоя на каталоге платформы — через сам пайплайн.

    uv run python scripts/build_platform_features.py

От scripts/build_decision_features.py отличается двумя вещами. Во-первых, каталог не
пересчитывается: берётся готовый индекс платформы с whitening и дескрипторами, и признаки
считает тот же WineScanner, что отвечает в сервисе, — обучение видит ровно то, что видит
пользователь, включая окно ре-ранкинга и порядок слияния веток. Во-вторых, запросы приходят
из трёх источников, и у части из них верного ответа нет по-настоящему:

- псевдофото из вырезок (`data/synthetic/manifest.csv`) — положительные примеры;
- обучающий манифест `data/train/manifest.csv`: свой набор (импорт: незнакомцы, Chateau Tamagne —
  известное) и незнакомцы с российской полки, не вошедшие в тест;
- изолированный тест `data/test/manifest.csv` (`--sources ...,test`) — только чтобы посчитать
  признаки для бенчмарка; `train_decider.py` такие запросы выбрасывает по пути (`is_holdout`).

Незнакомый запрос — это тот, у которого `true_id` начинается с `unknown:`. Метка у всех его
кандидатов нулевая, и train_decider.py использует его для порога вместо симуляции.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wine_scanner.catalog import TEST_MANIFEST, TRAIN_MANIFEST, load_manifest
from wine_scanner.decide import PairFeatures
from wine_scanner.embed import load_image
from wine_scanner.pipeline import RetrievalOnlyDecider, WineScanner

OUT_PATH = Path("eval/results/features_platform.jsonl")
SYNTHETIC_MANIFEST = Path("data/synthetic/manifest.csv")

MANIFESTS = {"train": TRAIN_MANIFEST, "test": TEST_MANIFEST}


def collect_queries(synthetic: Path, sources: set[str]) -> list[dict]:
    rows: list[dict] = []
    if "synthetic" in sources and synthetic.exists():
        for r in pd.read_csv(synthetic).itertuples(index=False):
            rows.append({"path": r.path, "true_id": r.true_id, "group": r.group})
    # Кадры по манифестам: известные — позитивы, незнакомые (`unknown:`) — негативы. Свой набор
    # и публичные кадры организаторов в манифестах уже есть, отдельных источников для них нет.
    # Префикс `live-` у всех кадров манифестов: по нему train_decider.py даёт им вес
    # `--live-weight` против псевдофото, и старые файлы признаков размечены так же.
    for name, manifest in MANIFESTS.items():
        if name in sources and manifest.exists():
            for q in load_manifest(manifest):
                rows.append({"path": str(q.path), "true_id": q.true_id, "group": f"live-{q.group}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("models/index_platform"))
    parser.add_argument(
        "--decider",
        default="none",
        help="папка решающего слоя или `none`: оценки в файл не пишутся, нужен только порядок",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sources",
        default="synthetic,train,test",
        help="какие источники запросов брать, через запятую: synthetic, train "
        "(data/train/manifest.csv — свой набор и незнакомцы вне теста), test (data/test/manifest.csv, "
        "включая data/eval; только для бенчмарка, в обучение не попадает)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="дописать в существующий файл только те запросы, которых в нём ещё нет",
    )
    parser.add_argument(
        "--keep-from",
        type=Path,
        default=None,
        help="перенести из этого файла признаков запросы источников, которые не пересчитываются",
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = collect_queries(args.synthetic, sources)
    if args.limit:
        queries = queries[: args.limit]
    unknown = sum(q["true_id"].startswith("unknown:") for q in queries)
    print(f"запросов: {len(queries)}, из них незнакомых: {unknown}")

    # Решающий слой здесь нужен только чтобы пайплайн собрался: его оценки не пишутся,
    # в файл идут сырые признаки пар. По умолчанию — заглушка, чтобы признаки новой версии
    # можно было собрать, когда обученной модели под них ещё нет. Защита выключена: она
    # правит вероятность, а не признаки, и в обучении ей делать нечего.
    decider = RetrievalOnlyDecider() if args.decider == "none" else None
    scanner = WineScanner(
        index_dir=args.index,
        decider_dir=Path(args.decider if decider is None else "models/_none"),
        decider=decider,
        ocr_cache=True,
        crop_cache=Path("models/crop_cache"),
        guard="off",
    )

    kept: list[str] = []
    if args.keep_from is not None and args.keep_from.exists():
        prefixes = {"synthetic": "synthetic", "train": "live-", "test": "live-"}
        skip = tuple(prefixes[s] for s in sources)
        for line in args.keep_from.read_text(encoding="utf-8").splitlines():
            if not json.loads(line)["group"].startswith(skip):
                kept.append(line)
        print(f"перенесено пар из {args.keep_from}: {len(kept)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["query"])
        queries = [q for q in queries if q["path"] not in done]
        print(f"уже посчитано запросов: {len(done)}, осталось: {len(queries)}")
    written = 0
    with args.out.open("a" if args.resume else "w", encoding="utf-8") as fh:
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
