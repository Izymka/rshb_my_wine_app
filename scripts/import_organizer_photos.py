"""Кадры организаторов (organizer_100) — в `data/train` как живые кадры для обучения и порога.

    uv run python scripts/import_organizer_photos.py            # импорт + таблица проверки
    uv run python scripts/import_organizer_photos.py --dry-run  # только таблица

Решение 23.09.2026: 100 фотографий организаторов — единственный заметный запас живых кадров
знакомых вин (в `data/train` их было 6), без которых решающий слой учится отличать синтетику
от живого кадра, а не знакомое вино от незнакомого (ноутбук 06). Единственным holdout остаётся
`data/test`; сверка organizer_100 после импорта перестаёт быть независимой оценкой.

Разметка — `eval/results/organizer_100_manual_slugs.csv`. Кадр попадает в train, только если
про него нечего спорить:

* **знакомое вино** — ручной slug есть в каталоге, вина нет в `data/test`, и в примечании нет
  признаков неоднозначной разметки (другой дизайн этикетки, несколько slug на одну этикетку);
* **незнакомое вино** — slug пуст, но вино опознано по названию, и либо разметчик явно
  записал, чего нет в каталоге (примечание хотя бы у одного кадра этого вина), либо ни одна
  карточка каталога не похожа на название (`token_set_ratio` < `--catalog-match`). Похожая
  карточка без примечания — `check`: кадр ждёт человека и в train не идёт;
* вино не должно совпадать с незнакомцем из `data/test` (сравнение транслитерированного
  названия с `wine_id`), а файл — ни с одним файлом теста по sha256.

Решение по каждому из 100 кадров — в `eval/results/organizer_100_import_review.csv`.
Первый запуск её создаёт; дальше она — **источник правды**: человек правит `slug` и `note`,
и повторный запуск берёт решения из неё, ничего в ней не переписывая (`--regenerate` —
собрать заново из ручной разметки). Проверки на тест при этом повторяются всегда: правка
таблицы не может провести в train вино или файл из `data/test`.
"""

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from wine_scanner.catalog import TEST_MANIFEST, TRAIN_MANIFEST, TRAIN_ROOT
from wine_scanner.ocr.normalize import normalize

SOURCE = Path("data/eval/organizer_100")
LABELS = Path("eval/results/organizer_100_manual_slugs.csv")
CATALOG = Path("data/catalog/catalog.csv")
REVIEW = Path("eval/results/organizer_100_import_review.csv")
FIELDS = ("path", "wine_id", "true_slug", "group", "source", "note")
# Примечания, после которых правильный slug — предмет спора, а не факт.
AMBIGUOUS = ("другой дизайн", "slug с одной этикеткой", "две карточки", "дубл", "см. #")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", normalize(text).replace(" ", "-")).strip("-")


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def best_match(query: str, choices: dict[str, str]) -> tuple[str, float]:
    folded = normalize(query)
    scored = ((key, fuzz.token_set_ratio(folded, text)) for key, text in choices.items())
    return max(scored, key=lambda pair: pair[1], default=("", 0.0))


def decide(row: dict, context: dict, args) -> dict:
    """Решение по одному кадру и его причина."""
    slug, wine, note = row["slug"].strip(), row["wine"].strip(), row["note"].strip()
    source = SOURCE / row["image_path"]
    result = {"n": row["n"], "image_path": row["image_path"], "wine": wine, "slug": slug,
              "note": note, "wine_id": "", "decision": "", "reason": "",
              "best_catalog_slug": "", "best_catalog_score": "", "test_match": ""}
    if not source.exists():
        return {**result, "decision": "skip", "reason": "нет файла"}
    if sha256(source) in context["test_hashes"]:
        return {**result, "decision": "skip", "reason": "тот же файл есть в data/test"}

    if slug:
        result["wine_id"] = slug
        if slug not in context["catalog"]:
            return {**result, "decision": "skip", "reason": "slug нет в каталоге"}
        if slug in context["test_known"]:
            return {**result, "decision": "skip", "reason": "вино есть в data/test",
                    "test_match": slug}
        if any(marker in note for marker in AMBIGUOUS):
            return {**result, "decision": "skip", "reason": "неоднозначная разметка"}
        return {**result, "decision": "train-known", "reason": "ручной slug"}

    if not wine:
        return {**result, "decision": "skip", "reason": "вино не опознано"}
    result["wine_id"] = f"unknown-org-{slugify(wine)}"
    test_id, test_score = best_match(wine, context["test_unknown"])
    if test_score >= args.test_match:
        return {**result, "decision": "skip", "reason": "незнакомец из data/test",
                "test_match": f"{test_id} ({test_score:.0f})"}
    catalog_slug, catalog_score = best_match(wine, context["catalog"])
    result.update(best_catalog_slug=catalog_slug, best_catalog_score=f"{catalog_score:.0f}")
    if wine in context["noted_unknown"]:
        return {**result, "decision": "train-unknown", "reason": "разметчик: в каталоге нет"}
    if catalog_score >= args.catalog_match:
        return {**result, "decision": "check", "reason": "похожая карточка каталога"}
    return {**result, "decision": "train-unknown", "reason": "похожих карточек нет"}


TEST_REASONS = {"вино есть в data/test", "тот же файл есть в data/test",
                "незнакомец из data/test", "нет файла"}
# Слова, которыми проверяющий снимает спор о разметке: «можно использовать», «slug верный».
APPROVED = ("использ", "верн", "провер")


def reviewed(row: dict, context: dict, args) -> dict:
    """Решение по строке, которую уже просмотрел человек."""
    slug, wine, note = row["slug"].strip(), row["wine"].strip(), row["note"].strip()
    result = {**row}
    source = SOURCE / row["image_path"]
    if row["reason"] in TEST_REASONS or not source.exists():
        return {**result, "decision": "skip"}
    if sha256(source) in context["test_hashes"]:
        return {**result, "decision": "skip", "reason": "тот же файл есть в data/test"}
    if slug:
        result["wine_id"] = slug
        if slug not in context["catalog"]:
            return {**result, "decision": "skip", "reason": "slug нет в каталоге"}
        if slug in context["test_known"]:
            return {**result, "decision": "skip", "reason": "вино есть в data/test"}
        approved = slug in context["approved_slugs"]
        if any(m in note for m in AMBIGUOUS) and not approved:
            return {**result, "decision": "skip", "reason": "неоднозначная разметка"}
        return {**result, "decision": "train-known"}
    # Строку без примечания, которую первый запуск уже принял как незнакомое вино, человек
    # просмотрел и оставил как есть — решение в силе.
    if not note and row["decision"] != "train-unknown":
        return {**result, "decision": "skip", "reason": "не проверено человеком"}
    test_id, test_score = best_match(wine, context["test_unknown"])
    if test_score >= args.test_match:
        return {**result, "decision": "skip", "reason": "незнакомец из data/test"}
    result["wine_id"] = row["wine_id"] or f"unknown-org-{slugify(wine)}"
    return {**result, "decision": "train-unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="только таблица проверки")
    parser.add_argument("--regenerate", action="store_true",
                        help="пересобрать таблицу проверки из ручной разметки (правки человека "
                        "в ней будут потеряны)")
    parser.add_argument("--catalog-match", type=float, default=85.0,
                        help="token_set_ratio, с которого название считается похожим на карточку")
    parser.add_argument("--test-match", type=float, default=80.0,
                        help="token_set_ratio, с которого вино считается незнакомцем из теста")
    args = parser.parse_args()

    catalog = pd.read_csv(CATALOG, dtype=str, keep_default_na=False)
    test = pd.read_csv(TEST_MANIFEST, dtype=str, keep_default_na=False)
    context = {
        "catalog": {r.slug: normalize(f"{r.winery} {r.name}") for r in catalog.itertuples()},
        "test_known": set(test.loc[test.true_slug != "", "true_slug"]),
        "test_unknown": {
            w: normalize(w.removeprefix("unknown-").replace("-", " "))
            for w in test.loc[test.true_slug == "", "wine_id"].unique()
        },
        "test_hashes": {sha256(Path(p)) for p in test.path if Path(p).exists()},
    }
    with LABELS.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    # Примечание разметчик пишет один раз на вино, а не на каждый его кадр.
    context["noted_unknown"] = {
        row["wine"].strip() for row in labels if not row["slug"].strip() and row["note"].strip()
    }
    if REVIEW.exists() and not args.regenerate:
        with REVIEW.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        # Одобрение slug относится к вину, а не к кадру: проверяющий пишет его в note или в
        # reason одного из кадров, и оно действует на все кадры с этим slug.
        context["approved_slugs"] = {
            row["slug"].strip() for row in rows
            if row["slug"].strip()
            and any(a in f"{row['note']} {row['reason']}".lower() for a in APPROVED)
        }
        decisions = [reviewed(row, context, args) for row in rows]
        print(f"решения из просмотренной таблицы: {REVIEW} (файл не меняется)")
    else:
        decisions = [decide(row, context, args) for row in labels]
        REVIEW.parent.mkdir(parents=True, exist_ok=True)
        with REVIEW.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
            writer.writeheader()
            writer.writerows(decisions)

    summary = pd.DataFrame(decisions).groupby(["decision", "reason"]).size()
    print(summary.to_string())
    print(f"таблица проверки: {REVIEW}")
    if args.dry_run:
        return

    # Строки organizer собираются заново: после просмотра кадр мог сменить метку.
    manifest = [row for row in read_manifest(TRAIN_MANIFEST) if row["source"] != "organizer"]
    known_paths = {row["path"] for row in manifest}
    added = 0
    for d in decisions:
        if not d["decision"].startswith("train-"):
            continue
        target = TRAIN_ROOT / d["wine_id"] / d["image_path"]
        path = target.as_posix()
        if path in known_paths:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / d["image_path"], target)
        known = d["decision"] == "train-known"
        manifest.append({
            "path": path,
            "wine_id": d["wine_id"],
            "true_slug": d["slug"] if known else "",
            "group": "organizer" if known else "unknown-organizer",
            "source": "organizer",
            "note": f"organizer_100 #{d['n']}" + (f": {d['note']}" if d["note"] else ""),
        })
        known_paths.add(path)
        added += 1
    with TRAIN_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"кадров organizer в {TRAIN_MANIFEST}: {added}")


if __name__ == "__main__":
    main()
