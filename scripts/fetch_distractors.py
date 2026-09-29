"""Скачать отвлекающие карточки для честного замера отбора кандидатов.

    uv run python scripts/fetch_distractors.py --shards 1

Зачем это нужно. R@50 у нас 0.989, но измерен на каталоге в 1024 карточки — а у платформы их
будут десятки тысяч. Пока мы не знаем, как отбор кандидатов ведёт себя на таком объёме,
решение про Э9 (дообучение ArcFace) принимать не на чем: улучшать нечего ровно до тех пор,
пока R@50 не начнёт падать. Проверка стоит вечера скачивания против 20–40 GPU-часов вслепую.

Источник — `cipher982/wine-images-126k` с HuggingFace: 107 821 фотография винных бутылок,
CC BY 4.0, разложены по 11 тар-архивам ровно по 10 000 штук. Тары, а не parquet, выбраны
намеренно: тар читает стандартная библиотека, parquet потребовал бы pyarrow — новой
зависимости ради разового скрипта.

Важная оговорка про происхождение. Составитель выложил набор под CC BY 4.0, но сами снимки
собраны с витрин магазинов и правами составителя не покрыты. Для внутреннего замера этого
достаточно: мы ничего не публикуем и не обучаем на них модель, которая пойдёт в продукт.
Обучать на них нельзя без отдельной проверки прав.
"""

import argparse
import json
import tarfile
from pathlib import Path

REPO = "cipher982/wine-images-126k"
ROOT = Path("data/third-party datasets/wine-images-126k")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download(shard: int, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    name = f"shards/wine-images-{shard:05d}.tar"
    return Path(
        hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(root))
    )


def unpack(archive: Path, images_dir: Path) -> list[dict]:
    """Разложить архив по файлам и собрать метаданные.

    Имя вина в тарах лежит рядом с картинкой отдельным json-файлом (соглашение webdataset).
    Если его нет, обходимся идентификатором: для отвлекающей карточки имя нужно только
    текстовой ветке, а её вклад в R@50 мы считаем отдельно.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    saved: list[dict] = []

    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            stem = Path(member.name).stem

            if suffix == ".json":
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                try:
                    names[stem] = json.loads(payload.read()).get("wine_name", "")
                except (ValueError, UnicodeDecodeError):
                    continue
                continue

            if suffix not in IMAGE_SUFFIXES:
                continue

            target = images_dir / f"{stem}{suffix}"
            if not target.exists():
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                target.write_bytes(payload.read())
            saved.append({"item_id": stem, "image_path": str(target)})

    for row in saved:
        row["name"] = names.get(row["item_id"], "")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1, help="сколько архивов по 10 000 картинок")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--keep-archives", action="store_true", help="не удалять тары после распаковки"
    )
    args = parser.parse_args()

    catalog: list[dict] = []
    for shard in range(args.shards):
        archive = download(shard, args.root)
        print(f"архив {archive.name}: {archive.stat().st_size / 1e6:.0f} МБ")
        rows = unpack(archive, args.root / "images")
        catalog += rows
        print(f"  распаковано картинок: {len(rows)}")
        if not args.keep_archives:
            archive.unlink()

    index_path = args.root / "catalog.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in catalog:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with_names = sum(1 for row in catalog if row["name"])
    print(f"\nвсего карточек: {len(catalog)}, из них с названием: {with_names}")
    print(f"список: {index_path}")


if __name__ == "__main__":
    main()
