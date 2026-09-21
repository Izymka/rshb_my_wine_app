"""Synchronize a query manifest with files intentionally removed from a dataset.

The command only removes rows whose referenced files no longer exist.  It never discovers
or adds files automatically: every benchmark query must remain explicitly documented.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def sync(manifest: Path, report: Path, dry_run: bool = False) -> dict[str, object]:
    with manifest.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"В манифесте нет заголовка: {manifest}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [row for row in rows if Path(row["path"]).is_file()]
    removed = [row for row in rows if not Path(row["path"]).is_file()]
    result: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "before": len(rows),
        "after": len(kept),
        "removed_missing_files": removed,
        "dry_run": dry_run,
    }

    if not dry_run:
        temporary = manifest.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        temporary.replace(manifest)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Удалить из manifest строки отсутствующих файлов")
    parser.add_argument("--manifest", type=Path, default=Path("data/test/manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/derived/test_manifest_sync.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = sync(args.manifest, args.report, dry_run=args.dry_run)
    print(
        f"{result['manifest']}: before={result['before']}, after={result['after']}; "
        f"missing_rows={len(result['removed_missing_files'])}"
    )


if __name__ == "__main__":
    main()
