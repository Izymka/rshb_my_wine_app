"""The JSONL writer must remain valid when separate workers share a directory."""

import json
from concurrent.futures import ThreadPoolExecutor

from api.eval_log import EvalLog


def test_concurrent_writers_and_reverse_order(tmp_path):
    writers = [EvalLog(tmp_path), EvalLog(tmp_path)]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda number: writers[number % 2].submit({"id": f"{number:032x}"}, b"photo"),
                    range(40),
                )
            )
        for writer in writers:
            writer.close()
        lines = (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 40
        assert len({json.loads(line)["id"] for line in lines}) == 40
        assert [row["id"] for row in writers[0].recent(2)] == [
            json.loads(line)["id"] for line in reversed(lines[-2:])
        ]
        assert len(writers[0].recent(2, offset=2)) == 2
        assert len(list((tmp_path / "images").iterdir())) == 40
    finally:
        for writer in writers:
            writer.close()
