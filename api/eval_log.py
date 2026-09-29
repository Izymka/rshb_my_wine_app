"""Persistent, non-blocking audit log for evaluation requests."""

import json
import logging
import mimetypes
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


@contextmanager
def file_lock(path: Path):
    """Interprocess lock shared by readers and writers (also on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class EvalLog:
    def __init__(self, root: Path):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-log")

    def submit(self, row: dict, image: bytes | None) -> None:
        future = self.executor.submit(self._write, row, image)

        def report_error(done):
            if error := done.exception():
                log.error("не удалось записать eval-лог", exc_info=error)

        future.add_done_callback(report_error)

    def _write(self, row: dict, image: bytes | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if image is not None:
            extension = mimetypes.guess_extension(row.get("image_content_type") or "") or ".image"
            if extension not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
                extension = ".image"
            name = f"{row['id']}{extension}"
            image_path = self.root / "images" / name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_path.with_suffix(".tmp")
            temporary.write_bytes(image)
            os.replace(temporary, image_path)
            row["image_url"] = f"/v1/eval/logs/images/{name}"
        line = json.dumps(
            row,
            ensure_ascii=False,
            default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
        ) + "\n"
        with file_lock(self.root / "requests.lock"):
            with (self.root / "requests.jsonl").open("a", encoding="utf-8") as output:
                output.write(line)
                output.flush()

    def recent(self, limit: int = 200, offset: int = 0) -> list[dict]:
        path = self.root / "requests.jsonl"
        with file_lock(self.root / "requests.lock"):
            if not path.exists():
                return []
            with path.open(encoding="utf-8") as source:
                lines = deque(source, maxlen=limit + offset)
        return [json.loads(line) for line in list(reversed(lines))[offset:]]

    def close(self) -> None:
        self.executor.shutdown(wait=True)
