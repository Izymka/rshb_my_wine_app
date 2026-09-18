"""Проверка провайдеров LLM: отвечает ли судья на картинку и сомелье на текст, за сколько.

Зачем. Ключи облаков живут своей жизнью: у ключа может не быть нужной области действия
(«Unknown api key»), у аккаунта — роли, у OpenAI — кредитов. Поднимать ради проверки весь
сервис долго; этот скрипт собирает клиентов ровно так, как их собирает сервис
(`chat_from_env`), и задаёт по одному вопросу: судье — кадр с этикеткой, сомелье — текст.

    uv run python scripts/check_llm.py                       # переменные из .env
    uv run python scripts/check_llm.py --image data/eval/queries/019c68d0.jpg
"""

import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image

from wine_scanner.llm import ChatError, chat_from_env, jpeg_bytes

DEFAULT_IMAGE = Path("data/eval/queries/019c68d0.jpg")


def load_env(path: Path) -> None:
    """Прочитать KEY=VALUE из .env, не перекрывая уже заданное в окружении."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def describe(client) -> str:
    clients = getattr(client, "clients", [client])
    return " → ".join(f"{c.name} ({getattr(c, 'model', '?')})" for c in clients)


def check(prefix: str, run) -> bool:
    try:
        client = chat_from_env(prefix)
    except ValueError as error:
        print(f"{prefix}: не настроен — {error}")
        return False
    print(f"{prefix}: {describe(client)}")
    started = time.perf_counter()
    try:
        answer = run(client)
    except ChatError as error:
        print(f"  ошибка: {error}")
        return False
    print(f"  {time.perf_counter() - started:.1f} с: {answer.strip()[:300]}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()
    load_env(args.env)

    ok = check(
        "WINE_LLM",
        lambda c: c.chat(
            [{"role": "user", "content": "Одним предложением: к чему подать сухое красное саперави?"}]
        ),
    )
    if args.image.exists():
        image = jpeg_bytes(Image.open(args.image))
        ok &= check(
            "WINE_VLM",
            lambda c: c.chat_with_image(
                'Что написано на этикетке? Ответь JSON вида {"read_text": "..."}',
                image,
                json_mode=True,
            ),
        )
    else:
        print(f"WINE_VLM: кадра {args.image} нет, картинку не проверяем")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
