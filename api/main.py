"""HTTP-обёртка над пайплайном.

    uv sync --extra api --extra rerank --extra ocr --extra decide
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

Слой намеренно тонкий: принять файлы, отдать результат, ничего не решать. Вся логика — в
wine_scanner.pipeline, и это не вопрос вкуса. Как только сервис начинает добавлять своё
(свой порог, свой отбор кандидатов, свою обработку картинки), метрики бенчмарка перестают
описывать то, что видит пользователь, и разойдутся они молча.

Модели поднимаются один раз на старте, до приёма запросов. Загрузка занимает десятки секунд,
и делать её лениво на первом запросе нельзя: этот запрос упрётся в таймаут, а если запросов
придёт несколько сразу, модели начнут грузиться параллельно и съедят память.
"""

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from wine_scanner.pipeline import WineScanner

# Телефон отдаёт HEIC, и без этой строки сервис будет падать ровно на тех снимках,
# которые пользователь делает чаще всего.
register_heif_opener()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wine_scanner.api")

MAX_FRAMES = 12
MAX_BYTES = 20 * 1024 * 1024
# Предел на весь запрос, а не только на файл. Без него двенадцать разрешённых файлов дают
# 240 МБ в памяти процесса, в котором рядом живут четыре модели.
MAX_TOTAL_BYTES = 60 * 1024 * 1024

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    state["scanner"] = WineScanner(
        index_dir=Path(os.environ.get("WINE_INDEX", "models/index")),
        decider_dir=Path(os.environ.get("WINE_DECIDER", "models/decider")),
    )
    state["load_seconds"] = time.perf_counter() - started
    yield
    state.clear()


app = FastAPI(
    title="Сканер вина",
    description="Поиск карточки вина по фотографии этикетки",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    """Сквозной номер запроса.

    Свой берём только если клиент не прислал: между телефоном и нами стоит ещё один сервис,
    и номер нужен, чтобы жалоба пользователя связалась с нашей строкой лога, а не искалась
    по времени.

    Стоит промежуточным слоем, а не в самом обработчике, потому что нужнее всего он там, где
    обработчик до ответа не дошёл: на отказе по размеру и на ошибке внутри пайплайна.
    """
    number = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = number
    response = await call_next(request)
    response.headers["x-request-id"] = number
    return response


def scanner() -> WineScanner:
    obj = state.get("scanner")
    if obj is None:  # pragma: no cover - возможно только при обращении до старта
        raise HTTPException(status_code=503, detail="модели ещё не загружены")
    return obj


def read_image(payload: bytes, filename: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        # Телефон почти всегда пишет ориентацию отдельным тегом EXIF. Без разворота половина
        # кадров придёт в модель лежащими на боку — см. wine_scanner.embed.preprocess.load_image.
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400, detail=f"{filename}: не удалось прочитать изображение"
        ) from error


async def collect(files: list[UploadFile]) -> list[Image.Image]:
    """Прочитать кадры запроса, следя за размером.

    Лимиты проверяются по ходу чтения, а не после: смысл предела в том, чтобы не дать
    сложить в память больше, чем разрешено, а проверка постфактум это уже сделала.
    """
    images, total = [], 0
    for item in files:
        name = item.filename or "кадр"
        # Starlette знает размер части заранее — на слишком большом файле это позволяет
        # отказать, не читая его целиком.
        if item.size is not None and item.size > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"{name}: файл больше {MAX_BYTES} байт")

        payload = await item.read()
        if len(payload) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"{name}: файл больше {MAX_BYTES} байт")

        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413, detail=f"запрос больше {MAX_TOTAL_BYTES} байт"
            )
        images.append(read_image(payload, name))
    return images


@app.get("/health")
def health() -> dict:
    """Живость сервиса и то, на каких артефактах он поднят."""
    engine = scanner()
    return {
        "status": "ok",
        "version": engine.version,
        "catalog_size": len(engine.index.item_ids),
        "threshold": engine.threshold,
        "index_config": engine.config,
        "devices": engine.devices(),
        "decider": engine.decider.meta,
        "load_seconds": round(float(state.get("load_seconds", 0.0)), 1),
    }


@app.post("/scan")
async def scan(files: Annotated[list[UploadFile], File()], request: Request) -> dict:
    """Опознать вино по снимку или по серии снимков.

    Несколько файлов трактуются как серия кадров одного вина: из неё берётся самый резкий
    (Э12). Это не то же самое, что несколько независимых запросов, и именно так снимает
    камера в приложении — держать пользователя, пока он выбирает лучший кадр, незачем.
    """
    number = request.state.request_id

    if not files:
        raise HTTPException(status_code=400, detail="не передано ни одного файла")
    if len(files) > MAX_FRAMES:
        raise HTTPException(status_code=413, detail=f"кадров больше {MAX_FRAMES}")

    images = await collect(files)
    engine = scanner()
    result = engine.identify(images[0]) if len(images) == 1 else engine.identify_burst(images)

    log.info(
        "%s кадров=%d ответ=%s вино=%s p=%.3f %.0f мс",
        number,
        result.frames,
        "да" if result.answered else "отказ",
        result.best.item_id if result.best else "-",
        result.best.probability if result.best else 0.0,
        result.timings["total"] * 1000,
    )
    return {"request_id": number, "version": engine.version, **result.to_dict()}
