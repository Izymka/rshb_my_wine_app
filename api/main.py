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

import asyncio
import io
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field

from api.eval_log import EvalLog
from wine_scanner.embed import load_image
from wine_scanner.pipeline import REPORTED_CANDIDATES, ScanResult, WineScanner
from wine_scanner.sommelier import Sommelier

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
eval_log = (
    EvalLog(Path(os.environ.get("WINE_EVAL_LOG_DIR", "data/eval_requests")))
    if os.environ.get("WINE_EVAL_LOG", "0") == "1"
    else None
)

# Политика eval-ручки на незнакомом вине. С 24.09.2026 по умолчанию отвечаем всегда: заказчик
# проверяет ~100 фотографий вин каталога, и `slug: null` засчитывается ошибкой. WINE_EVAL_REFUSE=1
# вернёт честный отказ (ТЗ п. 5). Продуктовый /scan это не трогает: там отказ остаётся отказом.
EVAL_REFUSE = os.environ.get("WINE_EVAL_REFUSE", "0") == "1"

# VLM-судья (WINE_VLM=1) с 25.09.2026 работает только в eval-ручке. Там он исправляет
# близнецов одной винодельни (тест: 85 → 91 из 93), а в /scan почти не говорит «нет» и
# поднимает ложные приёмы незнакомых вин с 0.07 до 0.34. WINE_VLM_SCAN=1 вернёт его в /scan.
SCAN_JUDGE = os.environ.get("WINE_VLM_SCAN", "0") == "1"

# Порог отказа /scan поверх порога решающего слоя (meta.json, 0.533 — «максимум пользы» на train).
# 29.09.2026 в сервисе 0.40: на ужатых кадрах OCR читает хуже, и верный ответ оставался ниже
# порога. Цена по OOF train: верно 84.7 → 85.8 %, неверно 4.2 → 6.3 %, ложные приёмы 5.8 → 7.1 %;
# тест (справка): верно 85 → 87 из 100, ложные приёмы 9.8 → 12.7 %. Пусто — порог решающего слоя.
SCAN_THRESHOLD = float(os.environ["WINE_THRESHOLD"]) if os.environ.get("WINE_THRESHOLD") else None

# Пайплайн не потокобезопасен по замыслу (одна видеокарта, одни модели) и занимает секунды.
# Считаем его в рабочем потоке под замком: цикл событий остаётся отзывчивым — /health отвечает
# во время скана, — а запросы всё равно идут по одному, как и раньше.
scan_lock = asyncio.Lock()


async def run_identify(
    engine: WineScanner, images: list[Image.Image], use_judge: bool, trace: bool = False
) -> ScanResult:
    async with scan_lock:
        if len(images) == 1:
            options = {"trace": True} if trace else {}
            return await asyncio.to_thread(
                engine.identify, images[0], use_judge=use_judge, **options
            )
        return await asyncio.to_thread(engine.identify_burst, images, use_judge)


def warm_up(engine: WineScanner) -> float:
    """Прогнать один каталожный кадр через весь пайплайн до приёма запросов.

    Первый запрос после старта в полтора-два раза медленнее остальных: ленивые
    инициализации в OCR и матчере, компиляция ядер на видеокарте, холодные кэши. Скрипту
    оценки организаторов это стоит ответа — у него таймаут 10 с и ни одного повтора, и
    16.09.2026 два первых кадра из трёх ушли в null именно так. Прогрев переносит эту цену
    на старт, где её никто не считает.
    """
    started = time.perf_counter()
    path = next(iter(engine.path_by_id.values()), None)
    if path and Path(path).exists():
        try:
            engine.identify(load_image(path))
        except Exception:  # noqa: BLE001 — прогрев не должен ронять сервис
            log.exception("прогрев не удался, сервис поднимается холодным")
    return time.perf_counter() - started


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    engine = WineScanner(
        index_dir=Path(os.environ.get("WINE_INDEX", "models/index")),
        decider_dir=Path(os.environ.get("WINE_DECIDER", "models/decider")),
        threshold=SCAN_THRESHOLD,
    )
    state["warmup_seconds"] = warm_up(engine)
    state["scanner"] = engine
    state["sommelier"] = Sommelier.from_env()
    state["load_seconds"] = time.perf_counter() - started
    log.info(
        "модели загружены за %.1f с, из них прогрев %.1f с",
        state["load_seconds"],
        state["warmup_seconds"],
    )
    yield
    if eval_log is not None:
        await asyncio.to_thread(eval_log.close)
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
    is_eval = request.url.path == "/v1/eval/predict" and request.method == "POST"
    started = time.perf_counter() if is_eval and eval_log is not None else None
    requested_at = datetime.now(UTC).isoformat() if started is not None else None
    try:
        response = await call_next(request)
    except Exception:
        if started is not None:
            submit_eval_log(request, requested_at, started, 500)
        raise
    if started is not None:
        submit_eval_log(request, requested_at, started, response.status_code)
    response.headers["x-request-id"] = number
    return response


def submit_eval_log(request: Request, requested_at: str, started: float, status: int) -> None:
    """Queue disk I/O after inference; never wait for a lock in the request path."""
    if eval_log is None:
        return
    result = getattr(request.state, "eval_result", None)
    report = result.judge if result else None
    called = bool(report and report.get("applied") != "skipped")
    row = {
        "id": uuid.uuid4().hex,
        "request_id": request.state.request_id,
        "requested_at": requested_at,
        "status_code": status,
        "response_ms": round((time.perf_counter() - started) * 1000, 1),
        "filename": getattr(request.state, "eval_filename", None),
        "image_content_type": getattr(request.state, "eval_content_type", None),
        "model": getattr(request.state, "eval_model", None),
        "response": getattr(request.state, "eval_response", None),
        # `slug` может быть непустым и при отказе: такова политика eval-ручки.
        "model_answered": result.answered if result else None,
        "model_confidence_pct": round(result.best.probability * 100, 1)
        if result and result.best
        else None,
        "eval_found": request.state.eval_response["found"]
        if getattr(request.state, "eval_response", None)
        else None,
        "judge_called": called,
        "judge_result": report,
        "judge_ms": round(result.timings.get("judge", 0) * 1000, 1) if called else None,
        "judge_error_fallback": bool(called and report.get("error")),
    }
    eval_log.submit(row, getattr(request.state, "eval_image", None))


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
        # Кто отвечает за судью и сомелье, и сколько раз откатывались: без этого на демо не
        # понять, работает ли основной провайдер или всё держится на запасном.
        "llm": {
            "judge": getattr(engine.judge, "provider", None),
            "judge_calls": getattr(engine.judge, "calls", 0),
            "judge_errors": getattr(engine.judge, "errors", 0),
            # Сколько раз ответил решающий слой без судьи: облако на паузе или не хватило времени.
            "judge_skipped": getattr(engine.judge, "skipped", None),
            "sommelier": getattr(state.get("sommelier"), "provider", None),
            "sommelier_fallbacks": getattr(
                getattr(state.get("sommelier"), "llm", None), "failures", None
            ),
        },
        "load_seconds": round(float(state.get("load_seconds", 0.0)), 1),
        "warmup_seconds": round(float(state.get("warmup_seconds", 0.0)), 1),
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
    result = await run_identify(engine, images, use_judge=SCAN_JUDGE)

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


def eval_answer(result: ScanResult) -> dict:
    """Ответ в формате скрипта оценки кейсодержателя.

    Скрипт читает из объекта одно поле — `slug` — и ждёт строку. Ниже порога отдаём `null`
    и рядом похожие вина: ТЗ (п. 5) требует на незнакомое вино честно сказать «не найдено»
    и предложить максимально похожие, а не назвать чужое. Ключ каталога платформы — slug,
    поэтому item_id здесь и есть slug.
    """
    extra = {"confidence": result.confidence, "guard": result.guard, "judge": result.judge}
    if result.answered and result.best:
        return {
            "slug": result.best.item_id,
            "found": True,
            "probability": result.best.probability,
            **extra,
        }
    return {
        # Ниже порога slug остаётся null, если только политика не велит отвечать всегда.
        "slug": result.best.item_id if (result.best and not EVAL_REFUSE) else None,
        "found": False,
        "similar": [
            {
                "slug": c.item_id,
                "name": c.payload.get("name"),
                "winery": c.payload.get("winery"),
                "probability": c.probability,
            }
            for c in result.candidates[:REPORTED_CANDIDATES]
        ],
        **extra,
    }


@app.post("/v1/eval/predict")
async def predict(image: Annotated[UploadFile, File()], request: Request) -> dict:
    """Один кадр — один slug. Endpoint для participant_test.sh организаторов.

    Тот же пайплайн, что и в /scan, отличается форма ответа и судья: скрипт шлёт одно
    multipart-поле `image` и разбирает плоский объект, а VLM-судья (если WINE_VLM=1) здесь
    зовётся всегда — каталог закрыт, и выбрать верного близнеца важнее, чем отказать.
    Скрипт ждёт ответа 10 с; судья с таймаутом 4 с укладывается в p95 ~3 с на запрос.
    """
    number = request.state.request_id
    if eval_log is not None:
        request.state.eval_filename = image.filename
        request.state.eval_content_type = image.content_type
        payload = await image.read()
        request.state.eval_image = payload
        if len(payload) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"файл больше {MAX_BYTES} байт")
        images = [read_image(payload, image.filename or "кадр")]
    else:
        images = await collect([image])
    engine = scanner()
    result = await run_identify(engine, images, use_judge=True, trace=eval_log is not None)
    if eval_log is not None:
        request.state.eval_result = result
        request.state.eval_model = {
            **result.to_dict(),
            "candidates_full": [
                {
                    "item_id": candidate.item_id,
                    "probability": candidate.probability,
                    "payload": candidate.payload,
                    "features": candidate.features,
                }
                for candidate in result.candidates
            ],
            "version": engine.version,
            "trace": result.trace,
        }
    log.info(
        "%s eval ответ=%s slug=%s p=%.3f %.0f мс",
        number,
        "да" if result.answered else "отказ",
        result.best.item_id if result.best else "-",
        result.best.probability if result.best else 0.0,
        result.timings["total"] * 1000,
    )
    response = eval_answer(result)
    if eval_log is not None:
        request.state.eval_response = response
    return response


@app.get("/v1/eval/logs", response_class=HTMLResponse)
async def eval_logs_page() -> HTMLResponse:
    if eval_log is None:
        raise HTTPException(status_code=404)
    return HTMLResponse((Path(__file__).parent / "eval_logs.html").read_text(encoding="utf-8"))


@app.get("/v1/eval/logs/data")
async def eval_logs_data(offset: int = Query(default=0, ge=0)) -> list[dict]:
    if eval_log is None:
        raise HTTPException(status_code=404)
    return await asyncio.to_thread(eval_log.recent, 200, offset)


@app.get("/v1/eval/logs/images/{name}")
async def eval_log_image(name: str) -> FileResponse:
    valid_name = re.fullmatch(r"[0-9a-f]{32}\.(jpg|jpeg|png|webp|heic|heif|image)", name)
    if eval_log is None or not valid_name:
        raise HTTPException(status_code=404)
    image_path = eval_log.root / "images" / name
    if not image_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(image_path)


@app.get("/catalog/image/{slug}")
def catalog_image(slug: str) -> FileResponse:
    """Эталон карточки — для интерфейса: показать, с чем сравнили, и похожие вина картинками."""
    engine = scanner()
    payload = engine.payload_by_id.get(slug, {})
    path = payload.get("source_image_path") or engine.path_by_id.get(slug)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="нет такой карточки или её картинки")
    return FileResponse(path)


class SommelierQuestion(BaseModel):
    item_id: str
    question: str = Field(min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list, max_length=12)


@app.get("/sommelier/status")
def sommelier_status() -> dict:
    """Настроен ли сомелье — интерфейс по этому решает, показывать ли блок."""
    return {"available": state.get("sommelier") is not None}


@app.post("/sommelier")
async def sommelier(body: SommelierQuestion) -> dict:
    """Цифровой сомелье: вопрос о найденном вине, ответ заземлён в его карточке.

    404, если сомелье не настроен (нет WINE_SOMMELIER=1) — интерфейс тогда блок не показывает.
    503, если провайдер не ответил: карточка уже показана, подсказка — приятное дополнение.
    """
    helper = state.get("sommelier")
    if helper is None:
        raise HTTPException(status_code=404, detail="сомелье не настроен")
    engine = scanner()
    payload = engine.payload_by_id.get(body.item_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="нет такой карточки")
    card = {k: v for k, v in payload.items() if k not in {"image_path", "vintage_box"}}
    try:
        answer = await asyncio.to_thread(helper.ask, card, body.question, body.history)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"item_id": body.item_id, "answer": answer}
