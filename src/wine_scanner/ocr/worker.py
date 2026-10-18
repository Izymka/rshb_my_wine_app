"""Процесс-воркер PaddleOCR — самостоятельный скрипт без импорта пакета wine_scanner.

    <python окружения OCR> src/wine_scanner/ocr/worker.py --device gpu:0

Зачем отдельный интерпретатор. PaddleOCR на видеокарте — это `paddlepaddle-gpu`, а он жёстко
фиксирует свои CUDA-библиотеки (cuDNN 9.5, cuBLAS 12.6 для сборки cu126). Torch в проекте собран
под CUDA 12.8 и тянет другие версии тех же пакетов: в одном окружении они не разрешаются, а в
одном процессе две версии cuDNN — это падения. Поэтому OCR живёт в своём окружении (в Docker —
`/opt/ocr`), и этот файл обязан импортировать только numpy и paddleocr.

Протокол по stdin/stdout: каждое сообщение — 8 байт длины (big-endian) и pickle. Запрос —
массив HxWx3 uint8, ответ — {"texts", "scores", "polys"} или {"error": "..."}. Всё, что
библиотеки печатают сами, уходит в stderr: stdout занят протоколом.

Модуль же импортируется из paddle.py ради `new_engine` и `payload`: верхний уровень файла
намеренно не тянет ничего тяжелее numpy.
"""

import os
import pickle
import struct
import sys

import numpy as np

# Детектор строк меняется переменной WINE_OCR_DET (PP-OCRv5_server_det, PP-OCRv6_small_det и
# т. п.; PP-OCRv6 требует paddleocr 3.7). Распознаватель — переменной WINE_OCR_REC: имя модели
# PaddleOCR или папка с моделью. В официальном PP-OCRv6 кириллицы нет, поэтому по умолчанию
# остаётся восточнославянский eslav_PP-OCRv5_mobile_rec. Папка с `inference.onnx` (например,
# models/ocr_rec/PP-OCRv6_medium_rec_ru — PP-OCRv6 medium, дообученный на русском) читается через
# onnxruntime, детектор при этом остаётся на Paddle. Детектор и распознаватель входят в ключ кэша.
DEFAULT_DET = "PP-OCRv5_mobile_det"
DEFAULT_REC = "eslav_PP-OCRv5_mobile_rec"
ENGINE_KWARGS = dict(
    # oneDNN на Windows-сборке Paddle 3.x падает в детекторе (ConvertPirAttribute2RuntimeAttribute).
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def detector_name() -> str:
    return os.environ.get("WINE_OCR_DET") or DEFAULT_DET


def recognizer_name() -> str:
    """Распознаватель строк: имя модели PaddleOCR или папка с моделью (WINE_OCR_REC)."""
    return os.environ.get("WINE_OCR_REC") or DEFAULT_REC


def recognizer_key() -> str:
    """Как распознаватель входит в ключ кэша: у папки — её имя, а не путь на этой машине."""
    name = recognizer_name()
    return os.path.basename(os.path.normpath(name)) if os.path.isdir(name) else name


def recognizer_kwargs() -> dict:
    """Аргументы PaddleOCR для распознавателя из WINE_OCR_REC.

    Папка модели несёт своё имя в `inference.yml` (Global.model_name): по нему PaddleX выбирает
    предобработку и декодер. Если в папке только `inference.onnx`, подмодуль распознавания
    переключается на движок onnxruntime: строки готовятся и декодируются как у PaddleX, меняется
    лишь исполнитель сети. Устройство onnxruntime берёт у конвейера (gpu:0 → CUDA).
    """
    name = recognizer_name()
    if not os.path.isdir(name):
        return {"text_recognition_model_name": name}
    import yaml

    with open(os.path.join(name, "inference.yml"), encoding="utf-8") as stream:
        model_name = yaml.safe_load(stream)["Global"]["model_name"]
    kwargs = {"text_recognition_model_name": model_name, "text_recognition_model_dir": name}
    has_paddle = os.path.exists(os.path.join(name, "inference.json"))
    if os.path.exists(os.path.join(name, "inference.onnx")) and not has_paddle:
        from paddleocr._pipelines.base import _merge_dicts
        from paddlex.inference.pipelines import load_pipeline_config

        # Словарь paddlex_config заменяет конфигурацию конвейера целиком, поэтому переопределение
        # подмешивается к стандартной конфигурации OCR.
        kwargs["paddlex_config"] = _merge_dicts(
            load_pipeline_config("OCR"),
            {"SubModules": {"TextRecognition": {"engine": "onnxruntime"}}},
        )
    return kwargs


def detector_limits() -> dict:
    """До какого размера детектор строк приводит кадр: WINE_OCR_DET_SIDE и WINE_OCR_DET_LIMIT.

    По умолчанию у PaddleOCR 3.x `min` 64, то есть кадр идёт в детектор как есть. `max` N
    ограничивает детектор N пикселями по длинной стороне, а строки для распознавателя всё равно
    вырезаются из присланного кадра: на кадре в полном разрешении детектор работает на малом
    размере, а распознаватель читает чёткие строки.
    """
    limits = {}
    if os.environ.get("WINE_OCR_DET_SIDE"):
        limits["text_det_limit_side_len"] = int(os.environ["WINE_OCR_DET_SIDE"])
    if os.environ.get("WINE_OCR_DET_LIMIT"):
        limits["text_det_limit_type"] = os.environ["WINE_OCR_DET_LIMIT"]
    return limits


def new_engine(device: str = "cpu"):
    """Движок PaddleOCR: детектор строк из WINE_OCR_DET, распознаватель из WINE_OCR_REC."""
    # Avoid a slow source reachability probe on every service startup.  PaddleX still
    # downloads a missing official model normally; installed models are used from cache.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR as Engine

    return Engine(
        device=device,
        text_detection_model_name=detector_name(),
        **recognizer_kwargs(),
        **detector_limits(),
        **ENGINE_KWARGS,
    )


def payload(result) -> dict:
    """Тексты, уверенности и полигоны из ответа PaddleOCR 3.x — то, что переживает pickle."""
    if not result:
        return {}
    data = getattr(result[0], "json", result[0])
    data = data.get("res", data)
    polys = data.get("rec_polys") or data.get("dt_polys") or []
    return {
        "texts": [str(t) for t in data.get("rec_texts", [])],
        "scores": [float(s) for s in data.get("rec_scores", [])],
        "polys": [np.asarray(p, dtype=float).tolist() for p in polys],
    }


def send(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


def receive(stream):
    header = stream.read(8)
    if len(header) < 8:
        return None
    (size,) = struct.unpack(">Q", header)
    return pickle.loads(stream.read(size))


def main() -> None:
    # Папка скрипта стоит в sys.path первой, а рядом лежит наш paddle.py: без этого
    # `import paddle` внутри PaddleX нашёл бы его вместо библиотеки.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    # Протокол — на копии настоящего stdout; дескриптор 1 перенаправляется в stderr, чтобы
    # печать Paddle и PaddleX не портила поток сообщений.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    protocol_in = sys.stdin.buffer
    engine = new_engine(device)
    engine.predict(np.zeros((64, 64, 3), dtype=np.uint8))  # прогрев: модели и CUDA-контекст
    send(protocol_out, {"ready": True, "device": device})
    while True:
        array = receive(protocol_in)
        if array is None:
            break
        try:
            send(protocol_out, payload(engine.predict(array)))
        except Exception as error:  # noqa: BLE001 — ошибка уходит вызывающему, воркер живёт
            send(protocol_out, {"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
