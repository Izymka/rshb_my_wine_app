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

ENGINE_KWARGS = dict(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    # oneDNN на Windows-сборке Paddle 3.x падает в детекторе (ConvertPirAttribute2RuntimeAttribute).
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def new_engine(device: str = "cpu"):
    """Движок PP-OCRv5: mobile-детектор и восточнославянский распознаватель."""
    # Avoid a slow source reachability probe on every service startup.  PaddleX still
    # downloads a missing official model normally; installed models are used from cache.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR as Engine

    return Engine(device=device, **ENGINE_KWARGS)


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
