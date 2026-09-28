# syntax=docker/dockerfile:1
# Сервис сканера вина под NVIDIA GPU.
#
#   docker build -t wine-scanner .
#   docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models wine-scanner
#
# База — обычная Ubuntu, а не образ nvidia/cuda, и это осознанно. Колёса torch с PyPI под Linux
# уже везут с собой весь нужный рантайм CUDA (cuBLAS, cuDNN, NCCL приезжают пакетами
# nvidia-*-cu12), а драйвер в контейнер прокидывает nvidia-container-toolkit с хоста. Образ
# nvidia/cuda в такой схеме добавляет полтора гигабайта и, что хуже, ещё одну версию CUDA,
# которая может не совпасть с той, под которую собран torch.
#
# Требование к хосту: драйвер NVIDIA r570 и новее (torch собран под CUDA 12.8, Paddle-GPU —
# под 12.6) и установленный nvidia-container-toolkit. Карта — Turing или новее, от 6 ГБ
# видеопамяти (сервис в пике занимает ~2.2 ГБ, замер 28.09). Pascal (GTX 10xx) не подойдёт:
# torch выкинул эту архитектуру из сборок под 12.8.
#
# Кэш uv монтируется только на время сборки (--mount=type=cache) и в образ не попадает:
# раньше он оставался в слоях и весил 16 ГБ из 34.
#
# Веса не копируются в образ, а монтируются томом. Их около 700 МБ (индекс, дескрипторы
# каталога, детектор этикетки, решающий слой), и они меняются чаще кода: пересобирать образ
# ради переиндексации каталога — гарантированный способ однажды выкатить сервис со старым
# индексом и не понять, почему метрики поехали.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    TORCH_HOME=/opt/torch \
    # Без этих двух переменных nvidia-container-toolkit не прокинет драйвер в контейнер:
    # образы nvidia/cuda задают их сами, обычная Ubuntu — нет.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# libgl и libglib нужны opencv, ffmpeg — разбору серий кадров из Live Photo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv curl ca-certificates \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# PaddleOCR на видеокарте — своё окружение. paddlepaddle-gpu жёстко фиксирует CUDA-библиотеки
# под 12.6 (cuDNN 9.5, cuBLAS 12.6), torch проекта собран под 12.8: вместе они не ставятся, а в
# одном процессе две версии cuDNN — это падения. Сервис запускает src/wine_scanner/ocr/worker.py
# этим интерпретатором и общается с ним по stdin/stdout. Замер 28.09: OCR 27 мс на GPU против
# 1.4 с на CPU, /scan на живых кадрах train p95 3.3 → 1.9 с.
# Два шага: у индекса Paddle есть свой старый paddleocr, а uv берёт пакет из первого индекса,
# где его нашёл, — paddleocr ставится отдельно, только с PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/ocr --python python3.12 \
    && uv pip install --python /opt/ocr/bin/python \
        --index-url https://pypi.org/simple \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        "paddlepaddle-gpu==3.3.1" \
    && uv pip install --python /opt/ocr/bin/python "paddleocr==3.4.1"
ENV WINE_OCR_PYTHON=/opt/ocr/bin/python \
    WINE_OCR_DEVICE=gpu:0

# Зависимости отдельным слоем: пересобирается только при правке pyproject.toml или uv.lock.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra api --extra rerank --extra ocr --extra decide

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api --extra rerank --extra ocr --extra decide

# Чужие веса выкачиваются при первом обращении: XFeat через torch.hub, PaddleOCR — свои модели
# распознавания, детектор бутылки RT-DETR (COCO) — с HuggingFace. Если этого не сделать на
# сборке, первый запуск контейнера пойдёт в интернет, займёт минуты и упадёт там, где сети нет.
RUN uv run python -c "\
import torch; \
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor; \
from wine_scanner.detect.rtdetr import COCO_BOTTLE_MODEL; \
from wine_scanner.ocr.paddle import PaddleOCR; \
torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, trust_repo=True); \
RTDetrImageProcessor.from_pretrained(COCO_BOTTLE_MODEL); \
RTDetrForObjectDetection.from_pretrained(COCO_BOTTLE_MODEL); \
PaddleOCR()._get_engine()"

EXPOSE 8000

# Один воркер намеренно. Модели занимают около 2 ГБ видеопамяти и поднимаются десятки секунд;
# второй воркер удвоит и то и другое, а очередь запросов всё равно упрётся в одну карту.
# Масштабировать надо репликами контейнера на разных картах, а не воркерами внутри одной.
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
