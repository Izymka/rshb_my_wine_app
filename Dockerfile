# Сервис сканера вина под NVIDIA GPU.
#
#   docker build -t wine-scanner .
#   docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models wine-scanner
#
# База — обычная Ubuntu, а не образ nvidia/cuda, и это осознанно. Колёса torch с PyPI под Linux
# уже везут с собой весь нужный рантайм CUDA (cuBLAS, cuDNN, NCCL приезжают пакетами
# nvidia-*-cu13), а драйвер в контейнер прокидывает nvidia-container-toolkit с хоста. Образ
# nvidia/cuda в такой схеме добавляет полтора гигабайта и, что хуже, ещё одну версию CUDA,
# которая может не совпасть с той, под которую собран torch.
#
# Требование к хосту: драйвер NVIDIA под CUDA 13 (ветка r580 и новее) и установленный
# nvidia-container-toolkit. Карта — Turing или новее. Pascal (GTX 10xx) не подойдёт:
# CUDA 13 убрала поддержку этой архитектуры, а torch выкинул её ещё из сборок под 12.8.
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
    EASYOCR_MODULE_PATH=/opt/easyocr \
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

# Зависимости отдельным слоем: пересобирается только при правке pyproject.toml или uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
        --extra api --extra rerank --extra ocr --extra decide

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
RUN uv sync --frozen --extra api --extra rerank --extra ocr --extra decide

# Чужие веса выкачиваются при первом обращении: XFeat через torch.hub, EasyOCR — свои модели
# распознавания. Если этого не сделать на сборке, первый запуск контейнера пойдёт в интернет,
# займёт минуты и упадёт там, где сети нет.
RUN uv run python -c "\
import torch, easyocr; \
torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, trust_repo=True); \
easyocr.Reader(['ru','en'], gpu=False, verbose=False); \
easyocr.Reader(['fr','en'], gpu=False, verbose=False)"

EXPOSE 8000

# Один воркер намеренно. Модели занимают около 2 ГБ видеопамяти и поднимаются десятки секунд;
# второй воркер удвоит и то и другое, а очередь запросов всё равно упрётся в одну карту.
# Масштабировать надо репликами контейнера на разных картах, а не воркерами внутри одной.
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
