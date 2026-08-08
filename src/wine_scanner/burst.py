"""Серия кадров вместо одного снимка.

Одиночное фото — это ставка на удачу: смазался момент нажатия, попал блик, дрогнула рука.
Телефон при съёмке Live Photo и так пишет короткое видео вокруг кадра, а камера в приложении
даёт непрерывный поток. Значит вместо одного кадра почти бесплатно доступен десяток, и среди
них почти всегда есть удачный.

Здесь: извлечение кадров из видео и способы свести несколько результатов поиска в один.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .catalog import NON_WINE_DIRS, OWN_ROOT

FRAME_CACHE = Path("models/burst_frames")
VIDEO_SUFFIXES = {".mp4", ".mov"}


@dataclass
class Burst:
    """Серия кадров одного вина: то, что реально приходит с камеры."""

    frames: list[Path]
    true_id: str
    source: str = ""
    meta: dict = field(default_factory=dict)


def extract_frames(video: Path, cache_dir: Path = FRAME_CACHE, fps: float = 3.0) -> list[Path]:
    """Вытащить кадры из видео с заданной частотой, с кэшем на диске.

    3 кадра в секунду на видео Live Photo длиной около трёх секунд дают примерно девять
    кадров. Брать все 83 незачем: соседние кадры почти одинаковы, а времени на них уходит
    в девять раз больше.
    """
    out_dir = cache_dir / video.parent.name / video.stem
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
         str(out_dir / "frame_%03d.jpg")],
        check=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def load_bursts(root: Path = OWN_ROOT, fps: float = 3.0) -> list[Burst]:
    """Собрать серии из Live Photo, лежащих рядом со снимками.

    Правильный ответ — имя папки, ровно как для одиночных запросов. Имена видео с именами
    снимков не связаны (телефон называет их по-своему), но это и не нужно: серия относится
    к вину, а не к конкретному кадру.
    """
    bursts = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name in NON_WINE_DIRS:
            continue
        for video in sorted(folder.iterdir()):
            if video.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            frames = extract_frames(video, fps=fps)
            if frames:
                bursts.append(Burst(frames=frames, true_id=folder.name, source=video.name))
    return bursts


def sharpness(image: Image.Image) -> float:
    """Оценка резкости через дисперсию лапласиана.

    Лапласиан подчёркивает перепады яркости. У резкого кадра их много и они сильные, у
    смазанного края размыты и дисперсия мала. Метрика грубая, зато считается за миллисекунды
    и не требует никакой модели — то есть выбрать лучший кадр серии можно до того, как
    потрачены деньги на эмбеддинг.
    """
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fuse_max(per_frame: list[dict[str, float]]) -> dict[str, float]:
    """Кандидату — лучшая из его оценок по кадрам.

    Логика: достаточно одного удачного кадра, чтобы узнать вино. Плохие кадры не должны
    тянуть вниз ответ, подтверждённый хорошим.
    """
    fused: dict[str, float] = {}
    for scores in per_frame:
        for item_id, score in scores.items():
            fused[item_id] = max(fused.get(item_id, -1.0), score)
    return fused


def fuse_mean(per_frame: list[dict[str, float]]) -> dict[str, float]:
    """Средняя оценка по кадрам, где кандидат не найден — считаем нулём.

    Строже max: требует, чтобы кандидат подтверждался несколькими кадрами, а не одним.
    """
    totals: dict[str, float] = {}
    for scores in per_frame:
        for item_id, score in scores.items():
            totals[item_id] = totals.get(item_id, 0.0) + score
    return {k: v / len(per_frame) for k, v in totals.items()}


def fuse_rrf(per_frame: list[list[str]], k: int = 60) -> dict[str, float]:
    """Слияние рангов (Reciprocal Rank Fusion).

    Складывает не оценки, а позиции: кандидат получает сумму 1/(k + ранг) по всем кадрам.
    Тем самым не нужно приводить шкалы к общему виду — они могут быть какими угодно.
    Тот же приём понадобится на Э7 для слияния визуальной и текстовой веток.
    """
    fused: dict[str, float] = {}
    for ranking in per_frame:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def ranked(scores: dict[str, float]) -> list[str]:
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
