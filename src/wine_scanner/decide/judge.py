"""VLM-судья: второе мнение на спорных случаях.

Что это. Когда решающий слой колеблется между близнецами одной винодельни или отказывает при
сильной геометрии, кроп этикетки вместе с коротким списком кандидатов уходит мультимодальной
модели: «прочитай этикетку, скажи, какая из карточек это она, или что ни одна». Модель читает
кириллицу лучше OCR и сравнивает не строки, а смысл: «Розовый Алушта» против карточки
«красный Алушта» для неё очевидное «не то».

Чего это не заменяет. Судья не ищет по каталогу — кандидатов ему даёт пайплайн, и если верной
карточки в списке нет, судья её не придумает. Поэтому он стоит в конце, после отбора и
решающего слоя, и зовётся только когда есть что рассудить (`judge_reason`).

Как подключается. `WINE_VLM=1` плюс провайдер (llm.py): OpenAI-совместимый чат
(`WINE_VLM_BASE_URL`, `WINE_VLM_MODEL`, `WINE_VLM_API_KEY`) или GigaChat
(`WINE_VLM_PROVIDER=gigachat`, `GIGACHAT_CREDENTIALS`); `WINE_VLM_FALLBACK=gigachat` — откат
без VPN, если основной провайдер не ответил. Без `WINE_VLM=1` судья не создаётся вовсе. Любая
ошибка — сеть, таймаут, мусор в ответе — оставляет результат пайплайна как есть и оставляет
след в `ScanResult.judge`, чтобы бенчмарк видел, как часто судья молчал.

Что судья делает с ответом. Выбор карточки поднимает её наверх с вероятностью не ниже
уверенности судьи; «ни одна» с уверенностью от 0.7 переводит ответ в отказ. Судья идёт после
защиты от близнеца и может её перекрыть: он видел картинку, а правило — только слова.
"""

import json
import os
import re
from dataclasses import dataclass

import httpx
from PIL import Image

from ..llm import ChatError, OpenAICompatibleChat, chat_from_env, jpeg_bytes

DEFAULT_TIMEOUT = 4.0
DEFAULT_MAX_SIDE = 1024
DEFAULT_MAX_OPTIONS = 8
MARGIN_TRIGGER = 0.2
STRONG_INLIERS = 30
REFUSE_CONFIDENCE = 0.7
GUARD_EPS = 1e-3

PROMPT = (
    "На фотографии — этикетка бутылки вина. Ниже пронумерованный список карточек из каталога. "
    "Прочитай текст на этикетке (название, производитель, цвет, сладость, год) и определи, какая "
    "карточка описывает именно эту бутылку. Карточки одной линейки отличаются одним словом — "
    "сравнивай слова буквально: «красный» и «розовый» — разные вина, «брют» и «полусладкое» — тоже. "
    "Если ни одна карточка не подходит, choice должен быть null. "
    'Ответь строго одним JSON-объектом вида {"choice": <номер или null>, '
    '"confidence": <число от 0 до 1>, "read_text": "<текст, который ты прочитал на этикетке>"}.'
)


@dataclass
class Verdict:
    choice: int | None  # номер варианта, начиная с 1, или None
    confidence: float
    read_text: str = ""
    error: str = ""


def parse_verdict(text: str) -> Verdict:
    """Разобрать ответ модели, терпя ограду ```json и лишний текст вокруг объекта."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return Verdict(None, 0.0, "", error="parse")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(None, 0.0, "", error="parse")
    choice = data.get("choice")
    if isinstance(choice, str) and choice.strip().isdigit():
        choice = int(choice)
    if not isinstance(choice, int) or isinstance(choice, bool):
        choice = None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return Verdict(choice, confidence, str(data.get("read_text", "")))


def judge_reason(candidates, answered: bool, threshold: float, family_of: dict) -> str | None:
    """Стоит ли звать судью. None — случай не спорный."""
    if not candidates:
        return None
    best = candidates[0]
    if not answered:
        # Отказ при сильной геометрии: похоже, вино знакомое, но модель не решилась.
        return "refused_strong" if best.features.get("inliers", 0) >= STRONG_INLIERS else None
    second = candidates[1] if len(candidates) > 1 else None
    if second is None:
        return None
    family = family_of.get(best.item_id)
    if family and family == family_of.get(second.item_id):
        return "same_family"
    if best.probability - second.probability < MARGIN_TRIGGER:
        return "margin"
    return None


def describe(candidate) -> str:
    payload = candidate.payload
    parts = [payload.get("winery"), payload.get("name"), payload.get("category")]
    vintage = payload.get("vintage")
    if vintage:
        parts.append(str(vintage))
    return " — ".join(str(p) for p in parts if p)


def apply_verdict(candidates: list, verdict: Verdict, threshold: float) -> tuple[list, bool, str]:
    """Применить вердикт к списку кандидатов. Возвращает (кандидаты, answered, что сделано)."""
    if verdict.error:
        best = candidates[0]
        return candidates, best.probability >= threshold, "none"
    if verdict.choice is not None and 1 <= verdict.choice <= min(
        len(candidates), DEFAULT_MAX_OPTIONS
    ):
        chosen = candidates[verdict.choice - 1]
        chosen.probability = max(chosen.probability, verdict.confidence)
        reordered = [chosen] + [c for c in candidates if c is not chosen]
        return reordered, chosen.probability >= threshold, "choose"
    if verdict.choice is None and verdict.confidence >= REFUSE_CONFIDENCE:
        best = candidates[0]
        best.probability = min(best.probability, threshold - GUARD_EPS)
        return candidates, False, "refuse"
    best = candidates[0]
    return candidates, best.probability >= threshold, "none"


class VlmJudge:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_side: int = DEFAULT_MAX_SIDE,
        max_options: int = DEFAULT_MAX_OPTIONS,
        json_mode: bool = True,
        client: httpx.Client | None = None,
        llm=None,
    ):
        # Либо готовый чат-клиент (llm.py: OpenAI-совместимый, GigaChat, цепочка с откатом),
        # либо параметры OpenAI-совместимого — из них клиент собирается здесь.
        if llm is None:
            if not base_url or not model:
                raise ValueError("VlmJudge: нужен llm или base_url и model")
            llm = OpenAICompatibleChat(
                base_url=base_url, model=model, api_key=api_key, timeout=timeout, client=client
            )
        self.llm = llm
        self.max_side = max_side
        self.max_options = max_options
        self.json_mode = json_mode
        self.calls = 0
        self.errors = 0

    @classmethod
    def from_env(cls) -> "VlmJudge | None":
        if os.environ.get("WINE_VLM", "0") != "1":
            return None
        timeout = float(os.environ.get("WINE_VLM_TIMEOUT", DEFAULT_TIMEOUT))
        return cls(
            llm=chat_from_env("WINE_VLM", timeout=timeout),
            timeout=timeout,
            json_mode=os.environ.get("WINE_VLM_JSON_MODE", "1") == "1",
        )

    @property
    def provider(self) -> str:
        return getattr(self.llm, "name", "?")

    def judge(self, crop: Image.Image, options: list[str]) -> Verdict:
        self.calls += 1
        listing = "\n".join(f"{i}. {text}" for i, text in enumerate(options, start=1))
        prompt = f"{PROMPT}\n\nКарточки:\n{listing}"
        try:
            text = self.llm.chat_with_image(
                prompt, jpeg_bytes(crop, self.max_side), temperature=0.0, json_mode=self.json_mode
            )
        except ChatError as error:
            self.errors += 1
            return Verdict(None, 0.0, "", error=str(error))
        verdict = parse_verdict(text)
        if verdict.error:
            self.errors += 1
        return verdict

    def consult(
        self, crop: Image.Image, candidates: list, answered: bool, threshold: float, family_of: dict
    ):
        """Полный шаг судьи для пайплайна: решить, звать ли, позвать, применить."""
        reason = judge_reason(candidates, answered, threshold, family_of)
        if reason is None:
            return candidates, answered, None
        options = [describe(c) for c in candidates[: self.max_options]]
        verdict = self.judge(crop, options)
        candidates, answered, applied = apply_verdict(candidates, verdict, threshold)
        report = {
            "reason": reason,
            "choice": verdict.choice,
            "confidence": verdict.confidence,
            "read_text": verdict.read_text,
            "error": verdict.error or None,
            "applied": applied,
        }
        return candidates, answered, report
