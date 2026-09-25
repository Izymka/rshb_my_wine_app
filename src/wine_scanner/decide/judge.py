"""VLM-судья: второе мнение на спорных случаях.

Что это. Когда решающий слой колеблется между близнецами одной винодельни или отказывает при
сильной геометрии, кроп этикетки вместе с коротким списком кандидатов уходит мультимодальной
модели: «прочитай этикетку, скажи, какая из карточек это она, или что ни одна». Модель читает
кириллицу лучше OCR и сравнивает не строки, а смысл: «Розовый Алушта» против карточки
«красный Алушта» для неё очевидное «не то».

Чего это не заменяет. Судья не ищет по каталогу — кандидатов ему даёт пайплайн, и если верной
карточки в списке нет, судья её не придумает. Поэтому он стоит в конце, после отбора и
решающего слоя, и зовётся только когда есть что рассудить (`judge_reason`).

Как подключается. `WINE_VLM=1` плюс провайдер (llm.py): Yandex AI Studio по умолчанию
(`WINE_VLM_PROVIDER=yandex`, `YANDEX_LLM_API_KEY`, `YANDEX_FOLDER_ID`; работает без VPN) или
любой OpenAI-совместимый чат (`WINE_VLM_PROVIDER=openai`, `WINE_VLM_BASE_URL`, `WINE_VLM_MODEL`,
`WINE_VLM_API_KEY`); `WINE_VLM_FALLBACK=<провайдер>` — откат, если основной не ответил.
Без `WINE_VLM=1` судья не создаётся вовсе. Любая
ошибка — сеть, таймаут, мусор в ответе — оставляет результат пайплайна как есть и оставляет
след в `ScanResult.judge`, чтобы бенчмарк видел, как часто судья молчал.

Что судья делает с ответом. Выбор карточки поднимает её наверх с вероятностью не ниже
уверенности судьи; «ни одна» с уверенностью от 0.7 переводит ответ в отказ. Судья идёт после
защиты от близнеца и может её перекрыть: он видел картинку, а правило — только слова.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as WaitTimeout
from dataclasses import dataclass, replace

import httpx
from PIL import Image

from ..llm import ChatError, OpenAICompatibleChat, chat_from_env, jpeg_bytes
from ..ocr.normalize import fold_tokens

DEFAULT_TIMEOUT = 4.0
# Защита предела организатора (participant_test.sh: --max-time 10, без повторов). Судья
# отвечает только до DEFAULT_DEADLINE с от начала запроса, иначе ответ за решающим слоем:
# 25.09 с молчащим облаком запросы шли 7.4–9.5 с — пайплайн на тяжёлом кадре сам идёт ~5 с,
# и полный таймаут судьи сверху оставлял до края полсекунды.
DEFAULT_DEADLINE = 7.0
# Меньше этого остатка звать судью бессмысленно: ответ не успеет прийти.
MIN_WAIT = 1.0
# Облако недоступно (ошибка сети или HTTP, таймаут) MAX_FAILURES раз подряд — судья молчит
# COOLDOWN с и не тратит время запросов, потом одна пробная попытка.
DEFAULT_COOLDOWN = 60.0
DEFAULT_MAX_FAILURES = 2
DEFAULT_MAX_SIDE = 1024
DEFAULT_MAX_OPTIONS = 8
DEFAULT_MAX_FAMILY = 4
# Какие карточки показывать: `all` — первые max_options; `family` — лидер и соседи по винодельне.
OPTION_MODES = ("all", "family")
# Когда звать: `reasons` — judge_reason (отказ при сильной геометрии, сосед вторым, малый
# разрыв); `family` — всегда, когда у лидера есть сосед по винодельне среди кандидатов.
TRIGGERS = ("reasons", "family")
# Строки локального OCR (PaddleOCR) для судьи: `off` — судья читает этикетку только сам;
# `hint` — строки идут в промпт подсказкой; `hint-verify` — вдобавок смену лидера может
# подтвердить не только текст, прочитанный судьёй, но и строки OCR — независимый читатель.
OCR_MODES = ("off", "hint", "hint-verify")
# Сколько текста OCR отдавать: у пёстрого кадра бывает сотня строк, промпт от них раздувается.
OCR_MAX_LINES = 40
OCR_MAX_CHARS = 1500
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
OCR_HINT = (
    "Ниже строки, которые с этой этикетки прочитала программа OCR. Это подсказка: в них бывают "
    "ошибки в буквах, пропуски и чужие слова, картинка важнее. Используй их, чтобы разобрать "
    "мелкий или плохо видимый текст; в read_text пиши то, что видишь сам."
)


def ocr_block(lines: list[str] | None) -> str:
    """Строки OCR для промпта: без пустых, не больше OCR_MAX_LINES и OCR_MAX_CHARS."""
    kept, size = [], 0
    for line in lines or []:
        line = " ".join(str(line).split())
        if not line:
            continue
        if len(kept) >= OCR_MAX_LINES or size + len(line) > OCR_MAX_CHARS:
            break
        kept.append(line)
        size += len(line)
    return "\n".join(kept)


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
    """Строка карточки для судьи. Сорта нужны: у «Азюр» Рислинг и «Азюр» Вионье одно название."""
    payload = candidate.payload
    parts = [payload.get("winery"), payload.get("name"), payload.get("category")]
    grapes = payload.get("grapes")
    if grapes:
        parts.append(f"сорта: {grapes}")
    vintage = payload.get("vintage")
    if vintage:
        parts.append(str(vintage))
    return " — ".join(" ".join(str(p).split()) for p in parts if p)


def family_options(candidates: list, family_of: dict, max_options: int, max_family: int) -> list:
    """Лидер и его соседи по винодельне из первых `max_options` кандидатов.

    Далёкие карточки судье только мешают: без рассуждений он охотно выбирает шестую-седьмую
    карточку чужой линейки. Пусто, если соседей нет — рассуживать нечего.
    """
    leader = candidates[0]
    family = family_of.get(leader.item_id)
    if not family:
        return []
    siblings = [
        c for c in candidates[1:max_options] if family_of.get(c.item_id) == family
    ][: max_family - 1]
    return [leader, *siblings] if siblings else []


def _stems(text: str) -> set[str]:
    # Пять букв свёрнутой формы: «красный» и «красное», «полусладкое» и «полусладкий» — одно.
    return {token[:5] for token in fold_tokens(text) if len(token) > 2}


def supports_switch(read_text: str, chosen, leader) -> bool:
    """Подтверждает ли прочитанный судьёй текст выбор `chosen` вместо `leader`.

    Смотрим только на слова, которыми карточки различаются: хотя бы одно слово выбранной
    карточки, которого нет у лидера, должно быть в тексте этикетки. Если различий в словах нет
    (дубль каталога), текст ничего не подтверждает.
    """
    def words(candidate) -> str:
        payload = candidate.payload
        return " ".join(
            str(payload.get(key) or "") for key in ("name", "category", "grapes", "vintage")
        )

    distinct = _stems(words(chosen)) - _stems(words(leader))
    return bool(distinct & _stems(read_text))


def consistent_pick(read_text: str, shown: list, chosen, text_index):
    """Соседка, которую текст судьи подтверждает вместо его же выбора, или None.

    Судья часто читает этикетку верно, а карточку выбирает не ту: прочитал «ZINFANDEL
    SEMI-DRY ROSE» — выбрал «Semi-Dry Rose» с сортом Пино Нуар. Здесь прочитанное судьёй
    сравнивается с карточками теми же сигналами, что и текст OCR в пайплайне (TextIndex.signals).
    Выбор меняется, только когда текст *против* него — цвет, сорт или сладость противоречат, или
    у карточки есть различающие слова, но ни одного нет в тексте, — и *за* другую показанную
    карточку: она ничему не противоречит и подтверждена сортом или своим словом.
    """
    if not read_text:
        return None
    tokens = text_index.query_tokens(read_text)
    attrs = text_index.query_attributes(tokens, read_text)
    signals = {c.item_id: text_index.signals(tokens, attrs, c.item_id) for c in shown}

    def contradicts(s: dict) -> bool:
        return s["color_match"] == -1 or s["style_match"] == -1 or s.get("grape_match", 0) == -1

    def supported(s: dict) -> bool:
        return s.get("grape_match", 0) == 1 or s["disc_hit"] > 0

    own = signals[chosen.item_id]
    unsupported = own["disc_n"] > 0 and own["disc_hit"] == 0 and own.get("grape_match", 0) != 1
    if not (contradicts(own) or unsupported):
        return None
    alternatives = [
        c
        for c in shown
        if c is not chosen and not contradicts(signals[c.item_id]) and supported(signals[c.item_id])
    ]
    if not alternatives:
        return None

    def evidence(c) -> tuple:
        s = signals[c.item_id]
        return (
            s["color_match"],
            s.get("grape_match", 0),
            s["style_match"],
            round(s["disc_hit"], 2),
            round(s["name_cover"], 2),
        )

    return max(alternatives, key=evidence)


def apply_verdict(
    candidates: list, verdict: Verdict, threshold: float, options: list | None = None
) -> tuple[list, bool, str]:
    """Применить вердикт к списку кандидатов. Возвращает (кандидаты, answered, что сделано).

    `options` — карточки в том порядке, в каком их видел судья (по умолчанию первые кандидаты).
    """
    if options is None:
        options = candidates[:DEFAULT_MAX_OPTIONS]
    if verdict.error:
        best = candidates[0]
        return candidates, best.probability >= threshold, "none"
    if verdict.choice is not None and 1 <= verdict.choice <= len(options):
        chosen = options[verdict.choice - 1]
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
        options: str = "all",
        trigger: str = "reasons",
        verify: bool = False,
        max_family: int = DEFAULT_MAX_FAMILY,
        ocr: str = "off",
        check: bool = False,
        deadline: float | None = DEFAULT_DEADLINE,
        cooldown: float = DEFAULT_COOLDOWN,
        max_failures: int = DEFAULT_MAX_FAILURES,
        clock=time.monotonic,
    ):
        if options not in OPTION_MODES or trigger not in TRIGGERS or ocr not in OCR_MODES:
            raise ValueError(
                f"VlmJudge: options из {OPTION_MODES}, trigger из {TRIGGERS}, ocr из {OCR_MODES}"
            )
        # Либо готовый чат-клиент (llm.py: OpenAI-совместимый, Yandex, цепочка с откатом),
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
        self.options = options
        self.trigger = trigger
        # Смену лидера принимать, только если текст этикетки её подтверждает (supports_switch).
        self.verify = verify
        self.max_family = max_family
        self.ocr = ocr
        # Сверять выбор судьи с его же прочитанным текстом (consistent_pick).
        self.check = check
        # Предел на весь запрос (с его начала); None или 0 — ждать, сколько даст таймаут.
        self.deadline = deadline or None
        self.cooldown = cooldown
        self.max_failures = max_failures
        self.clock = clock
        self.calls = 0
        self.errors = 0
        # Сколько раз судью не позвали: облако на паузе (cooldown) или не хватило времени.
        self.skipped = {"cooldown": 0, "deadline": 0}
        self.failures_in_row = 0
        self.paused_until = 0.0
        # Вызов в своём потоке, чтобы бросить его по дедлайну: таймаут httpx считается от
        # начала вызова, а не от начала запроса. Брошенный поток доживёт до своего таймаута.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="judge")

    @classmethod
    def from_env(cls) -> "VlmJudge | None":
        if os.environ.get("WINE_VLM", "0") != "1":
            return None
        timeout = float(os.environ.get("WINE_VLM_TIMEOUT", DEFAULT_TIMEOUT))
        return cls(
            llm=chat_from_env("WINE_VLM", timeout=timeout),
            timeout=timeout,
            json_mode=os.environ.get("WINE_VLM_JSON_MODE", "1") == "1",
            # По умолчанию — вариант family-verify: выбран на живых кадрах train без вин теста
            # (eval/judge_validation.py), на тесте 85 → 91 из 93 без единой поломки top-1.
            options=os.environ.get("WINE_VLM_OPTIONS", "family"),
            trigger=os.environ.get("WINE_VLM_TRIGGER", "family"),
            verify=os.environ.get("WINE_VLM_VERIFY", "1") == "1",
            ocr=os.environ.get("WINE_VLM_OCR", "off"),
            # Сверка выбора с текстом судьи: на стенде 87 → 89 из 91, стресс 68 → 70 из 85,
            # 7 замен из 7 верные. Строки OCR в промпт (WINE_VLM_OCR) не помогли: 87 → 84–85.
            check=os.environ.get("WINE_VLM_CHECK", "1") == "1",
            deadline=float(os.environ.get("WINE_VLM_DEADLINE", DEFAULT_DEADLINE)),
            cooldown=float(os.environ.get("WINE_VLM_COOLDOWN", DEFAULT_COOLDOWN)),
            max_failures=int(os.environ.get("WINE_VLM_MAX_FAILURES", DEFAULT_MAX_FAILURES)),
        )

    def select(self, candidates: list, answered: bool, threshold: float, family_of: dict):
        """Звать ли судью и какие карточки показать: (причина, карточки) или (None, [])."""
        if not candidates:
            return None, []
        family = family_options(candidates, family_of, self.max_options, self.max_family)
        if self.trigger == "family":
            reason = "family" if family else None
        else:
            reason = judge_reason(candidates, answered, threshold, family_of)
        if reason is None:
            return None, []
        if self.options == "family":
            return (reason, family) if family else (None, [])
        return reason, candidates[: self.max_options]

    @property
    def provider(self) -> str:
        return getattr(self.llm, "name", "?")

    def judge(
        self,
        crop: Image.Image,
        options: list[str],
        ocr_lines: list[str] | None = None,
        wait: float | None = None,
    ) -> Verdict:
        """Спросить судью. `wait` — сколько секунд ждать ответа (None — сколько даст таймаут)."""
        self.calls += 1
        listing = "\n".join(f"{i}. {text}" for i, text in enumerate(options, start=1))
        prompt = f"{PROMPT}\n\nКарточки:\n{listing}"
        hint = ocr_block(ocr_lines) if self.ocr != "off" else ""
        if hint:
            prompt = f"{prompt}\n\n{OCR_HINT}\nСтроки OCR:\n{hint}"
        image = jpeg_bytes(crop, self.max_side)
        try:
            if wait is None:
                text = self._ask(prompt, image)
            else:
                text = self._pool.submit(self._ask, prompt, image).result(timeout=wait)
        except (ChatError, WaitTimeout) as error:
            self.errors += 1
            self._failed()
            message = str(error) if isinstance(error, ChatError) else f"дедлайн {wait:.1f} с"
            return Verdict(None, 0.0, "", error=message)
        # Облако ответило — даже непонятный ответ значит, что оно живо.
        self.failures_in_row = 0
        verdict = parse_verdict(text)
        if verdict.error:
            self.errors += 1
        return verdict

    def _ask(self, prompt: str, image: bytes) -> str:
        return self.llm.chat_with_image(prompt, image, temperature=0.0, json_mode=self.json_mode)

    def _failed(self) -> None:
        self.failures_in_row += 1
        if self.failures_in_row >= self.max_failures:
            self.paused_until = self.clock() + self.cooldown
            self.failures_in_row = 0

    def availability(self, elapsed: float | None) -> tuple[str | None, float | None]:
        """Можно ли звать судью сейчас: (почему нельзя или None, сколько ждать ответа)."""
        if self.clock() < self.paused_until:
            return "cooldown", None
        if self.deadline is None or elapsed is None:
            return None, None
        wait = self.deadline - elapsed
        if wait < MIN_WAIT:
            return "deadline", None
        return None, wait

    def verification_text(self, read_text: str, ocr_lines: list[str] | None) -> str:
        """Текст, которым подтверждается смена лидера: прочитанное судьёй, а в `hint-verify`
        ещё и строки OCR."""
        if self.ocr == "hint-verify" and ocr_lines:
            return " ".join([read_text, *map(str, ocr_lines)])
        return read_text

    def settle(self, verdict: Verdict, shown: list, leader, ocr_lines=None, text_index=None):
        """Какую карточку выбрал судья после проверок: (карточка или None, исход).

        Исход: `judge` — как сказал судья; `consistent` — выбор заменён соседкой, которая
        сходится с текстом, прочитанным самим судьёй (`check`); `unverified` — смену лидера
        текст не подтвердил (`verify`), ответ пайплайна остаётся.
        """
        chosen = (
            shown[verdict.choice - 1]
            if not verdict.error and verdict.choice and 1 <= verdict.choice <= len(shown)
            else None
        )
        outcome = "judge"
        if self.check and chosen is not None and text_index is not None:
            better = consistent_pick(verdict.read_text, shown, chosen, text_index)
            if better is not None:
                chosen, outcome = better, "consistent"
        if (
            self.verify
            and outcome == "judge"
            and chosen is not None
            and chosen is not leader
            and not supports_switch(
                self.verification_text(verdict.read_text, ocr_lines), chosen, leader
            )
        ):
            outcome = "unverified"
        return chosen, outcome

    def consult(
        self,
        crop: Image.Image,
        candidates: list,
        answered: bool,
        threshold: float,
        family_of: dict,
        ocr_lines: list[str] | None = None,
        text_index=None,
        elapsed: float | None = None,
    ):
        """Полный шаг судьи для пайплайна: решить, звать ли, позвать, применить.

        `ocr_lines` — строки локального OCR; судье они уходят только при `ocr != "off"`.
        `text_index` — для сверки выбора с текстом судьи (`check`); без него сверки нет.
        `elapsed` — сколько секунд запрос уже идёт; по нему судья укладывается в `deadline`.
        Без ответа судьи (пауза, дедлайн, ошибка) остаётся ответ решающего слоя.
        """
        reason, shown = self.select(candidates, answered, threshold, family_of)
        if reason is None:
            return candidates, answered, None
        # Что было до судьи: без этого не сказать, помог он или испортил ответ.
        leader = candidates[0]
        before_id, before_probability = leader.item_id, leader.probability
        answered_before = answered
        before = {
            "before_id": before_id,
            "before_probability": before_probability,
            "before_answered": answered_before,
        }
        skip, wait = self.availability(elapsed)
        if skip:
            self.skipped[skip] += 1
            report = {
                "reason": reason,
                "options": len(shown),
                "ocr": self.ocr,
                "choice": None,
                "confidence": 0.0,
                "read_text": "",
                "error": skip,
                "applied": "skipped",
                **before,
            }
            return candidates, answered, report
        verdict = self.judge(crop, [describe(c) for c in shown], ocr_lines, wait=wait)
        chosen, outcome = self.settle(verdict, shown, leader, ocr_lines, text_index)
        if outcome == "unverified":
            applied = "unverified"
        else:
            if outcome == "consistent":
                verdict = replace(verdict, choice=shown.index(chosen) + 1)
            candidates, answered, applied = apply_verdict(candidates, verdict, threshold, shown)
            if outcome == "consistent" and applied == "choose":
                applied = "consistent"
        report = {
            "reason": reason,
            "options": len(shown),
            "ocr": self.ocr,
            "choice": verdict.choice,
            "confidence": verdict.confidence,
            "read_text": verdict.read_text,
            "error": verdict.error or None,
            "applied": applied,
            **before,
        }
        return candidates, answered, report
