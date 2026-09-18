"""Клиенты языковых моделей для судьи и сомелье: один интерфейс, два провайдера, откат.

Зачем слой. Судья (decide/judge.py) и сомелье (sommelier.py) хотят одного и того же: отправить
текст (и, для судьи, картинку) и получить строку. Провайдеры при этом различаются в мелочах:
OpenAI-совместимые чаты (OpenAI, Gemini, OpenRouter) принимают картинку внутри сообщения
data-URI и Bearer-ключ; Yandex AI Studio говорит на том же диалекте, но ключ ждёт в форме
`Api-Key`, папку — отдельным заголовком, а модель — как `gpt://<папка>/<модель>`. Всё это
спрятано здесь.

Почему Yandex. Основной сценарий — живой прогон у организаторов без гарантии VPN. Yandex AI
Studio работает из России, картинки понимает (`qwen3.6-35b-a3b`), русский текст — родной.
GigaChat пробовали 17.09: на любую картинку этикетки отвечает отказом («чувствительная тема»),
для судьи не годится — убран целиком, чтобы не держать мёртвый код.

Откат. `FallbackChat` пробует клиентов по очереди и запоминает, кто ответил; сервису это
незаметно, в `/health` видно, кем отвечали.

Переменные окружения (см. .env.example):
- `WINE_VLM_PROVIDER` / `WINE_LLM_PROVIDER`: `yandex` (по умолчанию) или `openai`;
- `WINE_VLM_FALLBACK` / `WINE_LLM_FALLBACK`: провайдер на случай отказа основного;
- OpenAI-совместимый: `*_BASE_URL`, `*_MODEL`, `*_API_KEY`;
- Yandex: `YANDEX_LLM_API_KEY` (ключ сервисного аккаунта с ролью `ai.languageModels.user`
  и областью действия `yc.ai.languageModels.execute` — ключ, выпущенный только под OCR,
  сервис языковых моделей не знает; если переменной нет, берётся `YANDEX_OCR_API_KEY`),
  `YANDEX_FOLDER_ID`, `YANDEX_VLM_MODEL` (судья, `qwen3.6-35b-a3b`), `YANDEX_LLM_MODEL`
  (сомелье, `yandexgpt-5-lite`), `YANDEX_LLM_BASE_URL` (`https://ai.api.cloud.yandex.net/v1`;
  прежний адрес `https://llm.api.cloud.yandex.net/v1` тоже отвечает).
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field

import httpx
from PIL import Image

log = logging.getLogger("wine_scanner.llm")

DEFAULT_TIMEOUT = 12.0


class ChatError(RuntimeError):
    """Провайдер не ответил или ответил не тем."""


def jpeg_bytes(image: Image.Image, max_side: int = 1024, quality: int = 85) -> bytes:
    copy = image.convert("RGB").copy()
    copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _content(data: dict) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ChatError("в ответе нет choices[0].message.content") from error
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


@dataclass
class OpenAICompatibleChat:
    """Любой чат с API в стиле OpenAI: OpenAI, Gemini (endpoint /openai), OpenRouter и т. п."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    client: httpx.Client | None = None
    name: str = "openai"
    # Умеет ли провайдер response_format=json_object; иначе JSON просим словами в подсказке,
    # а разбор у судьи терпит ограду и лишний текст.
    supports_json_mode: bool = True

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.client is None:
            self.client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=2.0))

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _post(self, body: dict) -> str:
        headers = self._headers()
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
        except httpx.HTTPError as error:
            raise ChatError(f"{self.name}: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise ChatError(f"{self.name}: HTTP {response.status_code}: {response.text[:200]}")
        try:
            return _content(response.json())
        except ValueError as error:
            raise ChatError(f"{self.name}: ответ не JSON") from error

    def chat(self, messages: list[dict], temperature: float = 0.4, json_mode: bool = False) -> str:
        body = {"model": self.model, "temperature": temperature, "messages": messages}
        if json_mode and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._post(body)

    def chat_with_image(
        self, prompt: str, image: bytes, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
        return self.chat(messages, temperature=temperature, json_mode=json_mode)


YANDEX_LLM_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
YANDEX_VLM_MODEL = "qwen3.6-35b-a3b"
YANDEX_LLM_MODEL = "yandexgpt-5-lite"


@dataclass
class YandexChat(OpenAICompatibleChat):
    """Yandex AI Studio: OpenAI-совместимый чат, работает без VPN.

    Отличия от чистого OpenAI: ключ в заголовке `Api-Key`, папка — в `OpenAI-Project`,
    модель — `gpt://<папка>/<модель>` (короткое имя дописывается само). `response_format`
    у Yandex — только `json_schema`, поэтому `json_object` не шлём: судья просит JSON словами.
    """

    base_url: str = YANDEX_LLM_BASE_URL
    model: str = YANDEX_VLM_MODEL
    folder_id: str | None = None
    name: str = "yandex"
    supports_json_mode: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.folder_id and not self.model.startswith("gpt://"):
            self.model = f"gpt://{self.folder_id}/{self.model}"

    def _headers(self) -> dict:
        headers = {"x-data-logging-enabled": "false"}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        if self.folder_id:
            headers["OpenAI-Project"] = self.folder_id
        return headers


@dataclass
class FallbackChat:
    """Цепочка клиентов: первый ответивший побеждает, отказавшие считаются."""

    clients: list
    name: str = "fallback"
    used: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)

    def _run(self, method: str, *args, **kwargs) -> str:
        last: Exception | None = None
        for client in self.clients:
            try:
                result = getattr(client, method)(*args, **kwargs)
            except ChatError as error:
                self.failures[client.name] = self.failures.get(client.name, 0) + 1
                log.warning("LLM %s не ответил: %s", client.name, error)
                last = error
                continue
            self.used[client.name] = self.used.get(client.name, 0) + 1
            return result
        raise ChatError(f"ни один провайдер не ответил: {last}")

    def chat(self, messages: list[dict], temperature: float = 0.4, json_mode: bool = False) -> str:
        return self._run("chat", messages, temperature=temperature, json_mode=json_mode)

    def chat_with_image(
        self, prompt: str, image: bytes, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        return self._run(
            "chat_with_image", prompt, image, temperature=temperature, json_mode=json_mode
        )


def yandex_from_env(prefix: str, timeout: float = DEFAULT_TIMEOUT) -> YandexChat | None:
    """Клиент Yandex AI Studio; модель — `YANDEX_VLM_MODEL` для судьи, `YANDEX_LLM_MODEL` для сомелье."""
    api_key = os.environ.get("YANDEX_LLM_API_KEY") or os.environ.get("YANDEX_OCR_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        return None
    kind = prefix.removeprefix("WINE_")  # VLM | LLM
    default_model = YANDEX_VLM_MODEL if kind == "VLM" else YANDEX_LLM_MODEL
    return YandexChat(
        base_url=os.environ.get("YANDEX_LLM_BASE_URL", YANDEX_LLM_BASE_URL),
        model=os.environ.get(f"YANDEX_{kind}_MODEL", default_model),
        api_key=api_key,
        folder_id=folder_id,
        timeout=timeout,
    )


def openai_from_env(prefix: str, timeout: float = DEFAULT_TIMEOUT) -> OpenAICompatibleChat | None:
    """Клиент по переменным `{prefix}_BASE_URL`, `{prefix}_MODEL`, `{prefix}_API_KEY`."""
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    model = os.environ.get(f"{prefix}_MODEL")
    if not base_url or not model:
        return None
    return OpenAICompatibleChat(
        base_url=base_url,
        model=model,
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        timeout=timeout,
    )


def chat_from_env(prefix: str, timeout: float = DEFAULT_TIMEOUT):
    """Собрать клиента (с откатом) по `{prefix}_PROVIDER` и `{prefix}_FALLBACK`.

    `prefix` — `WINE_VLM` для судьи или `WINE_LLM` для сомелье; у сомелье переменные
    `WINE_LLM_*` могут отсутствовать — тогда берутся судейские.
    """
    providers = [os.environ.get(f"{prefix}_PROVIDER", "yandex")]
    fallback = os.environ.get(f"{prefix}_FALLBACK")
    if fallback and fallback not in providers:
        providers.append(fallback)

    clients = []
    for provider in providers:
        if provider == "yandex":
            client = yandex_from_env(prefix, timeout)
        elif provider == "openai":
            client = openai_from_env(prefix, timeout) or (
                openai_from_env("WINE_VLM", timeout) if prefix != "WINE_VLM" else None
            )
        else:
            raise ValueError(f"неизвестный провайдер LLM: {provider}")
        if client is None:
            raise ValueError(
                f"провайдер {provider} для {prefix} не настроен: нет ключей в окружении"
            )
        clients.append(client)
    return clients[0] if len(clients) == 1 else FallbackChat(clients)
