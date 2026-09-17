"""Клиенты языковых моделей для судьи и сомелье: один интерфейс, два провайдера, откат.

Зачем слой. Судья (decide/judge.py) и сомелье (sommelier.py) хотят одного и того же: отправить
текст (и, для судьи, картинку) и получить строку. Как это делается у провайдеров — разное:
OpenAI-совместимые чаты (OpenAI, Gemini, OpenRouter) принимают картинку внутри сообщения
data-URI и Bearer-ключ; GigaChat требует обменять ключ авторизации на токен, картинку сначала
загрузить отдельным запросом и сослаться на неё по идентификатору, а сертификат у него от
российского удостоверяющего центра. Всё это спрятано здесь.

Откат. Основной провайдер может быть за VPN, а живой прогон у организаторов — без гарантии
сети. `FallbackChat` пробует клиентов по очереди и запоминает, кто ответил; сервису это
незаметно, в `/health` видно, кем отвечали.

Переменные окружения (см. .env.example):
- `WINE_VLM_PROVIDER` / `WINE_LLM_PROVIDER`: `openai` (по умолчанию) или `gigachat`;
- `WINE_VLM_FALLBACK` / `WINE_LLM_FALLBACK`: провайдер на случай отказа основного;
- OpenAI-совместимый: `*_BASE_URL`, `*_MODEL`, `*_API_KEY`;
- GigaChat: `GIGACHAT_CREDENTIALS` (ключ авторизации из личного кабинета), `GIGACHAT_SCOPE`
  (`GIGACHAT_API_PERS` по умолчанию), `GIGACHAT_MODEL` (`GigaChat-2-Pro` по умолчанию —
  младшая модель картинки не понимает), `GIGACHAT_CA_BUNDLE` (путь к сертификату Минцифры;
  без него проверка TLS выключается, о чём пишется в лог).
"""

import io
import json
import logging
import os
import time
import uuid
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

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.client is None:
            self.client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=2.0))

    def _post(self, body: dict) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
        except httpx.HTTPError as error:
            raise ChatError(f"{self.name}: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise ChatError(f"{self.name}: HTTP {response.status_code}")
        try:
            return _content(response.json())
        except ValueError as error:
            raise ChatError(f"{self.name}: ответ не JSON") from error

    def chat(self, messages: list[dict], temperature: float = 0.4, json_mode: bool = False) -> str:
        body = {"model": self.model, "temperature": temperature, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._post(body)

    def chat_with_image(
        self, prompt: str, image: bytes, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        import base64

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


GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_DEFAULT_MODEL = "GigaChat-2-Pro"
# Токен живёт 30 минут; обновляем заранее, чтобы не поймать 401 посреди запроса.
GIGACHAT_TOKEN_MARGIN = 120.0


@dataclass
class GigaChatChat:
    """GigaChat: OAuth по ключу авторизации, картинки через /files, работает без VPN."""

    credentials: str
    model: str = GIGACHAT_DEFAULT_MODEL
    scope: str = "GIGACHAT_API_PERS"
    timeout: float = DEFAULT_TIMEOUT
    verify: bool | str = False
    client: httpx.Client | None = None
    name: str = "gigachat"
    _token: str | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            if self.verify is False:
                log.warning(
                    "GigaChat: проверка TLS выключена — задайте GIGACHAT_CA_BUNDLE "
                    "(сертификат Минцифры), чтобы включить"
                )
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=3.0), verify=self.verify
            )

    def _refresh_token(self) -> None:
        headers = {
            "Authorization": f"Basic {self.credentials}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            response = self.client.post(
                GIGACHAT_OAUTH_URL, headers=headers, data={"scope": self.scope}
            )
        except httpx.HTTPError as error:
            raise ChatError(f"gigachat oauth: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise ChatError(f"gigachat oauth: HTTP {response.status_code}")
        data = response.json()
        self._token = data["access_token"]
        # expires_at приходит в миллисекундах эпохи.
        self._expires_at = float(data.get("expires_at", 0)) / 1000.0 or time.time() + 1500

    def _headers(self) -> dict:
        if not self._token or time.time() > self._expires_at - GIGACHAT_TOKEN_MARGIN:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _post_json(self, path: str, body: dict) -> dict:
        try:
            response = self.client.post(
                f"{GIGACHAT_API_URL}{path}", json=body, headers=self._headers()
            )
            if response.status_code == 401:
                self._token = None
                response = self.client.post(
                    f"{GIGACHAT_API_URL}{path}", json=body, headers=self._headers()
                )
        except httpx.HTTPError as error:
            raise ChatError(f"gigachat: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise ChatError(f"gigachat: HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def upload_image(self, image: bytes) -> str:
        """Положить картинку в хранилище GigaChat, вернуть идентификатор для attachments."""
        try:
            response = self.client.post(
                f"{GIGACHAT_API_URL}/files",
                headers=self._headers(),
                files={"file": ("label.jpg", image, "image/jpeg")},
                data={"purpose": "general"},
            )
        except httpx.HTTPError as error:
            raise ChatError(f"gigachat files: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise ChatError(f"gigachat files: HTTP {response.status_code}")
        try:
            return response.json()["id"]
        except (KeyError, ValueError) as error:
            raise ChatError("gigachat files: в ответе нет id") from error

    def chat(self, messages: list[dict], temperature: float = 0.4, json_mode: bool = False) -> str:
        # GigaChat не знает response_format: просим JSON словами — разбор у судьи терпимый.
        body = {"model": self.model, "temperature": max(temperature, 0.01), "messages": messages}
        return _content(self._post_json("/chat/completions", body))

    def chat_with_image(
        self, prompt: str, image: bytes, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        file_id = self.upload_image(image)
        messages = [{"role": "user", "content": prompt, "attachments": [file_id]}]
        return self.chat(messages, temperature=temperature, json_mode=json_mode)


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


def gigachat_from_env(timeout: float = DEFAULT_TIMEOUT) -> GigaChatChat | None:
    credentials = os.environ.get("GIGACHAT_CREDENTIALS")
    if not credentials:
        return None
    bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    return GigaChatChat(
        credentials=credentials,
        model=os.environ.get("GIGACHAT_MODEL", GIGACHAT_DEFAULT_MODEL),
        scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        timeout=timeout,
        verify=bundle if bundle else False,
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
    providers = [os.environ.get(f"{prefix}_PROVIDER", "openai")]
    fallback = os.environ.get(f"{prefix}_FALLBACK")
    if fallback and fallback not in providers:
        providers.append(fallback)

    clients = []
    for provider in providers:
        if provider == "gigachat":
            client = gigachat_from_env(timeout)
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
