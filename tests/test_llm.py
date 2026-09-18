"""Клиенты LLM: Yandex AI Studio (заголовки, URI модели, картинка), цепочка с откатом."""

import json

import httpx
import pytest

from wine_scanner.llm import ChatError, FallbackChat, OpenAICompatibleChat, YandexChat, chat_from_env


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def yandex_transport(log: list, fail: bool = False):
    def handler(request: httpx.Request):
        log.append(json.loads(request.content))
        assert request.url.host == "ai.api.cloud.yandex.net"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Api-Key key"
        assert request.headers["OpenAI-Project"] == "b1gfolder"
        assert request.headers["x-data-logging-enabled"] == "false"
        if fail:
            return httpx.Response(401, json={"error": {"message": "Unknown api key"}})
        body = log[-1]
        content = body["messages"][0]["content"]
        if isinstance(content, list):
            return httpx.Response(200, json=completion('{"choice": 1, "confidence": 0.9}'))
        return httpx.Response(200, json=completion("К сыру."))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_yandex_text_and_image_flow():
    log = []
    chat = YandexChat(api_key="key", folder_id="b1gfolder", client=yandex_transport(log))
    assert chat.model == "gpt://b1gfolder/qwen3.6-35b-a3b"
    assert chat.chat([{"role": "user", "content": "к чему?"}]) == "К сыру."
    verdict = chat.chat_with_image("что это?", b"\xff\xd8jpeg", json_mode=True)
    assert verdict == '{"choice": 1, "confidence": 0.9}'
    image_body = log[-1]
    # Картинка ушла data-URI внутри сообщения; json_object Yandex не знает — не шлём.
    assert image_body["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert "response_format" not in image_body
    assert image_body["model"] == "gpt://b1gfolder/qwen3.6-35b-a3b"


def test_yandex_keeps_full_model_uri():
    chat = YandexChat(
        model="gpt://other/yandexgpt-5-lite/latest", api_key="k", folder_id="f",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    assert chat.model == "gpt://other/yandexgpt-5-lite/latest"


def test_yandex_errors_are_chat_errors():
    chat = YandexChat(api_key="key", folder_id="b1gfolder", client=yandex_transport([], fail=True))
    with pytest.raises(ChatError, match="401"):
        chat.chat([{"role": "user", "content": "?"}])


def test_openai_sends_json_mode_and_bearer():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion("{}"))

    chat = OpenAICompatibleChat(
        "https://api.example/v1", "m", api_key="sk", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    chat.chat([{"role": "user", "content": "?"}], json_mode=True)
    assert seen["auth"] == "Bearer sk"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_fallback_uses_second_provider_and_counts():
    def dead(request):
        raise httpx.ConnectError("no vpn")

    primary = OpenAICompatibleChat(
        "https://api.example/v1", "m", client=httpx.Client(transport=httpx.MockTransport(dead))
    )
    backup = YandexChat(api_key="key", folder_id="b1gfolder", client=yandex_transport([]))
    chain = FallbackChat([primary, backup])
    assert chain.chat([{"role": "user", "content": "?"}]) == "К сыру."
    assert chain.failures == {"openai": 1}
    assert chain.used == {"yandex": 1}


def test_fallback_raises_when_everyone_is_down():
    def dead(request):
        raise httpx.ConnectError("no")

    client = httpx.Client(transport=httpx.MockTransport(dead))
    chain = FallbackChat([OpenAICompatibleChat("https://a/v1", "m", client=client)])
    with pytest.raises(ChatError):
        chain.chat([{"role": "user", "content": "?"}])


def test_chat_from_env_builds_yandex_per_prefix(monkeypatch):
    for name in ("YANDEX_LLM_API_KEY", "WINE_VLM_PROVIDER", "WINE_LLM_PROVIDER", "WINE_LLM_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YANDEX_OCR_API_KEY", "ocr-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "f")
    # Без YANDEX_LLM_API_KEY берётся ключ OCR; судья и сомелье получают разные модели.
    judge = chat_from_env("WINE_VLM")
    helper = chat_from_env("WINE_LLM")
    assert isinstance(judge, YandexChat) and judge.model == "gpt://f/qwen3.6-35b-a3b"
    assert helper.model == "gpt://f/yandexgpt-5-lite"
    assert judge.api_key == "ocr-key"
    monkeypatch.setenv("YANDEX_LLM_API_KEY", "llm-key")
    monkeypatch.setenv("YANDEX_VLM_MODEL", "qwen-custom")
    assert chat_from_env("WINE_VLM").api_key == "llm-key"
    assert chat_from_env("WINE_VLM").model == "gpt://f/qwen-custom"


def test_chat_from_env_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("WINE_LLM_PROVIDER", "gigachat")
    with pytest.raises(ValueError, match="неизвестный провайдер"):
        chat_from_env("WINE_LLM")
