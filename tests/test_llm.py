"""Клиенты LLM: GigaChat (OAuth, загрузка картинки), цепочка с откатом."""

import json

import httpx
import pytest

from wine_scanner.llm import ChatError, FallbackChat, GigaChatChat, OpenAICompatibleChat


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def gigachat_transport(log: list, fail_chat: bool = False):
    def handler(request: httpx.Request):
        log.append((request.method, request.url.host, request.url.path))
        if request.url.path == "/api/v2/oauth":
            assert request.headers["Authorization"] == "Basic creds"
            assert request.headers["RqUID"]
            assert b"scope=GIGACHAT_API_PERS" in request.content
            return httpx.Response(200, json={"access_token": "tok", "expires_at": 4102444800000})
        assert request.headers["Authorization"] == "Bearer tok"
        if request.url.path == "/api/v1/files":
            assert b"image/jpeg" in request.content
            return httpx.Response(200, json={"id": "file-1"})
        if request.url.path == "/api/v1/chat/completions":
            if fail_chat:
                return httpx.Response(500, text="boom")
            body = json.loads(request.content)
            if "attachments" in body["messages"][0]:
                assert body["messages"][0]["attachments"] == ["file-1"]
                return httpx.Response(200, json=completion('{"choice": 1, "confidence": 0.9}'))
            return httpx.Response(200, json=completion("К сыру."))
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gigachat_text_and_image_flow():
    log = []
    chat = GigaChatChat(credentials="creds", client=gigachat_transport(log))
    assert chat.chat([{"role": "user", "content": "к чему?"}]) == "К сыру."
    assert chat.chat_with_image("что это?", b"\xff\xd8jpeg") == '{"choice": 1, "confidence": 0.9}'
    # Токен получен один раз, картинка загружена перед вопросом.
    assert [p for _, _, p in log].count("/api/v2/oauth") == 1
    assert [p for _, _, p in log].index("/api/v1/files") < len(log) - 1


def test_gigachat_errors_are_chat_errors():
    chat = GigaChatChat(credentials="creds", client=gigachat_transport([], fail_chat=True))
    with pytest.raises(ChatError):
        chat.chat([{"role": "user", "content": "?"}])


def test_fallback_uses_second_provider_and_counts():
    def dead(request):
        raise httpx.ConnectError("no vpn")

    primary = OpenAICompatibleChat(
        "https://api.example/v1", "m", client=httpx.Client(transport=httpx.MockTransport(dead))
    )
    backup = GigaChatChat(credentials="creds", client=gigachat_transport([]))
    chain = FallbackChat([primary, backup])
    assert chain.chat([{"role": "user", "content": "?"}]) == "К сыру."
    assert chain.failures == {"openai": 1}
    assert chain.used == {"gigachat": 1}


def test_fallback_raises_when_everyone_is_down():
    def dead(request):
        raise httpx.ConnectError("no")

    client = httpx.Client(transport=httpx.MockTransport(dead))
    chain = FallbackChat([OpenAICompatibleChat("https://a/v1", "m", client=client)])
    with pytest.raises(ChatError):
        chain.chat([{"role": "user", "content": "?"}])
