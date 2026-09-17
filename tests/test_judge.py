"""VLM-судья: разбор ответа, политика вызова, применение вердикта, поведение в пайплайне."""

import json

import httpx
import pytest
from PIL import Image

from wine_scanner.decide.judge import (
    Verdict,
    VlmJudge,
    apply_verdict,
    judge_reason,
    parse_verdict,
)
from wine_scanner.pipeline import Candidate


@pytest.mark.parametrize(
    ("text", "choice", "confidence"),
    [
        ('{"choice": 2, "confidence": 0.9, "read_text": "x"}', 2, 0.9),
        ('```json\n{"choice": null, "confidence": 0.8}\n```', None, 0.8),
        ('Вот ответ: {"choice": "3", "confidence": "0.5"} спасибо', 3, 0.5),
        ('{"choice": true, "confidence": 5}', None, 1.0),
    ],
)
def test_parse_verdict_is_lenient(text, choice, confidence):
    verdict = parse_verdict(text)
    assert verdict.error == ""
    assert verdict.choice == choice
    assert verdict.confidence == pytest.approx(confidence)


def test_parse_verdict_garbage():
    assert parse_verdict("не знаю").error == "parse"
    assert parse_verdict("{oops").error == "parse"


def cand(item_id, p, winery="W", inliers=0):
    return Candidate(item_id, p, {"winery": winery, "name": item_id}, {"inliers": inliers})


def test_judge_reason_triggers():
    fam = {"a": "Массандра", "b": "Массандра", "c": "Другая"}
    assert judge_reason([cand("a", 0.9), cand("b", 0.1)], True, 0.5, fam) == "same_family"
    assert judge_reason([cand("a", 0.55), cand("c", 0.45)], True, 0.5, fam) == "margin"
    assert judge_reason([cand("a", 0.1, inliers=50), cand("c", 0.0)], False, 0.5, fam) == "refused_strong"
    assert judge_reason([cand("a", 0.9), cand("c", 0.1)], True, 0.5, fam) is None
    assert judge_reason([], True, 0.5, fam) is None


def test_apply_verdict_choose_refuse_none():
    a, b = cand("a", 0.6), cand("b", 0.3)
    reordered, answered, applied = apply_verdict([a, b], Verdict(2, 0.95), 0.5)
    assert [c.item_id for c in reordered] == ["b", "a"]
    assert reordered[0].probability == pytest.approx(0.95)
    assert answered and applied == "choose"

    a = cand("a", 0.6)
    _, answered, applied = apply_verdict([a], Verdict(None, 0.9), 0.5)
    assert not answered and applied == "refuse" and a.probability < 0.5

    a = cand("a", 0.6)
    _, answered, applied = apply_verdict([a], Verdict(None, 0.3), 0.5)
    assert answered and applied == "none"

    a = cand("a", 0.6)
    _, answered, applied = apply_verdict([a], Verdict(None, 0.0, error="http 500"), 0.5)
    assert answered and applied == "none"


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_vlm_judge_sends_image_and_options():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=completion('{"choice": 1, "confidence": 0.8, "read_text": "МАССАНДРА"}'))

    judge = VlmJudge("https://api.example/v1", "vlm", api_key="k", client=transport(handler))
    verdict = judge.judge(Image.new("RGB", (2000, 1000)), ["Массандра — Портвейн", "Другое"])
    assert verdict.choice == 1 and verdict.read_text == "МАССАНДРА"
    assert seen["auth"] == "Bearer k"
    content = seen["body"]["messages"][0]["content"]
    assert "1. Массандра — Портвейн" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert seen["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: httpx.Response(500, text="boom"),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
        lambda r: httpx.Response(200, json={"weird": 1}),
        lambda r: httpx.Response(200, json=completion("не json")),
    ],
)
def test_vlm_judge_errors_are_verdicts(handler):
    judge = VlmJudge("https://api.example/v1", "vlm", client=transport(handler))
    verdict = judge.judge(Image.new("RGB", (10, 10)), ["a"])
    assert verdict.error
    assert judge.errors == 1


def test_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WINE_VLM", raising=False)
    assert VlmJudge.from_env() is None
    monkeypatch.setenv("WINE_VLM", "1")
    monkeypatch.delenv("WINE_VLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        VlmJudge.from_env()
