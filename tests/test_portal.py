"""Сведения с портала: разбор ответа, кэш и поведение при сбоях — без сети."""

import pytest

from api.portal import MISS_TTL, TTL, Portal, parse

ANSWER = {
    "publicRating": 4.93,
    "alcohol": 14,
    "temperature": "14-16",
    "dishes": [{"name": "Мясо и стейки", "image": {}}, {"image": {}}],
    "slug": "ruzh",
}


def test_parse_keeps_only_sane_values():
    facts = parse(ANSWER, "ruzh")
    assert facts == {
        "rating": 4.93,
        "alcohol": 14.0,
        "temperature": "14-16",
        "dishes": ["Мясо и стейки"],
        "url": "https://vino-svoe.ru/wines/ruzh",
    }
    broken = parse({"publicRating": 0, "alcohol": "крепкое", "temperature": "<b>", "dishes": None}, "x")
    assert broken["rating"] is None and broken["alcohol"] is None
    assert broken["temperature"] is None and broken["dishes"] == []


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_answer_is_cached_until_ttl():
    calls, clock = [], Clock()

    def fetch(url, timeout):
        calls.append(url)
        return ANSWER

    portal = Portal(api="https://api.test/v1", fetch=fetch, clock=clock)
    assert portal.facts("ruzh")["rating"] == 4.93
    assert portal.facts("ruzh")["rating"] == 4.93
    assert calls == ["https://api.test/v1/wines/ruzh"]
    clock.now += TTL + 1
    portal.facts("ruzh")
    assert len(calls) == 2


def test_unknown_wine_is_none_and_cached():
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return None

    portal = Portal(fetch=fetch, clock=Clock())
    assert portal.facts("nope") is None
    assert portal.facts("nope") is None
    assert len(calls) == 1


def test_failure_keeps_stale_answer_and_backs_off():
    clock, state = Clock(), {"fail": False, "calls": 0}

    def fetch(url, timeout):
        state["calls"] += 1
        if state["fail"]:
            raise TimeoutError("портал молчит")
        return ANSWER

    portal = Portal(fetch=fetch, clock=clock)
    portal.facts("ruzh")
    clock.now += TTL + 1
    state["fail"] = True
    assert portal.facts("ruzh")["rating"] == 4.93  # устаревший ответ, а не пустота
    assert portal.facts("ruzh")["rating"] == 4.93  # и без нового запроса до MISS_TTL
    assert state["calls"] == 2 and portal.errors == 1
    clock.now += MISS_TTL + 1
    portal.facts("ruzh")
    assert state["calls"] == 3


def test_slug_is_escaped_in_url():
    seen = []
    Portal(api="https://api.test/v1", fetch=lambda url, t: seen.append(url), clock=Clock()).facts("a/b?c")
    assert seen == ["https://api.test/v1/wines/a%2Fb%3Fc"]


@pytest.mark.parametrize("value", ["0", None])
def test_from_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("WINE_PORTAL", raising=False)
        assert isinstance(Portal.from_env(), Portal)
    else:
        monkeypatch.setenv("WINE_PORTAL", value)
        assert Portal.from_env() is None
