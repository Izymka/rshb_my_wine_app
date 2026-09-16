"""Проверка HTTP-слоя без единой модели.

Пайплайн подменён заглушкой: здесь проверяется ровно то, за что отвечает сервис, — контракт
ответа и лимиты запроса. Всё остальное он делать не должен, и если когда-нибудь начнёт,
метрики бенчмарка перестанут описывать то, что видит пользователь.
"""

import io

import pytest

pytest.importorskip("fastapi", reason="слой сервиса ставится группой api")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api import main  # noqa: E402
from wine_scanner.pipeline import Candidate, ScanResult  # noqa: E402

VERSION = {"index": "1234abcd5678", "decider": "8765dcba4321"}


def frame(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def answer(frames: int = 1, answered: bool = True) -> ScanResult:
    best = Candidate(
        item_id="wine_a", probability=0.9 if answered else 0.4,
        payload={"name": "Chateau Alpha", "winery": "Alpha"},
    )
    second = Candidate(item_id="wine_b", probability=0.3, payload={"name": "Chateau Beta"})
    return ScanResult(
        answered=answered,
        best=best,
        candidates=[best, second],
        text="CHATEAU ALPHA",
        timings={"total": 0.5},
        threshold=0.73,
        frames=frames,
    )


class FakeScanner:
    version = VERSION

    def __init__(self):
        self.calls: list[str] = []
        self.answered = True

    def identify(self, image, image_key=None) -> ScanResult:
        self.calls.append("single")
        return answer(answered=self.answered)

    def identify_burst(self, images) -> ScanResult:
        self.calls.append("burst")
        return answer(frames=len(images))


@pytest.fixture
def client(monkeypatch):
    """Клиент с подменённым пайплайном.

    Без контекстного менеджера намеренно: он запустил бы lifespan, то есть поднял четыре
    настоящие модели — десятки секунд и гигабайты ради проверки разбора запроса.
    """
    engine = FakeScanner()
    monkeypatch.setitem(main.state, "scanner", engine)
    test_client = TestClient(main.app)
    test_client.engine = engine
    return test_client


def test_request_id_of_the_caller_is_kept(client):
    """Свой номер клиента важнее нашего: по нему жалоба пользователя свяжется с нашим логом."""
    response = client.post(
        "/scan", files={"files": ("кадр.png", frame(), "image/png")},
        headers={"X-Request-Id": "laravel-42"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "laravel-42"
    assert response.headers["x-request-id"] == "laravel-42"


def test_request_id_appears_even_without_one(client):
    response = client.post("/scan", files={"files": ("кадр.png", frame(), "image/png")})

    assert response.json()["request_id"]


def test_answer_carries_the_version_of_the_artifacts(client):
    """Индекс пересобирается, и без версии ответы разных недель в логах неразличимы."""
    response = client.post("/scan", files={"files": ("кадр.png", frame(), "image/png")})

    assert response.json()["version"] == VERSION


def test_series_of_frames_goes_the_burst_way(client):
    """Несколько файлов — это серия кадров одного вина, а не несколько запросов."""
    files = [("files", ("a.png", frame("white"), "image/png")),
             ("files", ("b.png", frame("black"), "image/png"))]
    response = client.post("/scan", files=files)

    assert client.engine.calls == ["burst"]
    assert response.json()["frames"] == 2


def test_too_many_frames_are_refused(client):
    files = [("files", (f"{i}.png", frame(), "image/png")) for i in range(main.MAX_FRAMES + 1)]

    assert client.post("/scan", files=files).status_code == 413
    assert client.engine.calls == []


def test_whole_request_is_limited_not_just_one_file(monkeypatch, client):
    """Двенадцать файлов по разрешённому размеру не должны складываться в память целиком."""
    monkeypatch.setattr(main, "MAX_TOTAL_BYTES", len(frame()) + 10)
    files = [("files", ("a.png", frame(), "image/png")),
             ("files", ("b.png", frame(), "image/png"))]

    assert client.post("/scan", files=files).status_code == 413
    assert client.engine.calls == []


def test_request_id_survives_an_error(client):
    """Номер нужнее всего там, где ответа не получилось: иначе отказ в логе не найти."""
    files = [("files", (f"{i}.png", frame(), "image/png")) for i in range(main.MAX_FRAMES + 1)]
    response = client.post("/scan", files=files, headers={"X-Request-Id": "laravel-77"})

    assert response.status_code == 413
    assert response.headers["x-request-id"] == "laravel-77"


def test_broken_file_is_a_client_error(client):
    response = client.post("/scan", files={"files": ("кадр.png", b"not an image", "image/png")})

    assert response.status_code == 400


def test_eval_endpoint_returns_a_flat_slug(client):
    """Скрипт организаторов читает из ответа одно поле — slug — и ждёт там строку."""
    response = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "wine_a"
    assert body["found"] is True
    assert client.engine.calls == ["single"]


def test_eval_endpoint_refuses_honestly_with_similar_wines(client):
    """Незнакомое вино: slug строго null (скрипт запишет отказ), рядом — похожие, по ТЗ п. 5."""
    client.engine.answered = False
    response = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})

    body = response.json()
    assert body["slug"] is None
    assert body["found"] is False
    assert [c["slug"] for c in body["similar"]] == ["wine_a", "wine_b"]
    assert body["similar"][0]["winery"] == "Alpha"


def test_eval_endpoint_rejects_a_broken_file(client):
    """Не-200 скрипт сам превращает в null; главное — не отвечать 200 без slug."""
    response = client.post(
        "/v1/eval/predict", files={"image": ("q.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400
