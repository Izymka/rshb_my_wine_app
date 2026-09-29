"""Проверка HTTP-слоя без единой модели.

Пайплайн подменён заглушкой: здесь проверяется ровно то, за что отвечает сервис, — контракт
ответа и лимиты запроса. Всё остальное он делать не должен, и если когда-нибудь начнёт,
метрики бенчмарка перестанут описывать то, что видит пользователь.
"""

import io
import json
import os
from base64 import b64encode

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
        confidence={"top1": best.probability, "top5": 1.0, "margin": best.probability - 0.3},
        guard=None if answered else "twin",
    )


class FakeScanner:
    version = VERSION
    path_by_id = {"wine_a": "/нет/такого/a.png"}

    def __init__(self):
        self.payload_by_id: dict[str, dict] = {"wine_a": {"name": "Chateau Alpha"}}
        self.calls: list[str] = []
        self.judged: list[bool] = []
        self.answered = True

    def identify(self, image, image_key=None, use_judge=True, trace=False) -> ScanResult:
        self.calls.append("single")
        self.judged.append(use_judge)
        return answer(answered=self.answered)

    def identify_burst(self, images, use_judge=True) -> ScanResult:
        self.calls.append("burst")
        self.judged.append(use_judge)
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


def test_eval_endpoint_refuses_honestly_with_similar_wines(client, monkeypatch):
    """WINE_EVAL_REFUSE=1: незнакомое вино — slug строго null, рядом похожие, по ТЗ п. 5."""
    client.engine.answered = False
    monkeypatch.setattr(main, "EVAL_REFUSE", True)
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


def test_scan_carries_confidence_and_guard(client):
    """Уверенность для топ-1 и топ-5 — метрика из ТЗ п. 3; сработавшая защита — видна."""
    body = client.post("/scan", files={"files": ("кадр.png", frame(), "image/png")}).json()
    assert body["confidence"]["top1"] == pytest.approx(0.9)
    assert body["confidence"]["top5"] == 1.0
    assert body["guard"] is None


def test_eval_endpoint_carries_confidence_beside_slug(client):
    body = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")}).json()
    assert body["slug"] == "wine_a"
    assert body["confidence"]["top1"] == pytest.approx(0.9)


def test_eval_endpoint_answers_by_default():
    """Проверка заказчика — вина каталога, null засчитывается ошибкой: по умолчанию отвечаем."""
    if "WINE_EVAL_REFUSE" not in os.environ:
        assert main.EVAL_REFUSE is False


def test_eval_endpoint_can_be_told_to_always_answer(client, monkeypatch):
    """WINE_EVAL_REFUSE=0: ниже порога всё равно отдаём лучшего. found при этом честно false."""
    client.engine.answered = False
    monkeypatch.setattr(main, "EVAL_REFUSE", False)
    body = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")}).json()
    assert body["slug"] == "wine_a"
    assert body["found"] is False
    assert body["guard"] == "twin"


def test_catalog_image_404_for_unknown_slug(client):
    assert client.get("/catalog/image/nope").status_code == 404
    # Есть slug, но файла на диске нет — тоже 404, а не 500.
    assert client.get("/catalog/image/wine_a").status_code == 404


def test_catalog_image_prefers_source_image_from_database(client, tmp_path):
    """Карточка из PostgreSQL указывает исходный файл — он главнее вырезки из индекса."""
    source = tmp_path / "source.png"
    source.write_bytes(frame())
    client.engine.payload_by_id["wine_a"]["source_image_path"] = str(source)

    response = client.get("/catalog/image/wine_a")

    assert response.status_code == 200
    assert response.content == source.read_bytes()


def test_sommelier_is_404_when_not_configured(client, monkeypatch):
    monkeypatch.setitem(main.state, "sommelier", None)
    response = client.post("/sommelier", json={"item_id": "wine_a", "question": "к чему подать?"})
    assert response.status_code == 404


def test_sommelier_answers_from_card(client, monkeypatch):
    class FakeSommelier:
        def ask(self, card, question, history=None):
            return f"К {card['name']} — сыр. Вопрос был: {question}"

    client.engine.payload_by_id = {"wine_a": {"name": "Chateau Alpha", "image_path": "/x"}}
    monkeypatch.setitem(main.state, "sommelier", FakeSommelier())
    response = client.post("/sommelier", json={"item_id": "wine_a", "question": "к чему подать?"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("К Chateau Alpha")
    assert client.post("/sommelier", json={"item_id": "nope", "question": "?"}).status_code == 404


def test_judge_only_on_the_eval_endpoint(monkeypatch, client):
    """Судья исправляет близнецов для проверки организатора, но в /scan поднимает ложные
    приёмы незнакомых вин — поэтому там он выключен, пока не попросят WINE_VLM_SCAN=1."""
    client.post("/scan", files={"files": ("a.png", frame(), "image/png")})
    client.post("/v1/eval/predict", files={"image": ("a.png", frame(), "image/png")})
    monkeypatch.setattr(main, "SCAN_JUDGE", True)
    client.post("/scan", files={"files": ("a.png", frame(), "image/png")})

    assert client.engine.judged == [False, True, True]


def test_eval_log_records_model_photo_and_errors(monkeypatch, client, tmp_path):
    from api.eval_log import EvalLog

    audit = EvalLog(tmp_path)
    monkeypatch.setattr(main, "eval_log", audit)
    try:
        good = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})
        bad = client.post(
            "/v1/eval/predict", files={"image": ("bad.png", b"broken", "image/png")}
        )
        audit.close()
        rows = audit.recent()
        assert [row["status_code"] for row in rows] == [400, 200]
        assert rows[1]["response"]["slug"] == "wine_a"
        assert rows[1]["model"]["candidates_full"][0]["item_id"] == "wine_a"
        assert rows[1]["model_answered"] is True
        assert rows[1]["model_confidence_pct"] == 90.0
        assert rows[1]["eval_found"] is True
        assert rows[1]["judge_called"] is False
        assert rows[1]["response_ms"] >= 0
        assert (tmp_path / "images" / rows[1]["image_url"].split("/")[-1]).read_bytes() == frame()
        assert good.status_code == 200 and bad.status_code == 400
        assert client.get("/v1/eval/logs/data").json()[0]["status_code"] == 400
        assert client.get(rows[1]["image_url"]).content == frame()
        assert "Журнал запросов /v1/eval/predict" in client.get("/v1/eval/logs").text
        assert len((tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        lines = (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        assert all(json.loads(line) for line in lines)
    finally:
        audit.close()


def test_eval_log_marks_judge_error_fallback(monkeypatch, client, tmp_path):
    from api.eval_log import EvalLog

    audit = EvalLog(tmp_path)
    monkeypatch.setattr(main, "eval_log", audit)

    def with_judge(image, image_key=None, use_judge=True, trace=False):
        result = answer()
        result.timings["judge"] = 0.125
        result.judge = {"error": "timeout", "applied": "fallback", "before_id": "wine_a"}
        return result

    monkeypatch.setattr(client.engine, "identify", with_judge)
    try:
        client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})
        audit.close()
        row = audit.recent()[0]
        assert row["judge_called"] is True
        assert row["judge_error_fallback"] is True
        assert row["judge_ms"] == 125.0
        assert row["judge_result"]["before_id"] == "wine_a"
    finally:
        audit.close()


def test_eval_log_distinguishes_internal_refusal_from_returned_slug(monkeypatch, client, tmp_path):
    from api.eval_log import EvalLog

    audit = EvalLog(tmp_path)
    monkeypatch.setattr(main, "eval_log", audit)
    client.engine.answered = False
    monkeypatch.setattr(main, "EVAL_REFUSE", False)
    try:
        response = client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})
        audit.close()
        row = audit.recent()[0]
        assert response.json()["slug"] == "wine_a"
        assert row["model_answered"] is False
        assert row["eval_found"] is False
        assert client.get("/v1/eval/logs/data?answered=false").json()[0]["id"] == row["id"]
        assert client.get("/v1/eval/logs/data?answered=true").json() == []
        assert client.get("/v1/eval/logs/revision").json()["revision"] != "0"
    finally:
        audit.close()


def test_eval_log_optional_basic_auth(monkeypatch, client, tmp_path):
    from api.eval_log import EvalLog

    audit = EvalLog(tmp_path)
    monkeypatch.setattr(main, "eval_log", audit)
    monkeypatch.setattr(main, "EVAL_LOG_USER", "reader")
    monkeypatch.setattr(main, "EVAL_LOG_PASSWORD", "secret")
    try:
        client.post("/v1/eval/predict", files={"image": ("q.png", frame(), "image/png")})
        audit.close()
        image_url = audit.recent()[0]["image_url"]
        for path in ("/v1/eval/logs", "/v1/eval/logs/data", "/v1/eval/logs/revision", image_url):
            denied = client.get(path)
            assert denied.status_code == 401
            assert denied.headers["www-authenticate"] == 'Basic realm="Eval logs"'
            token = b64encode(b"reader:secret").decode()
            assert client.get(path, headers={"Authorization": f"Basic {token}"}).status_code == 200
    finally:
        audit.close()
