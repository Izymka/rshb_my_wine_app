"""OCR-бэкенды: разбор ответа облака, откат на PaddleOCR, ключи кэша."""

import json

import httpx
import pytest
from PIL import Image

from wine_scanner.ocr import LabelOCR, TextLine
from wine_scanner.ocr.yandex import OCRBackendError, YandexOCR, parse_response

CANNED = {
    "result": {
        "textAnnotation": {
            "width": "200",
            "height": "100",
            "blocks": [
                {
                    "lines": [
                        {
                            "text": "МАССАНДРА",
                            "boundingBox": {
                                "vertices": [
                                    {"x": "20", "y": "10"},
                                    {"x": "180", "y": "10"},
                                    {"x": "180", "y": "30"},
                                    {"x": "20", "y": "30"},
                                ]
                            },
                            "words": [{"text": "МАССАНДРА", "confidence": 0.97}],
                        },
                        {"text": "  ", "boundingBox": {"vertices": []}},
                        {
                            "text": "ПОРТВЕЙН",
                            "boundingBox": {
                                "vertices": [{"x": "0", "y": "50"}, {"x": "100", "y": "70"}]
                            },
                        },
                    ]
                }
            ],
        }
    }
}


def test_parse_response_gives_fractional_boxes():
    lines = parse_response(CANNED, 200, 100)
    assert [line.text for line in lines] == ["МАССАНДРА", "ПОРТВЕЙН"]
    assert lines[0].source == "yandex"
    assert lines[0].confidence == pytest.approx(0.97)
    assert lines[0].box == pytest.approx((0.1, 0.1, 0.9, 0.3))
    # Без уверенности по словам — константа, а не ноль.
    assert lines[1].confidence == pytest.approx(0.9)


def test_parse_response_rejects_garbage():
    with pytest.raises(OCRBackendError):
        parse_response({"error": "quota"}, 10, 10)


class FakeCloud:
    def __init__(self, lines=None, fail=False):
        self.lines = lines or []
        self.fail = fail
        self.calls = 0

    def read(self, image):
        self.calls += 1
        if self.fail:
            raise OCRBackendError("сеть")
        return self.lines


@pytest.fixture
def image():
    return Image.new("RGB", (64, 32), "white")


def test_cloud_lines_are_returned(tmp_path, image):
    cloud = FakeCloud([TextLine("АЛУШТА", 0.9, "yandex")])
    ocr = LabelOCR(cache_dir=tmp_path, backend="yandex", cloud=cloud)
    assert [line.text for line in ocr.read(image)] == ["АЛУШТА"]
    assert cloud.calls == 1
    assert ocr.fallbacks == 0


def test_cloud_failure_falls_back_to_paddle(tmp_path, image, monkeypatch):
    ocr = LabelOCR(cache_dir=tmp_path, backend="yandex", cloud=FakeCloud(fail=True))
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: [TextLine("местный", 0.5, "paddle")])
    assert [line.text for line in ocr.read(image)] == ["местный"]
    assert ocr.fallbacks == 1


def test_cache_key_differs_between_backends(tmp_path, image):
    local = LabelOCR(cache_dir=tmp_path)
    same_local = LabelOCR(cache_dir=tmp_path, backend="paddle")
    cloud = LabelOCR(cache_dir=tmp_path, backend="yandex", cloud=FakeCloud())
    assert local._cache_path(image) == same_local._cache_path(image)
    assert local._cache_path(image) != cloud._cache_path(image)


def test_cloud_reads_larger_frames(tmp_path):
    cloud = LabelOCR(cache_dir=tmp_path, backend="yandex", cloud=FakeCloud())
    assert cloud.effective_max_side() == 1024
    assert LabelOCR(cache_dir=tmp_path).effective_max_side() == 640


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError):
        LabelOCR(cache_dir=tmp_path, backend="easyocr")


def test_hybrid_calls_cloud_only_for_weak_local_text(tmp_path, image, monkeypatch):
    cloud = FakeCloud([TextLine("облако", 0.9, "yandex")])
    ocr = LabelOCR(cache_dir=tmp_path, backend="hybrid", cloud=cloud)
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: [TextLine("коротко", 0.5, "paddle")])
    assert [line.text for line in ocr.read(image)] == ["облако"]
    assert cloud.calls == 1


STRONG = [TextLine(t, 0.9, "paddle") for t in ("МАССАНДРА", "ПОРТВЕЙН", "АЛУШТА")]


@pytest.mark.parametrize(
    ("local", "cloud_expected", "reason"),
    [
        ([TextLine("коротко", 0.9, "paddle")], True, "few_lines"),
        ([TextLine(t, 0.4, "paddle") for t in "абв"], True, "low_conf"),
        (STRONG, False, None),
    ],
)
def test_hybrid_route_says_why(tmp_path, image, monkeypatch, local, cloud_expected, reason):
    ocr = LabelOCR(cache_dir=tmp_path, backend="hybrid", cloud=FakeCloud(STRONG[:1]))
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: local)
    _, route = ocr.read_with_route(image)
    assert route["cloud"] is cloud_expected
    assert route["reason"] == reason
    assert route["local_lines"] == len(local)
    assert route["local_conf_max"] == pytest.approx(max(line.confidence for line in local))


def test_hybrid_route_marks_fallback(tmp_path, image, monkeypatch):
    ocr = LabelOCR(cache_dir=tmp_path, backend="hybrid", cloud=FakeCloud(fail=True))
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: [TextLine("x", 0.9, "paddle")])
    lines, route = ocr.read_with_route(image)
    assert [line.source for line in lines] == ["paddle"]
    assert route["cloud"] and route["fallback"]


def test_cloud_failure_is_not_cached(tmp_path, image, monkeypatch):
    cloud = FakeCloud([TextLine("облако", 0.9, "yandex")], fail=True)
    ocr = LabelOCR(cache_dir=tmp_path, backend="hybrid", cloud=cloud)
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: [TextLine("x", 0.9, "paddle")])
    assert ocr.read_with_route(image, use_cache=True)[1]["fallback"]
    cloud.fail = False
    lines, route = ocr.read_with_route(image, use_cache=True)
    assert [line.text for line in lines] == ["облако"] and not route["fallback"]
    assert cloud.calls == 2


def test_route_survives_cache_without_second_cloud_call(tmp_path, image, monkeypatch):
    cloud = FakeCloud([TextLine("облако", 0.9, "yandex")])
    ocr = LabelOCR(cache_dir=tmp_path, backend="hybrid", cloud=cloud)
    monkeypatch.setattr(ocr, "_read_paddle", lambda img: [TextLine("x", 0.9, "paddle")])
    first = ocr.read_with_route(image, use_cache=True)
    second = ocr.read_with_route(image, use_cache=True)
    assert cloud.calls == 1
    assert second[1] == first[1]
    # Кэш, записанный до маршрутов: маршрут восстанавливается локально, облако не зовётся.
    for sidecar in tmp_path.glob("*.route.json"):
        sidecar.unlink()
    lines, route = ocr.read_with_route(image, use_cache=True)
    assert cloud.calls == 1
    assert [line.text for line in lines] == ["облако"]
    assert route["cloud"] and route["reason"] == "few_lines"


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_yandex_client_parses_ok(image):
    def handler(request):
        body = json.loads(request.content)
        assert body["mimeType"] == "JPEG" and body["languageCodes"] == ["ru", "en"]
        assert request.headers["Authorization"] == "Api-Key k"
        assert request.headers["x-folder-id"] == "f"
        return httpx.Response(200, json=CANNED)

    ocr = YandexOCR(api_key="k", folder_id="f", client=transport(handler))
    assert [line.text for line in ocr.read(image)] == ["МАССАНДРА", "ПОРТВЕЙН"]


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(500, text="boom"),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
        lambda request: httpx.Response(200, text="not json"),
    ],
)
def test_yandex_client_errors_become_backend_errors(image, handler):
    ocr = YandexOCR(api_key="k", client=transport(handler))
    with pytest.raises(OCRBackendError):
        ocr.read(image)


def test_yandex_without_key_is_backend_error(image, monkeypatch):
    monkeypatch.delenv("YANDEX_OCR_API_KEY", raising=False)
    with pytest.raises(OCRBackendError):
        YandexOCR(api_key=None, client=transport(lambda r: httpx.Response(200))).read(image)
