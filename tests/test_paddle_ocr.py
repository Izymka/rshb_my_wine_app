"""PaddleOCR adapter: parsing stays independent from model downloads."""

import pytest
from PIL import Image

from wine_scanner.ocr.paddle import PaddleOCR


class Result:
    json = {
        "res": {
            "rec_texts": ["МАССАНДРА", "шум"],
            "rec_scores": [0.91, 0.1],
            "rec_polys": [
                [[10, 20], [190, 20], [190, 40], [10, 40]],
                [[0, 0], [2, 0], [2, 2], [0, 2]],
            ],
        }
    }


class Engine:
    def predict(self, image):
        return [Result()]


def test_paddle_adapter_filters_confidence_and_normalizes_box():
    ocr = PaddleOCR(min_confidence=0.3)
    ocr._engine = Engine()
    lines = ocr.read(Image.new("RGB", (200, 100), "white"))
    assert [(line.text, line.source) for line in lines] == [("МАССАНДРА", "paddle")]
    assert lines[0].box == (0.05, 0.2, 0.95, 0.4)


def test_subprocess_worker_gets_absolute_interpreter_path(tmp_path, monkeypatch):
    """`.venv-ocr/Scripts/python.exe` из .env: CreateProcess на Windows не находит путь с «/»."""
    import subprocess

    from wine_scanner.ocr.paddle import SubprocessPaddleOCR

    python = tmp_path / "venv" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    seen = {}

    class Stop(Exception):
        pass

    def fake_popen(args, **kwargs):
        seen["args"] = args
        raise Stop

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ocr = SubprocessPaddleOCR(python="venv/python.exe", device="gpu:0")
    try:
        ocr._start()
    except Stop:
        pass
    assert seen["args"][0] == str(python.resolve())
    # Не файл (имя из PATH) — передаётся как есть.
    ocr = SubprocessPaddleOCR(python="python", device="cpu")
    try:
        ocr._start()
    except Stop:
        pass
    assert seen["args"][0] == "python"


def test_recognizer_from_env_name_or_model_dir(tmp_path, monkeypatch):
    from wine_scanner.ocr.worker import DEFAULT_REC, recognizer_kwargs

    monkeypatch.delenv("WINE_OCR_REC", raising=False)
    assert recognizer_kwargs() == {"text_recognition_model_name": DEFAULT_REC}
    monkeypatch.setenv("WINE_OCR_REC", "PP-OCRv5_server_rec")
    assert recognizer_kwargs() == {"text_recognition_model_name": "PP-OCRv5_server_rec"}
    # Папка Paddle-модели: имя берётся из inference.yml, движок остаётся Paddle.
    model = tmp_path / "rec_ru"
    model.mkdir()
    (model / "inference.yml").write_text("Global:\n  model_name: PP-OCRv6_medium_rec\n")
    (model / "inference.json").write_text("{}")
    monkeypatch.setenv("WINE_OCR_REC", str(model))
    assert recognizer_kwargs() == {
        "text_recognition_model_name": "PP-OCRv6_medium_rec",
        "text_recognition_model_dir": str(model),
    }


def test_onnx_recognizer_runs_on_onnxruntime(tmp_path, monkeypatch):
    pytest.importorskip("paddlex")
    from wine_scanner.ocr.worker import recognizer_kwargs

    model = tmp_path / "rec_ru"
    model.mkdir()
    (model / "inference.yml").write_text("Global:\n  model_name: PP-OCRv6_medium_rec\n")
    (model / "inference.onnx").write_bytes(b"")
    monkeypatch.setenv("WINE_OCR_REC", str(model))
    kwargs = recognizer_kwargs()
    assert kwargs["text_recognition_model_dir"] == str(model)
    config = kwargs["paddlex_config"]
    assert config["SubModules"]["TextRecognition"]["engine"] == "onnxruntime"
    # Остальная конфигурация конвейера на месте: переопределение подмешано, а не подменило её.
    assert "TextDetection" in config["SubModules"]
