"""PaddleOCR adapter: parsing stays independent from model downloads."""

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
