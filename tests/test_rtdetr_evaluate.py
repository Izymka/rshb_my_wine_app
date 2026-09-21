import torch

from scripts.train_rtdetr import evaluate


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        return None


class FakeProcessor:
    def post_process_object_detection(self, outputs, threshold, target_sizes):
        assert threshold == 0.3
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [30.0, 20.0, 70.0, 80.0]]),
                "scores": torch.tensor([0.99, 0.7]),
            }
        ]


def test_evaluation_uses_production_selection_not_highest_confidence():
    batch = {
        "pixel_values": torch.zeros(1, 3, 10, 10),
        "orig_sizes": torch.tensor([[100, 100]]),
        "gt_boxes": [torch.tensor([[30.0, 20.0, 70.0, 80.0]])],
    }
    metrics = evaluate(FakeModel(), FakeProcessor(), [batch], torch.device("cpu"))
    assert metrics["mean_iou"] == 1
    assert metrics["detection_rate"] == 1
