import numpy as np
import torch

from engine.detect import Detector


class RecordingModel:
    """Stands in for YOLOE: records the arguments, returns no boxes."""

    def __init__(self):
        self.calls = []

    def _record(self, kwargs):
        self.calls.append(kwargs)
        result = type("Result", (), {"boxes": type("Boxes", (), {"id": None, "xyxy": torch.zeros(0, 4),
                                                                 "cls": torch.zeros(0)})(),
                                     "names": {}})()
        return [result]

    def track(self, frame, **kwargs):
        return self._record(kwargs)

    def predict(self, image, **kwargs):
        return self._record(kwargs)


def test_half_precision_only_on_gpu():
    detector = Detector.__new__(Detector)
    detector.model = RecordingModel()
    frame = np.zeros((64, 64, 3), np.uint8)
    detector(frame)
    detector.boxes(frame)
    expected = torch.cuda.is_available()
    assert [c["half"] for c in detector.model.calls] == [expected, expected]
