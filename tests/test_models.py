"""Smoke tests with the real models. They run on CPU or GPU and download the
weights on first use, so they are slow: skip with -m "not models"."""
import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from engine.detect import Detector
from engine.identify import Embedder

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def street():
    return cv2.imread(str(ASSETS / "bus.jpg"))


def test_detector_finds_tracked_objects(street):
    objects, _ = Detector()(street)
    assert objects
    h, w = street.shape[:2]
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    assert len({o["track_id"] for o in objects}) == len(objects)


def test_embedder_scores_the_same_crop_above_noise(street):
    embedder = Embedder()
    noise = np.random.default_rng(0).integers(0, 255, street.shape, dtype=np.uint8)
    box = [50, 230, 800, 740]
    a, shifted = embedder(street, [box, [58, 235, 808, 745]])
    n = embedder(noise, [box])[0]
    assert abs(np.linalg.norm(a) - 1) < 1e-3
    assert a @ shifted > 0.8 > a @ n
