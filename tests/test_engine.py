import numpy as np
import pytest

from engine.engine import Engine

DIM = 8


def vector(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


class FakeDetector:
    """Plays back a scripted scene: one list of objects per frame, no surfaces."""

    def __init__(self, scene):
        self.scene = iter(scene)

    def __call__(self, frame):
        return next(self.scene), []


class FakeEmbedder:
    """Fingerprint = a fixed vector per object, keyed by the box's left edge."""

    def __init__(self, by_x1):
        self.by_x1 = by_x1

    def __call__(self, frame, boxes):
        return np.array([self.by_x1[b[0]] for b in boxes], np.float32).reshape(len(boxes), DIM)


def obj(track_id, label, box):
    return {"track_id": track_id, "label": label, "conf": 0.9, "box": box}


def run(engine, frames, fps=10):
    frame = np.zeros((480, 640, 3), np.uint8)
    out = []
    for i in range(frames):
        out.append(engine.process(frame, i / fps)[0])
    return out


@pytest.fixture
def leave_and_return(tmp_path):
    """A remote in view for 1 s, gone for 2 s, back for 1 s under a new track id;
    a mug in view the whole time."""
    mug = obj(9, "mug", [400, 100, 460, 160])
    scene = []
    for i in range(40):
        objects = [mug]
        if i < 10:
            objects.append(obj(1, "remote", [100, 100, 160, 160]))
        elif i >= 30:
            objects.append(obj(2, "remote", [100, 100, 160, 160]))
        scene.append(objects)
    embedder = FakeEmbedder({100: vector(0), 400: vector(1)})
    return Engine(tmp_path, detector=FakeDetector(scene), embedder=embedder)


def test_returning_object_keeps_its_identity_on_a_new_track(leave_and_return):
    frames = run(leave_and_return, 40)
    remote_ids = {d["object_id"] for f in frames for d in f
                  if d["label"] == "remote" and d["object_id"] is not None}
    assert len(remote_ids) == 1
    assert sum(d["reidentified"] for f in frames for d in f) == 1
    assert len(leave_and_return.memory.objects) == 2


def test_two_second_absence_starts_a_new_sighting(leave_and_return):
    run(leave_and_return, 40)
    remote = next(o for o in leave_and_return.memory.objects.values() if o.label == "remote")
    assert len(remote.sightings) == 2


def test_stillness_is_measured_against_the_camera_not_the_screen(tmp_path):
    """The head turns, so every object slides 5 px a frame; the hand also moves
    one object 25 px a frame. Only that one is moving."""
    static = {x: vector(i) for i, x in enumerate((100, 200, 300, 400))}
    scene = []
    for i in range(12):
        objects = [obj(t, "thing", [x + 5 * i, 100, x + 5 * i + 40, 140])
                   for t, x in enumerate(static, start=1)]
        objects.append(obj(9, "pen", [500 + 25 * i, 300, 540 + 25 * i, 340]))
        scene.append(objects)
    by_x1 = {x + 5 * i: v for x, v in static.items() for i in range(12)}
    by_x1.update({500 + 25 * i: vector(7) for i in range(12)})
    engine = Engine(tmp_path, detector=FakeDetector(scene), embedder=FakeEmbedder(by_x1))
    run(engine, 12)
    rested = {o.label: o.sightings[-1].rested for o in engine.memory.objects.values() if o.label == "pen"}
    things = [o.sightings[-1].rested for o in engine.memory.objects.values() if o.label == "thing"]
    assert rested == {"pen": False}
    assert things == [True] * 4
