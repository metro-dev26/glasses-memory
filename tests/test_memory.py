import numpy as np

from engine.memory import SIGHTING_GAP, Memory

FRAME = np.zeros((240, 320, 3), np.uint8)
BOX = [10, 10, 60, 60]


def remember_remote(folder):
    memory = Memory(folder)
    obj = memory.create("remote", np.eye(4, dtype=np.float32)[:2], FRAME, BOX)
    s = memory.sighting(obj, 1.0, BOX)
    memory.snapshot(s, FRAME, 1.0, "on the desk", ["mug"])
    return memory, obj


def test_short_absence_extends_the_sighting_long_one_starts_a_new_one(tmp_path):
    memory, obj = remember_remote(tmp_path)
    memory.sighting(obj, 1.0 + SIGHTING_GAP, BOX)
    assert len(obj.sightings) == 1
    memory.sighting(obj, 1.0 + 2 * SIGHTING_GAP + 0.1, BOX)
    assert len(obj.sightings) == 2


def test_save_and_load_round_trip(tmp_path):
    memory, obj = remember_remote(tmp_path)
    memory.rename(obj.id, "tv remote")
    memory.save()
    loaded = Memory(tmp_path)
    again = loaded.objects[obj.id]
    assert (again.name, again.label) == ("tv remote", "remote")
    assert (again.gallery == obj.gallery).all()
    [s] = again.sightings
    assert (s.where, s.nearby, s.box) == ("on the desk", ["mug"], BOX)
    assert loaded.snapshot_path(s.id).exists()
    assert loaded.create("mug", obj.gallery, FRAME, BOX).id == obj.id + 1


def test_forget_all(tmp_path):
    memory, _ = remember_remote(tmp_path)
    memory.save()
    memory.forget_all()
    assert memory.objects == {}
    assert Memory(tmp_path).objects == {}
