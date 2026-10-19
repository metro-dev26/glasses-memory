import numpy as np

from engine.memory import Object
from engine.rounds import Round, Rounds, Seen, compare, moved, surface


def objects(*labels):
    return {i: Object(i, label, label, np.zeros((1, 4))) for i, label in enumerate(labels, 1)}


def rnd(id, seen):
    return Round(id, f"Round {id}", 0.0, seen=seen)


def url(sid):
    return f"/snapshot/{sid}.jpg"


def kinds(changes):
    return sorted((c["kind"], c["object_id"]) for c in changes)


def test_surface_is_read_from_where():
    assert surface("on the desk, next to the cup") == "desk"
    assert surface("next to the cup") == ""
    assert surface("") == ""


def test_moved_needs_evidence_on_both_sides():
    assert moved(Seen(1, "on the desk"), Seen(2, "on the bed"))
    assert not moved(Seen(1, "on the desk"), Seen(2, ""))            # surface not seen this time
    assert not moved(Seen(1, "on the desk", ["cup"]), Seen(2, "on the desk", ["laptop"]))  # neighbours alone


def test_missing_new_and_moved():
    objs = objects("cup", "stapler", "laptop", "helmet")
    before = rnd(1, {1: Seen(10, "on the desk"), 2: Seen(11, "on the desk"), 3: Seen(12, "on the desk")})
    after = rnd(2, {1: Seen(20, "on the desk"), 3: Seen(21, "on the bed"), 4: Seen(22, "on the desk")})
    changes = compare(before, after, objs, url)
    assert kinds(changes) == [("missing", 2), ("moved", 3), ("new", 4)]
    assert changes[0]["kind"] == "missing"                      # the alarm comes first
    missing = changes[0]
    assert missing["before"] == "/snapshot/11.jpg" and missing["after"] == ""


def test_same_label_missing_and_new_is_one_probable_move():
    # re-identification failed: the stapler came back as object 2
    objs = objects("stapler", "stapler")
    before = rnd(1, {1: Seen(10, "on the desk")})
    after = rnd(2, {2: Seen(20, "on the bed")})
    changes = compare(before, after, objs, url)
    assert kinds(changes) == [("moved", 1)]
    assert "Probably" in changes[0]["text"]


def test_ambiguous_labels_are_not_paired():
    objs = objects("bottle", "bottle", "bottle")
    before = rnd(1, {1: Seen(10), 2: Seen(11)})
    after = rnd(2, {3: Seen(20)})
    assert kinds(compare(before, after, objs, url)) == [("missing", 1), ("missing", 2), ("new", 3)]


def test_rounds_are_stored_and_survive_a_restart(tmp_path):
    objs = objects("cup")
    rounds = Rounds(tmp_path)
    assert rounds.last() is None
    rounds.start("Morning")
    obj = objs[1]
    from engine.memory import Sighting
    obj.sightings.append(Sighting(7, 1, 0.0, 1.0, [0, 0, 1, 1], "on the desk", ["pen"]))
    rounds.see(obj)
    r, previous = rounds.end()
    assert previous is None and r.name == "Morning"

    again = Rounds(tmp_path)
    last = again.last()
    assert last.id == r.id and last.seen[1] == Seen(7, "on the desk", ["pen"])
    assert again.start().id == r.id + 1
