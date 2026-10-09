import numpy as np
import pytest

from engine import ask
from engine.memory import Memory

FRAME = np.zeros((240, 320, 3), np.uint8)
WORDS = {"remote": 0, "tv remote": 0, "mug": 1, "bicycle": 2}


def fake_embed_text(texts):
    out = np.zeros((len(texts), 4), np.float32)
    for row, text in zip(out, texts):
        row[WORDS.get(text, 3)] = 1.0
    return out


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setattr(ask, "embed_text", fake_embed_text)
    return Memory(tmp_path)


def see(memory, label, t, where):
    obj = memory.create(label, np.eye(4, dtype=np.float32)[:1], FRAME, [0, 0, 10, 10])
    s = memory.sighting(obj, t, [0, 0, 10, 10])
    memory.snapshot(s, FRAME, t, where, [])
    return obj


def test_answer_says_where_and_how_long_ago(memory):
    see(memory, "remote", 10.0, "on the bed")
    a = ask.by_text(memory, "where's my remote?", now=40.0)
    assert a["found"] and a["text"] == "Last seen 30 seconds ago on the bed."
    assert a["snapshot"].startswith("/snapshot/")


def test_unknown_object_returns_closest_names(memory):
    see(memory, "remote", 10.0, "on the bed")
    see(memory, "mug", 12.0, "on the desk")
    a = ask.by_text(memory, "where is my bicycle", now=20.0)
    assert not a["found"] and set(a["candidates"]) == {"remote", "mug"}


def test_empty_memory_and_empty_question(memory):
    assert ask.by_text(memory, "where's my remote", now=0)["text"] == "I don't remember anything yet."
    see(memory, "mug", 1.0, "on the desk")
    assert ask.by_text(memory, "where is my", now=2.0)["text"] == "What should I look for?"


def test_object_stored_twice_answers_with_the_latest_sighting(memory):
    see(memory, "remote", 5.0, "on the desk")
    later = see(memory, "remote", 30.0, "on the bed")
    assert ask.by_text(memory, "remote", now=35.0)["object_id"] == later.id


@pytest.mark.parametrize("seconds, text", [(4, "just now"), (30, "30 seconds ago"),
                                           (300, "5 minutes ago"), (7200, "2 hours ago")])
def test_ago(seconds, text):
    assert ask.ago(seconds) == text
