import numpy as np

from engine.identify import GALLERY_SIZE, add_view, best_match

DIM = 8


def unit(*weights):
    v = np.zeros(DIM, np.float32)
    v[:len(weights)] = weights
    return v / np.linalg.norm(v)


A, B = unit(1), unit(0, 1)


def test_clear_match():
    assert best_match(A, {1: A[None], 2: B[None]}) == (1, 1.0)


def test_weak_match_is_not_a_match():
    oid, best = best_match(unit(1, 1.2), {1: A[None]})
    assert oid is None and best < 0.7


def test_object_in_view_under_another_track_cannot_match():
    assert best_match(A, {1: A[None]}, exclude={1})[0] is None


def test_close_call_between_different_objects_is_no_match():
    halfway = unit(1, 0.95)
    assert best_match(halfway, {1: A[None], 2: B[None]})[0] is None


def test_runner_up_that_looks_like_the_winner_does_not_block_the_match():
    stored_twice = unit(1, 0.05)
    assert best_match(A, {1: A[None], 2: stored_twice[None]})[0] == 1


def test_empty_memory():
    assert best_match(A, {}) == (None, 0.0)


def test_gallery_keeps_new_viewpoints_only():
    gallery = add_view(np.zeros((0, DIM), np.float32), A)
    assert len(add_view(gallery, unit(1, 0.1))) == 1
    assert len(add_view(gallery, B)) == 2


def test_gallery_is_capped_at_the_latest_views():
    gallery = np.zeros((0, 64), np.float32)
    views = np.eye(64, dtype=np.float32)[:GALLERY_SIZE + 3]
    for v in views:
        gallery = add_view(gallery, v)
    assert len(gallery) == GALLERY_SIZE
    assert (gallery[-1] == views[-1]).all()
