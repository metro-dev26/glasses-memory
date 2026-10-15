import pytest

from engine.video import frames


def test_samples_at_ten_fps(synthetic_video):
    times = [t for t, _ in frames(synthetic_video)]
    assert len(times) == 30
    assert times[0] == 0.0
    assert all(abs((b - a) - 0.1) < 1e-6 for a, b in zip(times, times[1:]))


def test_missing_video_raises_instead_of_yielding_nothing(tmp_path):
    with pytest.raises(ValueError, match="cannot read video"):
        list(frames(tmp_path / "nope.mp4"))


def test_recorded_folder_replays_in_time_order(tmp_path):
    import cv2
    import numpy as np
    from engine.video import frames
    for ts, shade in [(100.5, 2), (100.0, 1), (101.25, 3)]:
        cv2.imwrite(str(tmp_path / f"{ts:.3f}.jpg"), np.full((8, 8, 3), shade * 60, np.uint8))
    got = list(frames(tmp_path))
    assert [t for t, _ in got] == [0.0, 0.5, 1.25]
    assert [int(f.mean()) // 60 for _, f in got] == [1, 2, 3]
