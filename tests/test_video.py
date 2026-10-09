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
