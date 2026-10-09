import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """3 s of 30 fps, 320x240 video: a white square sliding right on black."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(90):
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[100:140, 10 + i * 2:50 + i * 2] = 255
        writer.write(frame)
    writer.release()
    return path
