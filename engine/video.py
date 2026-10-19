"""Read a recorded video the way the phone page sends it: about 10 frames a
second, long side resized to 960 px. A folder of frames saved by the live
server (--record) replays as it was recorded."""
from pathlib import Path

import cv2

FPS = 10
LONG_SIDE = 960


def resize(frame, long_side=LONG_SIDE):
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def frames(path, fps=FPS):
    """Yield (time in seconds, BGR frame), sampled at `fps`."""
    if Path(path).is_dir():
        yield from recorded_frames(path)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
        raise ValueError(f"cannot read video: {path}")
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            yield n / cap.get(cv2.CAP_PROP_FPS), resize(frame)
        n += 1


def recorded_frames(folder):
    """Frames saved by the live server, named <unix time>.jpg: every one, at its
    own time from the first, since they are exactly what the engine processed."""
    files = sorted(Path(folder).glob("*.jpg"), key=lambda f: float(f.stem))
    if not files:
        raise ValueError(f"no recorded frames in: {folder}")
    start = float(files[0].stem)
    for f in files:
        frame = cv2.imread(str(f))
        if frame is not None:
            yield float(f.stem) - start, frame
