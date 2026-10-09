"""Read a recorded video the way the phone page sends it: about 10 frames a
second, long side resized to 960 px."""
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
    cap = cv2.VideoCapture(str(path))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            yield n / cap.get(cv2.CAP_PROP_FPS), resize(frame)
        n += 1
