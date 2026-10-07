"""Show the memory what each object looks like: scrub to a close-up, drag a box, name it.

    python register.py videos/myvideo.mp4

Keys:  a / d   back / forward 0.5 s     A / D   back / forward 3 s
       space   draw a box on this frame (drag, then Enter), then type the name in the terminal
       u       undo the last object     q       save and quit
Writes registry/<video>.json: object -> [close-up time (s), box (x1, y1, x2, y2)].
"""
import json
import sys
from pathlib import Path

import cv2

from track import FPS, LazyFrames

ROOT = Path(__file__).parent
WINDOW = "register (space = box, q = save)"


def draw(frame, i, registry):
    img = frame.copy()
    for name, (t, (x1, y1, x2, y2)) in registry.items():
        if int(t * FPS) == i:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"{i / FPS:.1f}s  |  {len(registry)} registered", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main(video):
    frames = LazyFrames(video)
    out = ROOT / "registry" / f"{Path(video).stem}.json"
    registry = json.loads(out.read_text()) if out.exists() else {}
    i = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        frame = frames.raw(i)
        cv2.imshow(WINDOW, draw(frame, i, registry))
        key = chr(cv2.waitKey(0) & 0xFF)
        steps = {"a": -5, "d": 5, "A": -30, "D": 30}
        if key in steps:
            i = min(max(0, i + steps[key]), len(frames) - 1)
        elif key == " ":
            x, y, w, h = cv2.selectROI(WINDOW, frame, showCrosshair=False)
            if w and h:
                name = input("name of this object (e.g. 'yellow stapler'): ").strip().lower()
                if name:
                    registry[name] = [round(i / FPS, 1), [int(x), int(y), int(x + w), int(y + h)]]
                    print(f"  registered {name} at {i / FPS:.1f}s")
        elif key == "u" and registry:
            print(f"  removed {registry.popitem()[0]}")
        elif key == "q":
            break
    cv2.destroyAllWindows()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(registry, indent=1))
    print(f"wrote {out} ({len(registry)} objects)")


if __name__ == "__main__":
    main(sys.argv[1])
