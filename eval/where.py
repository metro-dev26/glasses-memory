"""Score "where is X?" answers on a video with written-down ground truth.

Each object is asked two ways:
  photo - by its close-up from registry/<video>.json (a crop of that frame),
          matched by DINOv3 fingerprint: no names involved
  text  - by its name, matched against the detector's labels
An answer is "on time" when its snapshot is within TOLERANCE seconds of the
truth time. Whether the place is right is checked by a person looking at the
snapshot: this script prints the paths, it does not judge the picture.

    python -m eval.where videos/sample4.mp4 eval/truth/sample4.json
"""
import json
import shutil
import sys
from pathlib import Path

import cv2

from engine.engine import Engine
from engine.video import frames, resize

TOLERANCE = 3.0          # seconds; truth times are written by hand while recording


def main(video, truth_file):
    truth = {k: v for k, v in json.loads(Path(truth_file).read_text()).items() if not k.startswith("_")}
    registry = json.loads((Path("registry") / f"{Path(video).stem}.json").read_text())
    store = Path("store") / f"eval-{Path(video).stem}"
    shutil.rmtree(store, ignore_errors=True)

    engine = Engine(store)
    for ts, frame in frames(video):
        engine.process(frame, ts)
    engine.memory.save()
    print(f"{len(engine.memory.objects)} objects in memory\n")

    cap = cv2.VideoCapture(video)
    on_time = {"photo": 0, "text": 0}
    for name, t in truth.items():
        when, (x1, y1, x2, y2) = registry[name]
        cap.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
        _, close_up = cap.read()
        photo = resize(close_up[y1:y2, x1:x2])
        print(f"{name}  (truth: {t['where']} at {t['time']}s)")
        for how, a in (("photo", engine.ask_photo(photo)), ("text", engine.ask(name))):
            if not a["found"]:
                print(f"  {how:5s}  not found, closest {a['candidates']}")
                continue
            hit = abs(a["seen_at"] - t["time"]) <= TOLERANCE
            on_time[how] += hit
            obj = engine.memory.objects[a["object_id"]]
            print(f"  {how:5s}  {'ON TIME' if hit else 'off    '}  #{obj.id} '{obj.name}' at {a['seen_at']:5.1f}s "
                  f"{a['text']!r}  {store}/snapshots/{Path(a['snapshot']).name}")
    n = len(truth)
    print(f"\non time: photo {on_time['photo']}/{n}, text {on_time['text']}/{n} (place still to be checked by eye)")


if __name__ == "__main__":
    main(*sys.argv[1:])
