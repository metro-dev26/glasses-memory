"""Glasses memory: remember where objects were last seen in first-person video.

Two steps:
  index  - run the detector over a video once, save every detection + an
           appearance embedding (CLIP) of each detected crop.
  ask    - answer "where did I last see X?" from the saved index, rejecting
           false sightings that don't look like the object or don't persist.

    python memory.py index videos/sample2.mp4 "yellow stapler" "blue pen"
    python memory.py ask videos/sample2.mp4 "yellow stapler"

If refs/<label>.jpg exists (a close-up photo of YOUR object), it becomes the
reference appearance. Otherwise we fall back to the detector's own top crops.
"""
import json
import sys
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOWorld

ROOT = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_STEP = 3        # look at every 3rd frame: 30 fps video -> 10 checks/sec
MIN_CONF = 0.15       # detector score below this is ignored outright
MIN_SIMILARITY = 0.80 # crop must look this much like the reference crop
MIN_RUN = 3           # need this many consistent checks in a row (~0.3 s)
MAX_GAP = 1.0         # seconds; sightings closer than this belong to one run


def load_models(prompts):
    detector = YOLOWorld(str(ROOT / "models" / "yolov8s-worldv2.pt"))
    detector.set_classes(prompts)
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return detector, clip_model, preprocess


def embed(clip_model, preprocess, bgr_crop):
    """CLIP embedding of an image crop, normalised to length 1."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        emb = clip_model.encode_image(preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE))
    return (emb / emb.norm()).squeeze().float().cpu().numpy()


def reference_from_photo(label):
    """Embed the object in refs/<label>.jpg, cropped to the detector's best box."""
    photo = ROOT / "refs" / f"{label}.jpg"
    if not photo.exists():
        return None
    detector, clip_model, preprocess = load_models([label])
    img = cv2.imread(str(photo))
    boxes = detector.predict(img, device=DEVICE, conf=0.05, verbose=False)[0].boxes
    if len(boxes):
        x1, y1, x2, y2 = map(int, boxes.xyxy[boxes.conf.argmax()].tolist())
        img = img[y1:y2, x1:x2]
    return embed(clip_model, preprocess, img)


def index_path(video):
    return ROOT / "out" / (Path(video).stem + ".index.json")


def build_index(video, prompts):
    detector, clip_model, preprocess = load_models(prompts)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detections = []
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % FRAME_STEP == 0:
            result = detector.predict(frame, device=DEVICE, conf=MIN_CONF, verbose=False)[0]
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                emb = embed(clip_model, preprocess, frame[y1:y2, x1:x2]).tolist()
                detections.append({
                    "label": prompts[int(cls)],
                    "time": round(frame_no / fps, 2),
                    "frame": frame_no,
                    "conf": round(float(conf), 3),
                    "box": [x1, y1, x2, y2],
                    "emb": emb,
                })
        frame_no += 1

    out = index_path(video)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"video": str(video), "prompts": prompts, "detections": detections}))
    print(f"indexed {frame_no} frames, {len(detections)} detections -> {out}")


def group_into_runs(sightings):
    """Split time-sorted sightings into runs separated by gaps > MAX_GAP."""
    runs = []
    for s in sightings:
        if runs and s["time"] - runs[-1][-1]["time"] <= MAX_GAP:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def ask(video, label):
    index = json.loads(index_path(video).read_text())
    sightings = sorted((d for d in index["detections"] if d["label"] == label), key=lambda d: d["time"])
    if not sightings:
        print(f"never saw '{label}'")
        return

    ref = reference_from_photo(label)
    if ref is not None:
        print("reference           : your photo")
    else:
        # No photo: assume the 3 most confident detections are the real object.
        best = sorted(sightings, key=lambda d: d["conf"], reverse=True)[:3]
        ref = np.mean([d["emb"] for d in best], axis=0)
        ref /= np.linalg.norm(ref)
        print("reference           : detector's top 3 crops (no photo in refs/)")

    for s in sightings:
        s["sim"] = float(np.dot(s["emb"], ref))
    kept = [s for s in sightings if s["sim"] >= MIN_SIMILARITY]
    runs = [r for r in group_into_runs(kept) if len(r) >= MIN_RUN]

    naive = sightings[-1]
    print(f"naive last sighting : {naive['time']:5.1f}s  (conf {naive['conf']:.2f}, sim {naive['sim']:.2f})")
    if not runs:
        print("no sighting survived the filters")
        return
    last = max(runs[-1], key=lambda d: d["conf"])
    print(f"filtered last seen  : {runs[-1][0]['time']:5.1f}-{runs[-1][-1]['time']:.1f}s  "
          f"(best frame {last['time']}s, conf {last['conf']:.2f}, sim {last['sim']:.2f})")

    cap = cv2.VideoCapture(index["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, last["frame"])
    ok, frame = cap.read()
    x1, y1, x2, y2 = last["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(frame, f"{label} @ {last['time']}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    shot = ROOT / "out" / f"{Path(video).stem}__{label.replace(' ', '_')}.jpg"
    cv2.imwrite(str(shot), frame)
    print(f"snapshot            : {shot}")


if __name__ == "__main__":
    cmd, video, *rest = sys.argv[1:]
    if cmd == "index":
        build_index(video, rest)
    elif cmd == "ask":
        ask(video, " ".join(rest))
