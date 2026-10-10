"""Detect every object in a frame and link boxes across frames into tracks.

YOLOE in prompt-free mode finds objects from its built-in vocabulary (no
registration), and ByteTrack gives each box a track id that stays the same
while the object stays in view.
"""
from pathlib import Path

import torch
from ultralytics import YOLOE

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "models" / "yoloe-11s-seg-pf.pt"   # phase 0: 56 fps on the 3050
IMGSZ = 640
MIN_CONF = 0.25          # ultralytics' default; the dashboard dims anything < 0.4
QUANTIZE = 16 if torch.cuda.is_available() else None   # fp16 on GPU; on CPU fp16 is ~40x slower than fp32
MAX_AREA = 0.4           # boxes covering more of the frame are walls, rooms, the bed you sit on
IGNORE = {"person", "man", "woman", "boy", "girl", "child", "hand", "finger", "arm",
          "leg", "foot", "face", "head", "skin", "wall", "ceiling", "room", "floor",
          "bedroom", "living room", "office", "home", "house", "interior"}
SURFACES = {"table", "desk", "bed", "shelf", "bookshelf", "cabinet", "cupboard", "counter",
            "countertop", "nightstand", "couch", "sofa", "chair", "stool", "drawer",
            "dresser", "bench", "tray", "box", "notebook", "book", "pillow", "carpet", "rug"}


class Detector:
    def __init__(self, model=MODEL):
        self.model = YOLOE(str(model))

    def __call__(self, frame):
        """Return (objects, surfaces) for one BGR frame.
        objects: tracked things worth remembering, each {track_id, label, conf, box}.
        surfaces: things objects rest on, used to describe where ("on the desk")."""
        h, w = frame.shape[:2]
        # agnostic_nms: YOLOE can put two boxes with two labels on one object
        # ("remote" and "phone"); suppressing overlaps across labels keeps one.
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=IMGSZ,
                                  conf=MIN_CONF, agnostic_nms=True, quantize=QUANTIZE,
                                  verbose=False)[0]
        boxes = result.boxes
        if boxes.id is None:
            return [], []
        objects, surfaces = [], []
        for box, cls, conf, tid in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(),
                                       boxes.conf.tolist(), boxes.id.int().tolist()):
            label = result.names[cls]
            det = {"track_id": tid, "label": label, "conf": round(conf, 3),
                   "box": [round(v) for v in box]}
            if label in SURFACES:
                surfaces.append(det)
            x1, y1, x2, y2 = box
            if label in IGNORE or (x2 - x1) * (y2 - y1) > MAX_AREA * w * h:
                continue
            objects.append(det)
        return objects, surfaces

    def boxes(self, image):
        """Boxes of the objects in a single photo. Uses predict, not track, so a
        photo never disturbs the live tracks."""
        result = self.model.predict(image, imgsz=IMGSZ, conf=MIN_CONF, agnostic_nms=True,
                                    quantize=QUANTIZE, verbose=False)[0]
        return [[round(v) for v in box] for box, cls in
                zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist())
                if result.names[cls] not in IGNORE]

    def reset(self):
        """Forget all tracks, e.g. before a new video."""
        for tracker in getattr(self.model.predictor, "trackers", None) or []:
            tracker.reset()
