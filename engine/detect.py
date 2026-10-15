"""Detect every object in a frame and link boxes across frames into tracks.

YOLOE looks for the everyday things in VOCAB (text prompts, no registration),
and ByteTrack gives each box a track id that stays the same while the object
stays in view.

Prompt-free mode (yoloe-*-pf.pt) was tried first: it picks names from ~4,500
built-in words and, on a live phone test, called a helmet "alarm", a weighing
scale "smartphone" and a router "phonebook". Asking by text only works through
names, so a short list of things people actually lose names them far better.
An object missing from VOCAB is not detected at all: add it here.
"""
from pathlib import Path

import torch
from ultralytics import YOLOE

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "models" / "yoloe-11l-seg.pt"   # large: ~23 ms a frame on the 3050, names
                                               # the helmet and power bank that small missed
IMGSZ = 640
MIN_CONF = 0.25          # ultralytics' default; the dashboard dims anything < 0.4
QUANTIZE = 16 if torch.cuda.is_available() else None   # fp16 on GPU; on CPU fp16 is ~40x slower than fp32
MAX_AREA = 0.4           # boxes covering more of the frame are walls, rooms, the bed you sit on
IGNORE = {"person", "man", "woman", "boy", "girl", "child", "hand", "finger", "arm",
          "leg", "foot", "face", "head", "skin", "wall", "ceiling", "room", "floor",
          "bedroom", "living room", "office", "home", "house", "interior"}
VOCAB = ["helmet", "weighing scale", "power bank", "router", "mug", "cup", "bottle",
         "water bottle", "computer mouse", "keyboard", "laptop", "phone", "tablet",
         "remote control", "headphones", "earphones", "charger", "cable", "keys", "wallet",
         "watch", "glasses", "sunglasses", "book", "notebook", "pen", "pencil", "stapler",
         "scissors", "tape", "box", "bag", "backpack", "shoe", "cap", "towel", "pillow", "lamp",
         "fan", "speaker", "tripod", "perfume", "deodorant", "comb", "toothbrush", "plate",
         "bowl", "spoon", "chair", "table", "desk", "bed", "shelf", "cabinet", "monitor",
         "printer", "plant", "clock", "camera", "guitar", "dumbbell", "medicine", "jar",
         "tin can", "toy", "cloth",
         "person", "hand"]   # detected only so IGNORE can drop them: without these, a face
                             # was forced into the nearest word ("cabinet") and stored
SURFACES = {"table", "desk", "bed", "shelf", "bookshelf", "cabinet", "cupboard", "counter",
            "countertop", "nightstand", "couch", "sofa", "chair", "stool", "drawer",
            "dresser", "bench", "tray", "box", "notebook", "book", "pillow", "carpet", "rug"}


class Detector:
    def __init__(self, model=MODEL):
        self.outlines = {}       # track id -> outline polygon of the last frame, for the embedder
        self.photo_outlines = []
        self.model = YOLOE(str(model))
        self.model.set_classes(VOCAB, self.model.get_text_pe(VOCAB))

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
        self.outlines = {}
        if boxes.id is None:
            return [], []
        outlines = result.masks.xy if getattr(result, "masks", None) is not None else [None] * len(boxes)
        objects, surfaces = [], []
        for box, cls, conf, tid, outline in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(),
                                                boxes.conf.tolist(), boxes.id.int().tolist(),
                                                outlines):
            self.outlines[tid] = outline
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
        outlines = result.masks.xy if getattr(result, "masks", None) is not None else [None] * len(result.boxes.xyxy)
        kept = [([round(v) for v in box], outline) for box, cls, outline in
                zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist(), outlines)
                if result.names[cls] not in IGNORE]
        self.photo_outlines = [outline for _, outline in kept]
        return [box for box, _ in kept]

    def reset(self):
        """Forget all tracks, e.g. before a new video."""
        for tracker in getattr(self.model.predictor, "trackers", None) or []:
            tracker.reset()
