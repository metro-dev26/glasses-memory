"""The live engine: one frame in, boxes with object ids out, memory updated.

Per frame: detect + track (YOLOE, ByteTrack) -> identify new tracks (DINOv3)
-> record sightings -> save a snapshot when an object comes to rest.
The same code runs on a recorded video (engine/run.py) and, later, on the
phone's live stream (the server).
"""
import math
import time

import numpy as np

from engine import ask, identify
from engine.detect import Detector
from engine.identify import Embedder
from engine.memory import Memory

MIN_TRACK_AGE = 5        # frames a new track must survive before it is identified (flickers die sooner)
VIEW_EVERY = 10          # frames between gallery updates for an identified track (1 s at 10 fps)
TRACK_TTL = 30           # frames after which a vanished track is dropped (ByteTrack keeps 30)
STILL = 0.03             # motion below this fraction of the box diagonal per frame = not moving
STILL_FRAMES = 3         # consecutive still frames before an object counts as put down
REST_SNAPSHOT_EVERY = 0.5    # seconds between snapshots of a resting object
MOVING_SNAPSHOT_EVERY = 1.0  # seconds between fallback snapshots before it ever rests


class Track:
    def __init__(self, label):
        self.label = label
        self.age = 0
        self.fingerprints = []
        self.object_id = None
        self.center = None
        self.still = 0
        self.last_frame = 0


class Engine:
    def __init__(self, store="store", detector=None, embedder=None):
        """`detector` and `embedder` default to YOLOE and DINOv3; tests pass stand-ins."""
        self.detector = detector or Detector()
        self.embedder = embedder or Embedder()
        self.memory = Memory(store)
        self.tracks = {}
        self.frame_no = 0
        self.fps = 0.0
        self.now = 0.0

    def process(self, frame, ts=None):
        """Run one BGR frame. Returns (detections, changed object ids)."""
        started = time.perf_counter()
        ts = time.time() if ts is None else ts
        self.now = ts
        self.frame_no += 1
        objects, surfaces = self.detector(frame)
        changed = set()

        for o in objects:
            tr = self.tracks.setdefault(o["track_id"], Track(o["label"]))
            tr.age += 1
            tr.last_frame = self.frame_no
        self.update_motion(objects)
        self.fingerprint(frame, objects)

        detections = []
        visible = {self.tracks[o["track_id"]].object_id for o in objects} - {None}
        for o in objects:
            tr = self.tracks[o["track_id"]]
            det = {**o, "object_id": tr.object_id, "name": o["label"], "reidentified": False}
            if tr.object_id is None and tr.age >= MIN_TRACK_AGE:
                self.identify_track(tr, frame, o, det, visible, changed)
            if tr.object_id is not None:
                obj = self.memory.objects[tr.object_id]
                det["name"] = obj.name
                self.observe(obj, tr, frame, ts, o, objects, surfaces, changed)
            detections.append(det)

        self.forget_old_tracks()
        self.fps = 0.9 * self.fps + 0.1 / max(time.perf_counter() - started, 1e-3)
        return detections, changed

    # --- per-frame steps -------------------------------------------------------

    def update_motion(self, objects):
        """How much each object moved in the world, not on screen: its on-screen
        shift minus the camera's shift, estimated as the median shift of the
        other objects in view (the camera on a head never stands still)."""
        shifts = {}
        for o in objects:
            tr = self.tracks[o["track_id"]]
            c = center(o["box"])
            if tr.center is not None and tr.age > 1:
                shifts[o["track_id"]] = (c[0] - tr.center[0], c[1] - tr.center[1])
            tr.center = c
        for tid, (dx, dy) in shifts.items():
            others = [s for t, s in shifts.items() if t != tid]
            if others:
                dx -= float(np.median([s[0] for s in others]))
                dy -= float(np.median([s[1] for s in others]))
            box = next(o["box"] for o in objects if o["track_id"] == tid)
            tr = self.tracks[tid]
            tr.still = tr.still + 1 if math.hypot(dx, dy) < STILL * diagonal(box) else 0

    def fingerprint(self, frame, objects):
        """Embed new tracks on each of their first frames, and identified tracks
        once a second so their object's gallery collects new viewpoints."""
        def due(t):
            return t.age <= MIN_TRACK_AGE if t.object_id is None else t.age % VIEW_EVERY == 0

        todo = [o for o in objects if due(self.tracks[o["track_id"]])]
        for o, f in zip(todo, self.embedder(frame, [o["box"] for o in todo])):
            tr = self.tracks[o["track_id"]]
            if tr.object_id is None:
                tr.fingerprints.append(f)
            else:
                obj = self.memory.objects[tr.object_id]
                obj.gallery = identify.add_view(obj.gallery, f)

    def identify_track(self, tr, frame, o, det, visible, changed):
        """A track has lasted long enough: is it an object we already know?"""
        query = np.mean(tr.fingerprints, axis=0)
        query /= np.linalg.norm(query)
        galleries = {oid: obj.gallery for oid, obj in self.memory.objects.items()}
        oid, _ = identify.best_match(query, galleries, exclude=visible)
        if oid is None:
            gallery = np.zeros((0, len(query)), dtype=np.float32)
            for f in tr.fingerprints:
                gallery = identify.add_view(gallery, f)
            oid = self.memory.create(tr.label, gallery, frame, o["box"]).id
            changed.add(oid)
        else:
            det["reidentified"] = True
            det["last_seen_ago_s"] = round(self.now - self.memory.objects[oid].last_seen)
        tr.object_id = det["object_id"] = oid
        tr.fingerprints = []
        visible.add(oid)

    def observe(self, obj, tr, frame, ts, o, objects, surfaces, changed):
        """Record the sighting, and snapshot the object when it is at rest."""
        s = self.memory.sighting(obj, ts, o["box"])
        resting = tr.still >= STILL_FRAMES
        due = (resting and ts - s.snapshot_at >= REST_SNAPSHOT_EVERY) or \
              (not s.rested and ts - s.snapshot_at >= MOVING_SNAPSHOT_EVERY)
        if due:
            where, nearby = describe(o, objects, surfaces, self.tracks, self.memory)
            if where != s.where:
                changed.add(obj.id)
            self.memory.snapshot(s, frame, ts, where, nearby)
            s.rested = s.rested or resting

    def forget_all(self):
        """Wipe the memory and the open tracks, so no half-identified track
        carries an old object id into the new memory."""
        self.memory.forget_all()
        self.tracks.clear()

    def warm_up(self):
        """Load the text model now, so the first question is not slow."""
        ask.embed_text(["warm up"])

    def forget_old_tracks(self):
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if self.frame_no - t.last_frame <= TRACK_TTL}

    # --- questions -------------------------------------------------------------

    def ask(self, query):
        return ask.by_text(self.memory, query, self.now)

    def ask_photo(self, image):
        return ask.by_photo(self.memory, self.detector, self.embedder, image, self.now)

    def stats(self, in_view):
        return {"fps": round(self.fps, 1), "in_view": in_view,
                "in_memory": len(self.memory.objects)}


def describe(o, objects, surfaces, tracks, memory):
    """Where an object is, in words, from what is around it in this frame:
    the surface under it ("on the desk") and the closest other objects."""
    x1, y1, x2, y2 = o["box"]
    foot = ((x1 + x2) / 2, y2)
    under = [s for s in surfaces if s["track_id"] != o["track_id"]
             and s["box"][0] <= foot[0] <= s["box"][2] and s["box"][1] <= foot[1] <= s["box"][3]]
    surface = min(under, key=lambda s: area(s["box"]))["label"] if under else None

    near = []
    for other in sorted(objects, key=lambda p: math.dist(center(p["box"]), center(o["box"]))):
        oid = tracks[other["track_id"]].object_id
        if other is o or oid is None:
            continue
        name = memory.objects[oid].name
        if name not in near and name != surface and name != memory.objects[tracks[o["track_id"]].object_id].name:
            near.append(name)
        if len(near) == 2:
            break

    parts = ([f"on the {surface}"] if surface else []) + \
            ([f"next to the {' and the '.join(near)}"] if near else [])
    return ", ".join(parts), near


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def diagonal(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])
