"""What the engine remembers: objects, their fingerprints, and every sighting.

No video is kept. Per object: a name, the detector's label, a gallery of
DINOv3 fingerprints, a thumbnail. Per sighting: when, the box, where in words,
and one snapshot of the moment it was left. Stored as SQLite plus one NumPy
file per gallery, so "forget everything" is deleting one folder.

    store/
      memory.db
      galleries/<object id>.npy
      thumbs/<object id>.jpg
      snapshots/<sighting id>.jpg
"""
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SIGHTING_GAP = 2.0       # seconds; a longer absence starts a new sighting


@dataclass
class Sighting:
    id: int
    object_id: int
    start: float
    end: float
    box: list
    where: str = ""
    nearby: list = field(default_factory=list)
    rested: bool = False          # was the object ever seen standing still in this sighting
    snapshot_at: float = -1.0     # time of the saved snapshot


@dataclass
class Object:
    id: int
    name: str
    label: str
    gallery: np.ndarray
    sightings: list = field(default_factory=list)

    @property
    def last_seen(self):
        return self.sightings[-1].end if self.sightings else 0.0

    @property
    def where(self):
        return self.sightings[-1].where if self.sightings else ""


class Memory:
    def __init__(self, folder):
        self.folder = Path(folder)
        for sub in ("galleries", "thumbs", "snapshots"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        self.objects = {}
        self.next_object = self.next_sighting = 1
        self.load()

    # --- writing -------------------------------------------------------------

    def create(self, label, gallery, frame, box):
        obj = Object(self.next_object, label, label, gallery)
        self.objects[obj.id] = obj
        self.next_object += 1
        x1, y1, x2, y2 = box
        thumb = frame[y1:y2, x1:x2]
        if thumb.size:
            cv2.imwrite(str(self.thumb_path(obj.id)), resize_max(thumb, 160))
        return obj

    def sighting(self, obj, ts, box):
        """The open sighting of `obj` at time `ts`, extended or newly started."""
        last = obj.sightings[-1] if obj.sightings else None
        if last and ts - last.end <= SIGHTING_GAP:
            last.end, last.box = ts, box
            return last
        s = Sighting(self.next_sighting, obj.id, ts, ts, box)
        self.next_sighting += 1
        obj.sightings.append(s)
        return s

    def snapshot(self, s, frame, ts, where, nearby):
        """Save `frame` as the picture of where this sighting's object is."""
        x1, y1, x2, y2 = s.box
        shot = frame.copy()
        cv2.rectangle(shot, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(str(self.snapshot_path(s.id)), shot, [cv2.IMWRITE_JPEG_QUALITY, 80])
        s.snapshot_at, s.where, s.nearby = ts, where, nearby

    def rename(self, object_id, name):
        self.objects[object_id].name = name

    def forget_all(self):
        shutil.rmtree(self.folder, ignore_errors=True)
        self.__init__(self.folder)

    # --- paths ---------------------------------------------------------------

    def thumb_path(self, object_id):
        return self.folder / "thumbs" / f"{object_id}.jpg"

    def snapshot_path(self, sighting_id):
        return self.folder / "snapshots" / f"{sighting_id}.jpg"

    # --- persistence ---------------------------------------------------------

    def save(self):
        db = sqlite3.connect(self.folder / "memory.db")
        db.executescript("""
            DROP TABLE IF EXISTS objects; DROP TABLE IF EXISTS sightings;
            CREATE TABLE objects (id INTEGER PRIMARY KEY, name TEXT, label TEXT);
            CREATE TABLE sightings (id INTEGER PRIMARY KEY, object_id INTEGER, start REAL,
                end REAL, box TEXT, "where" TEXT, nearby TEXT, rested INTEGER, snapshot_at REAL);
        """)
        for o in self.objects.values():
            db.execute("INSERT INTO objects VALUES (?, ?, ?)", (o.id, o.name, o.label))
            np.save(self.folder / "galleries" / f"{o.id}.npy", o.gallery)
            for s in o.sightings:
                db.execute("INSERT INTO sightings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (s.id, s.object_id, s.start, s.end, json.dumps(s.box), s.where,
                            json.dumps(s.nearby), s.rested, s.snapshot_at))
        db.commit()
        db.close()

    def load(self):
        path = self.folder / "memory.db"
        if not path.exists():
            return
        db = sqlite3.connect(path)
        for oid, name, label in db.execute("SELECT * FROM objects"):
            gallery = np.load(self.folder / "galleries" / f"{oid}.npy")
            self.objects[oid] = Object(oid, name, label, gallery)
        for sid, oid, start, end, box, where, nearby, rested, snap in db.execute(
                "SELECT * FROM sightings ORDER BY start"):
            self.objects[oid].sightings.append(
                Sighting(sid, oid, start, end, json.loads(box), where, json.loads(nearby),
                         bool(rested), snap))
        db.close()
        self.next_object = max(self.objects, default=0) + 1
        self.next_sighting = max((s.id for o in self.objects.values() for s in o.sightings),
                                 default=0) + 1


def resize_max(image, side):
    h, w = image.shape[:2]
    scale = side / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else image
