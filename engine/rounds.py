"""Safety rounds: walk the same space again and report what changed.

An operator's inspection round, with a memory. A round records which objects
were seen during it, and for each the last sighting (where it was, its
snapshot). Ending a round compares it with the round before it:

    missing   seen last round, not seen this round
    new       seen this round, not seen last round
    moved     seen in both, but on a different surface

Re-identification is not perfect: an object that was picked up and put down
often comes back as a new object. So when a missing and a new object share a
label, and are the only one of that label on each side, they are reported as
one probable move instead of two alarms.

Rounds are kept in the store's memory.db, in their own tables, so they survive
a restart and are wiped with "forget everything".
"""
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Seen:
    sighting: int              # the last sighting of the object during the round
    where: str = ""
    nearby: list = field(default_factory=list)


@dataclass
class Round:
    id: int
    name: str
    started: float
    ended: float = 0.0
    seen: dict = field(default_factory=dict)    # object id -> Seen


def surface(where):
    """"on the desk, next to the cup" -> "desk"; "" when no surface was seen."""
    head = where.split(",")[0].strip()
    return head[len("on the "):] if head.startswith("on the ") else ""


def moved(before: Seen, after: Seen):
    """True when the same object is on a different surface. Both rounds must
    have seen a surface under it: an unseen surface is not a different one.
    Changed neighbours are not used: they are names from the same unsure
    labels ("next to the bottle" with six bottles), and on a replay of desk1
    they flagged 13 objects that had not moved."""
    a, b = surface(before.where), surface(after.where)
    return bool(a and b and a != b)


def describe_place(seen: Seen):
    return seen.where or "somewhere not described"


def compare(previous: Round, current: Round, objects, image):
    """The changes from `previous` to `current`, as dashboard `change` messages.
    `objects` maps object id -> engine Object (for names and labels); `image`
    turns a sighting id into an image URL, or "" if there is none."""
    changes = []

    def change(kind, oid, text, before="", after=""):
        changes.append({"type": "change", "round_id": current.id, "kind": kind,
                        "object_id": oid, "name": objects[oid].name, "text": text,
                        "before": before, "after": after})

    gone = [oid for oid in previous.seen if oid not in current.seen and oid in objects]
    came = [oid for oid in current.seen if oid not in previous.seen and oid in objects]

    # A missing and a new object with the same, unique label: probably one object, moved.
    paired = set()
    for label in {objects[o].label for o in gone}:
        g = [o for o in gone if objects[o].label == label]
        c = [o for o in came if objects[o].label == label]
        if len(g) == 1 and len(c) == 1:
            b, a = previous.seen[g[0]], current.seen[c[0]]
            change("moved", g[0],
                   f"Probably moved: was {describe_place(b)}, a {label} is now {describe_place(a)}.",
                   image(b.sighting), image(a.sighting))
            paired |= {g[0], c[0]}

    for oid in gone:
        if oid not in paired:
            b = previous.seen[oid]
            change("missing", oid, f"Not seen this round. Last round it was {describe_place(b)}.",
                   image(b.sighting))
    for oid in came:
        if oid not in paired:
            a = current.seen[oid]
            change("new", oid, f"Not here last round. Now {describe_place(a)}.",
                   after=image(a.sighting))
    for oid in current.seen:
        if oid in previous.seen and oid in objects:
            b, a = previous.seen[oid], current.seen[oid]
            if moved(b, a):
                change("moved", oid, f"Was {describe_place(b)}, now {describe_place(a)}.",
                       image(b.sighting), image(a.sighting))

    order = {"missing": 0, "moved": 1, "new": 2}
    return sorted(changes, key=lambda c: order[c["kind"]])


class Rounds:
    """The running round, and every finished one, stored in `folder`/memory.db."""

    def __init__(self, folder):
        self.path = Path(folder) / "memory.db"
        self.running: Round | None = None
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY, name TEXT,
                    started REAL, ended REAL);
                CREATE TABLE IF NOT EXISTS round_seen (round_id INTEGER, object_id INTEGER,
                    sighting INTEGER, "where" TEXT, nearby TEXT,
                    PRIMARY KEY (round_id, object_id));
            """)

    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def start(self, name=""):
        with self.db() as db:
            last = db.execute("SELECT MAX(id) FROM rounds").fetchone()[0] or 0
        n = last + 1
        self.running = Round(n, name or f"Round {n}", time.time())
        return self.running

    def see(self, obj):
        """Note that `obj` is in view now, at its latest sighting."""
        if self.running and obj.sightings:
            s = obj.sightings[-1]
            self.running.seen[obj.id] = Seen(s.id, s.where, list(s.nearby))

    def end(self):
        """Finish the running round, store it, and return (round, previous round or None)."""
        r, self.running = self.running, None
        r.ended = time.time()
        previous = self.last()
        with self.db() as db:
            db.execute("INSERT INTO rounds VALUES (?, ?, ?, ?)", (r.id, r.name, r.started, r.ended))
            db.executemany('INSERT INTO round_seen VALUES (?, ?, ?, ?, ?)',
                           [(r.id, oid, s.sighting, s.where, json.dumps(s.nearby))
                            for oid, s in r.seen.items()])
        return r, previous

    def last(self):
        """The most recent finished round, or None."""
        with self.db() as db:
            row = db.execute("SELECT * FROM rounds ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            r = Round(*row)
            for oid, sid, where, nearby in db.execute(
                    'SELECT object_id, sighting, "where", nearby FROM round_seen WHERE round_id = ?',
                    (r.id,)):
                r.seen[oid] = Seen(sid, where, json.loads(nearby))
        return r
