"""Mock of the engine server, so the dashboard can be built without a GPU.

Speaks the protocol in docs/dashboard-brief.md section 4 exactly:
- /ws/dashboard replays a drawn scene at 10 fps: eight fake objects drifting
  on a desk; the keys leave the frame and come back as `reidentified`
- `ask` gets a canned answer for known names, `found: false` otherwise
- a running round emits missing / new / moved changes
- /ws/camera takes the phone's JPEG frames; while they arrive, the latest
  one replaces the drawn scene, so the phone page can be tested end to end

    cd dashboard && npm run build && npm run mock      # http://localhost:8000
"""
import asyncio
import base64
import io
import json
import math
import random
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

DIST = Path(__file__).parent.parent / "dist"
W, H = 640, 1138             # a portrait phone frame, as in the protocol example
FPS = 10
LOOP = 60.0                  # seconds; the scene repeats
AWAY = (18.0, 32.0)          # the keys are out of view in this part of the loop
FONT = ImageFont.load_default(size=22)
SMALL = ImageFont.load_default(size=16)

# object_id, label, colour, (w, h), centre, drift radius, drift speed
OBJECTS = [
    (1, "laptop", (70, 74, 84), (300, 190), (330, 380), 6, 0.10),
    (2, "mug", (176, 64, 52), (90, 110), (520, 620), 10, 0.21),
    (3, "remote", (40, 40, 44), (70, 210), (140, 640), 14, 0.17),
    (4, "notebook", (214, 196, 140), (190, 240), (300, 760), 8, 0.12),
    (5, "keys", (196, 160, 60), (90, 60), (480, 860), 12, 0.30),
    (6, "pen", (40, 80, 170), (24, 150), (110, 900), 16, 0.25),
    (7, "sunglasses", (30, 30, 30), (150, 60), (380, 980), 9, 0.19),
    (8, "bottle", (90, 150, 190), (80, 220), (560, 300), 7, 0.14),
]
WHERE = {1: "desk", 2: "desk, next to laptop", 3: "desk, next to notebook",
         4: "desk, next to remote", 5: "desk, next to mug", 6: "desk, next to notebook",
         7: "desk, next to keys", 8: "desk, next to laptop"}

app = FastAPI()
dashboards: set[WebSocket] = set()
state = {
    "names": {oid: label for oid, label, *_ in OBJECTS},
    "last_seen": {oid: time.time() - random.uniform(30, 300) for oid, *_ in OBJECTS},
    "round": None,           # {"round_id", "name", "started", "changes"}
    "rounds_done": 0,
    "camera_jpeg": None,
    "camera_at": 0.0,
    "camera_times": [],
}


# --- the drawn scene -----------------------------------------------------------

def position(obj, t):
    _, _, _, (w, h), (cx, cy), r, speed = obj
    a = t * speed * 2 * math.pi
    x, y = cx + r * math.sin(a), cy + r * math.cos(a * 0.7)
    return [round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2)]


def keys_away(t):
    return AWAY[0] <= t % LOOP < AWAY[1]


def draw_scene(t, objects=OBJECTS, hide=()):
    img = Image.new("RGB", (W, H), (58, 50, 44))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 180], fill=(150, 156, 160))                 # wall
    d.rectangle([0, 180, W, H], fill=(120, 86, 58))                   # desk
    for y in range(200, H, 46):                                       # wood grain
        d.line([0, y + 6 * math.sin(y), W, y], fill=(108, 76, 50), width=2)
    for obj in objects:
        oid, label, colour = obj[:3]
        if oid in hide or (oid == 5 and keys_away(t)):
            continue
        x1, y1, x2, y2 = position(obj, t)
        d.rounded_rectangle([x1 + 6, y1 + 8, x2 + 6, y2 + 8], 14, fill=(80, 56, 38))  # shadow
        d.rounded_rectangle([x1, y1, x2, y2], 14, fill=colour)
        d.text((x1 + 8, y1 + 6), label, font=SMALL, fill=(235, 235, 235))
    return img


def jpeg(img, quality=70):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# --- protocol messages ----------------------------------------------------------

class Tracks:
    """Fake ByteTrack: a new track id each time an object enters the view, and
    object_id null for its first five frames, as while the engine identifies it."""

    def __init__(self):
        self.next_id = 1
        self.live = {}           # object_id -> (track_id, frames seen)

    def step(self, visible):
        for oid in list(self.live):
            if oid not in visible:
                del self.live[oid]
        out = {}
        for oid in visible:
            tid, age = self.live.get(oid, (None, 0))
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.live[oid] = (tid, age + 1)
            out[oid] = (tid, age + 1)
        return out


def frame_message(seq, t, tracks):
    now = time.time()
    use_camera = state["camera_jpeg"] is not None and now - state["camera_at"] < 2
    if use_camera:
        img = Image.open(io.BytesIO(state["camera_jpeg"]))
        width, height = img.size
        data = state["camera_jpeg"]
    else:
        width, height = W, H
        data = jpeg(draw_scene(t))

    visible = [o for o in OBJECTS if not (o[0] == 5 and keys_away(t))]
    ages = tracks.step({o[0] for o in visible})
    detections = []
    for obj in visible:
        oid, label = obj[0], obj[1]
        tid, age = ages[oid]
        box = position(obj, t)
        if use_camera:           # keep the fake boxes inside the phone's frame
            box = [round(v * width / W) if i % 2 == 0 else round(v * height / H)
                   for i, v in enumerate(box)]
        conf = round(0.32 + 0.6 * (0.5 + 0.5 * math.sin(oid * 1.7 + t * 0.2)), 2)
        det = {"track_id": tid, "object_id": oid if age > 5 else None, "label": label,
               "name": state["names"].get(oid, label), "conf": conf, "box": box,
               "reidentified": False}
        if age == 6 and oid == 5 and t % LOOP >= AWAY[1]:   # the keys, back after leaving
            det["reidentified"] = True
            det["last_seen_ago_s"] = round(now - state["last_seen"][oid])
        if det["object_id"] is not None:
            state["last_seen"][oid] = now
        detections.append(det)

    return {"type": "frame", "seq": seq, "ts": now, "width": width, "height": height,
            "jpeg_b64": base64.b64encode(data).decode(),
            "detections": detections,
            "stats": {"fps": round(random.uniform(9.6, 10.4), 1),
                      "latency_ms": random.randint(150, 230),
                      "in_view": len(detections), "in_memory": len(state["names"])}}


def memory_message(t):
    in_view = {o[0] for o in OBJECTS if not (o[0] == 5 and keys_away(t))}
    return {"type": "memory", "objects": [
        {"object_id": oid, "name": name, "label": OBJECTS[oid - 1][1],
         "last_seen": state["last_seen"][oid], "where": WHERE[oid],
         "thumb": f"/thumb/{oid}.jpg", "in_view": oid in in_view}
        for oid, name in state["names"].items()]}


def answer(query):
    words = set(query.lower().replace("?", "").replace("'s", "").split()) - STOP
    for oid, name in state["names"].items():
        if words & set(name.lower().split()) or words & {OBJECTS[oid - 1][1]}:
            others = [state["names"][o] for o in (oid % 8 + 1, (oid + 1) % 8 + 1) if o in state["names"]]
            ago = time.time() - state["last_seen"][oid]
            mins = max(1, round(ago / 60))
            return {"type": "answer", "query": query, "found": True, "object_id": oid,
                    "text": f"Last seen {mins} minute{'s' * (mins != 1)} ago on the {WHERE[oid].replace(', ', ', ')}.",
                    "snapshot": f"/snapshot/{oid}.jpg", "box": position(OBJECTS[oid - 1], 0),
                    "nearby": others,
                    "sightings": [{"start": time.time() - 900, "end": time.time() - 840, "where": "shelf"},
                                  {"start": time.time() - 400, "end": time.time() - 380, "where": "bed"},
                                  {"start": time.time() - ago - 60, "end": time.time() - ago, "where": "desk"}],
                    "candidates": []}
    return {"type": "answer", "query": query, "found": False, "object_id": None,
            "text": "I don't remember seeing that.", "snapshot": None, "box": None,
            "nearby": [], "sightings": [], "candidates": list(state["names"].values())[:3]}


STOP = {"where", "is", "are", "my", "the", "a", "did", "i", "leave", "put", "last", "see"}

CHANGES = [
    (3.0, {"kind": "missing", "object_id": 7, "name": "sunglasses",
           "text": "Last round they were on the desk, next to the keys."}),
    (6.0, {"kind": "new", "object_id": 9, "name": "multimeter",
           "text": "Not here last round: on the desk, right of the notebook."}),
    (9.0, {"kind": "moved", "object_id": 2, "name": "mug",
           "text": "Moved from next to the laptop to the edge of the desk."}),
]


# --- websockets -----------------------------------------------------------------

async def broadcast(msg):
    data = json.dumps(msg)
    for ws in list(dashboards):
        try:
            await ws.send_text(data)
        except Exception:
            dashboards.discard(ws)


async def scene_loop():
    start, seq, tracks = time.time(), 0, Tracks()
    away_before = False
    while True:
        t = time.time() - start
        if dashboards:
            seq += 1
            await broadcast(frame_message(seq, t, tracks))
            if keys_away(t) != away_before:             # keys left or came back
                away_before = keys_away(t)
                await broadcast(memory_message(t))
            r = state["round"]
            if r:
                for at, change in CHANGES:
                    if change not in r["changes"] and time.time() - r["started"] >= at:
                        r["changes"].append(change)
                        await broadcast(change_message(r, change))
        await asyncio.sleep(1 / FPS)


def change_message(r, change):
    oid = change["object_id"]
    return {"type": "change", "round_id": r["round_id"], **change,
            "before": f"/keyframe/{oid}-before.jpg", "after": f"/keyframe/{oid}-after.jpg"}


@app.on_event("startup")
async def start():
    asyncio.create_task(scene_loop())


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    dashboards.add(ws)
    await ws.send_text(json.dumps(memory_message(0)))
    r = state["round"]
    if r:
        await ws.send_text(json.dumps({"type": "round", "state": "running", "round_id": r["round_id"],
                                       "name": r["name"], "started": r["started"]}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "ask":
                await asyncio.sleep(0.4)
                await ws.send_text(json.dumps(answer(msg["query"])))
            elif kind == "ask_photo":
                await asyncio.sleep(0.6)
                await ws.send_text(json.dumps({**answer("remote"), "query": "photo"}))
            elif kind == "rename" and msg["object_id"] in state["names"]:
                state["names"][msg["object_id"]] = msg["name"]
                await broadcast(memory_message(0))
            elif kind == "round_start":
                state["rounds_done"] += 1
                n = state["rounds_done"]
                state["round"] = {"round_id": n, "name": msg.get("name") or f"Round {n}",
                                  "started": time.time(), "changes": []}
                r = state["round"]
                await broadcast({"type": "round", "state": "running", "round_id": n,
                                 "name": r["name"], "started": r["started"]})
            elif kind == "round_end" and state["round"]:
                r, state["round"] = state["round"], None
                await broadcast({"type": "round", "state": "ended", "round_id": r["round_id"],
                                 "report": [change_message(r, c) for c in r["changes"]]})
            elif kind == "forget_all":
                state["names"] = {}
                await broadcast(memory_message(0))
                await asyncio.sleep(3)                    # then the scene is "seen" again
                state["names"] = {oid: label for oid, label, *_ in OBJECTS}
                await broadcast(memory_message(0))
    except WebSocketDisconnect:
        dashboards.discard(ws)


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state["camera_jpeg"] = await ws.receive_bytes()
            now = time.time()
            state["camera_at"] = now
            times = [x for x in state["camera_times"] if now - x < 2] + [now]
            state["camera_times"] = times
            if len(times) % 10 == 0:
                await ws.send_text(json.dumps({"type": "camera_ack", "fps": round(len(times) / 2, 1)}))
    except WebSocketDisconnect:
        pass


# --- images ---------------------------------------------------------------------

def image_response(img):
    return Response(jpeg(img, 85), media_type="image/jpeg")


@app.get("/thumb/{oid}.jpg")
def thumb(oid: int):
    obj = OBJECTS[(oid - 1) % len(OBJECTS)]
    x1, y1, x2, y2 = position(obj, 0)
    pad = 30
    return image_response(draw_scene(0).crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad)).resize((128, 128)))


@app.get("/snapshot/{oid}.jpg")
def snapshot(oid: int):
    return image_response(draw_scene(0))     # full size: the answer box is in frame pixels


@app.get("/keyframe/{name}.jpg")
def keyframe(name: str):
    oid, when = name.split("-")
    oid = int(oid)
    extra = [(9, "multimeter", (200, 60, 40), (120, 160), (470, 720), 0, 0)]
    if when == "before":
        img = draw_scene(0, hide=(9,))
    elif oid == 7:
        img = draw_scene(0, hide=(7,))
    elif oid == 9:
        img = draw_scene(0, OBJECTS + extra)
    else:
        moved = [o if o[0] != 2 else (*o[:4], (580, 1040), 0, 0) for o in OBJECTS]
        img = draw_scene(0, moved)
    return image_response(img.resize((W // 3, H // 3)))


@app.get("/phone")
def phone_redirect():
    return FileResponse(DIST / "phone" / "index.html")


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
