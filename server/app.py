"""The live server: phone frames in, engine in the middle, dashboard out.

    /ws/camera      the phone page sends binary JPEG frames
    /ws/dashboard   messages exactly as in docs/dashboard-brief.md section 4
    /thumb/<object id>.jpg, /snapshot/<sighting id>.jpg   images from the memory store
    /, /phone       the built dashboard and camera page (dashboard/dist)

    python -m server.app --store store/live          # http://localhost:8000
    python -m server.app --video videos/x.mp4 --loop  # a recorded video instead of the phone

Only the newest camera frame is ever processed. If the engine is busy when
frames arrive, the older ones are dropped, so the delay stays at one frame
instead of growing into a queue.
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from engine.ask import embed_text
from engine.engine import Engine
from engine.video import FPS, frames

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dashboard" / "dist"
SAVE_EVERY = 10.0        # seconds between memory saves, so a crash loses at most this much
ACK_EVERY = 1.0          # seconds between camera_ack messages to the phone
VIEW_UPDATE_EVERY = 1.0  # seconds; most often a memory message is sent only because the view changed
REPORT_EVERY = 30.0



def tailscale_name():
    """This machine's MagicDNS name (e.g. laptop.tailnet.ts.net), or None."""
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                             text=True, timeout=5).stdout
        return json.loads(out)["Self"]["DNSName"].rstrip(".") or None
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


# Host names the server answers to. A page whose own domain is re-pointed at
# 127.0.0.1 (DNS rebinding) sends its own name as Host, and is refused. Only
# this machine's own Tailscale name is accepted, not every *.ts.net name;
# more can be added as GM_ALLOW_HOSTS=name1,name2.
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "::1", tailscale_name(),
                 *os.environ.get("GM_ALLOW_HOSTS", "").split(",")]
ALLOWED_HOSTS = [h for h in ALLOWED_HOSTS if h]      # seconds between latency lines in the server log


class Live:
    """Everything the server shares between connections."""

    def __init__(self, store):
        self.engine = Engine(store)
        # The engine, its tracker and the GPU are used by one thread at a time:
        # frames, questions, renames and saves all take this lock.
        self.lock = threading.Lock()
        self.latest = None               # (jpeg bytes, arrival time) of the newest frame
        self.new_frame = asyncio.Event()
        self.dashboards = set()
        self.seq = 0
        self.in_view = set()
        self.processed = deque(maxlen=50)   # finish times, for frames per second
        self.latencies = deque(maxlen=600)  # ms from a frame's arrival to its message being ready
        self.round = None
        self.rounds_started = 0


live: Live = None  # created at startup, after the store path is known
STORE = "store/live"
VIDEO = None             # a recorded video to play instead of the phone
LOOP = False


# --- messages ------------------------------------------------------------------

def fps():
    t = live.processed
    return round((len(t) - 1) / (t[-1] - t[0]), 1) if len(t) > 1 and t[-1] > t[0] else 0.0


def memory_message():
    return {"type": "memory", "objects": [
        {"object_id": o.id, "name": o.name, "label": o.label, "last_seen": o.last_seen,
         "where": o.where, "thumb": f"/thumb/{o.id}.jpg", "in_view": o.id in live.in_view}
        for o in live.engine.memory.objects.values()]}


async def broadcast(msg):
    data = json.dumps(msg)
    for ws in list(live.dashboards):
        try:
            await ws.send_text(data)
        except Exception:
            live.dashboards.discard(ws)


def run_frame(jpeg, arrived):
    """Decode and process one frame (in a worker thread). Returns the frame
    message, the ids of objects that changed, and the ids now in view."""
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, set(), set()
    with live.lock:
        detections, changed = live.engine.process(frame, time.time())
        stats = live.engine.stats(len(detections))
    live.processed.append(time.perf_counter())
    latency = (time.perf_counter() - arrived) * 1000
    live.latencies.append(latency)
    live.seq += 1
    h, w = frame.shape[:2]
    # The engine's own "fps" is how fast it could go; the dashboard shows how
    # many frames it actually processed, which is what the viewer sees.
    stats.update(fps=fps(), latency_ms=round(latency))
    msg = {"type": "frame", "seq": live.seq, "ts": time.time(), "width": w, "height": h,
           # the phone's own JPEG is sent back: boxes are in its pixels, and
           # re-encoding would cost time for no gain
           "jpeg_b64": base64.b64encode(jpeg).decode(),
           "detections": detections, "stats": stats}
    in_view = {d["object_id"] for d in detections if d["object_id"] is not None}
    return msg, changed, in_view


# --- background tasks ------------------------------------------------------------

async def process_loop():
    last_memory = 0.0
    while True:
        await live.new_frame.wait()
        live.new_frame.clear()
        jpeg, arrived = live.latest
        msg, changed, in_view = await asyncio.to_thread(run_frame, jpeg, arrived)
        if msg is None:
            continue
        await broadcast(msg)
        # New, renamed or moved objects go out at once. A change in what is in
        # view happens nearly every frame, so that alone sends at most once a
        # second: the whole memory is resent each time.
        view_changed = in_view != live.in_view
        live.in_view = in_view
        if changed or (view_changed and time.monotonic() - last_memory >= VIEW_UPDATE_EVERY):
            last_memory = time.monotonic()
            await broadcast(memory_message())


async def video_loop(path, loop):
    """Play a recorded video into the same path as the phone's frames, at real
    speed and the phone's 10 fps, so the dashboard cannot tell the difference.
    Looping replays the same objects, which then come back re-identified."""
    while True:
        it = frames(path)
        next_at = time.monotonic()
        while (item := await asyncio.to_thread(next, it, None)) is not None:
            ok, jpeg = cv2.imencode(".jpg", item[1], [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                live.latest = (jpeg.tobytes(), time.perf_counter())
                live.new_frame.set()
            next_at += 1 / FPS
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
        if not loop:
            print(f"[live] end of {path}", flush=True)
            return


async def save_loop():
    while True:
        await asyncio.sleep(SAVE_EVERY)
        await asyncio.to_thread(save)


async def report_loop():
    while True:
        await asyncio.sleep(REPORT_EVERY)
        lat = sorted(live.latencies)
        if lat:
            print(f"[live] {fps()} fps, server latency p50 {statistics.median(lat):.0f} ms, "
                  f"p95 {lat[int(len(lat) * 0.95)]:.0f} ms over {len(lat)} frames", flush=True)


def save():
    with live.lock:
        live.engine.memory.save()


@asynccontextmanager
async def lifespan(_app):
    global live
    live = Live(STORE)
    # The text model loads on the first question; loading it here keeps the
    # first real question from freezing the video. (ask() skips the model when
    # memory is empty, so the embedder is called directly.)
    await asyncio.to_thread(embed_text, ["warm up"])
    tasks = [asyncio.create_task(f()) for f in (process_loop, save_loop, report_loop)]
    if VIDEO:
        tasks.append(asyncio.create_task(video_loop(VIDEO, LOOP)))
    yield
    for t in tasks:
        t.cancel()
    save()


app = FastAPI(lifespan=lifespan)
# Added here, not in __main__, so it is on however the app is started.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# --- websockets ------------------------------------------------------------------

def same_origin(ws: WebSocket):
    """Browsers send any page's websocket to any host, so without this check a
    web page open on the laptop could watch the camera or forget everything
    through localhost. A browser always sends Origin; it must be this server.
    Clients with no Origin are not browsers, so no other page can drive them.
    Host itself is checked by TrustedHostMiddleware (see ALLOWED_HOSTS)."""
    origin = ws.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == ws.headers.get("host")


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    if not same_origin(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    last_ack = time.monotonic()
    try:
        while True:
            live.latest = (await ws.receive_bytes(), time.perf_counter())
            live.new_frame.set()
            if time.monotonic() - last_ack >= ACK_EVERY:
                last_ack = time.monotonic()
                await ws.send_text(json.dumps({"type": "camera_ack", "fps": fps()}))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    if not same_origin(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    live.dashboards.add(ws)
    await ws.send_text(json.dumps(memory_message()))
    if live.round:
        await ws.send_text(json.dumps(live.round))
    try:
        while True:
            await handle(ws, json.loads(await ws.receive_text()))
    except WebSocketDisconnect:
        live.dashboards.discard(ws)


async def handle(ws, msg):
    kind = msg.get("type")
    if kind == "ask":
        answer = await asyncio.to_thread(locked, live.engine.ask, msg.get("query", ""))
        await ws.send_text(json.dumps(answer))
    elif kind == "ask_photo":
        image = cv2.imdecode(np.frombuffer(base64.b64decode(msg["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            await ws.send_text(json.dumps({"type": "answer", "query": "photo", "found": False,
                                           "object_id": None, "text": "That photo could not be read.",
                                           "snapshot": None, "box": None, "nearby": [],
                                           "sightings": [], "candidates": []}))
            return
        answer = await asyncio.to_thread(locked, live.engine.ask_photo, image)
        await ws.send_text(json.dumps(answer))
    elif kind == "rename":
        oid, name = msg.get("object_id"), (msg.get("name") or "").strip()
        if oid in live.engine.memory.objects and name:
            await asyncio.to_thread(locked, live.engine.memory.rename, oid, name)
            await broadcast(memory_message())
    elif kind == "forget_all":
        await asyncio.to_thread(locked, forget_all)
        live.in_view = set()
        await broadcast(memory_message())
    elif kind == "round_start":
        # Rounds and change detection are phase 3: until then a round only has
        # a name and a clock, and ends with an empty report.
        live.rounds_started += 1
        n = live.rounds_started
        live.round = {"type": "round", "state": "running", "round_id": n,
                      "name": msg.get("name") or f"Round {n}", "started": time.time()}
        await broadcast(live.round)
    elif kind == "round_end" and live.round:
        ended, live.round = live.round, None
        await broadcast({"type": "round", "state": "ended", "round_id": ended["round_id"], "report": []})


def locked(fn, *args):
    with live.lock:
        return fn(*args)


def forget_all():
    live.engine.memory.forget_all()
    # The engine's open tracks still point at the deleted object ids; dropping
    # them makes every object in view be identified again from scratch.
    # Proposed to Sujan as an Engine.forget_all(), so the server would not
    # reach into engine internals.
    live.engine.tracks.clear()


# --- images and pages --------------------------------------------------------------

def image(path):
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/thumb/{object_id}.jpg")
def thumb(object_id: int):
    return image(live.engine.memory.thumb_path(object_id))


@app.get("/snapshot/{sighting_id}.jpg")
def snapshot(sighting_id: int):
    return image(live.engine.memory.snapshot_path(sighting_id))


@app.get("/keyframe/{name}")
def keyframe(name: str):
    raise HTTPException(404, "keyframes arrive with rounds, in phase 3")


@app.get("/phone")
def phone():
    return FileResponse(DIST / "phone" / "index.html")


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=STORE)
    # localhost only: `tailscale serve` is what makes it reachable, over HTTPS
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--video", help="play this recorded video instead of waiting for the phone")
    ap.add_argument("--loop", action="store_true", help="replay the video when it ends")
    args = ap.parse_args()
    if args.video and not Path(args.video).is_file():
        ap.error(f"no such video: {args.video}")
    STORE, VIDEO, LOOP = args.store, args.video, args.loop
    uvicorn.run(app, host=args.host, port=args.port)
