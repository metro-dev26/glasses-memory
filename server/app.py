"""The live server: phone frames in, engine in the middle, dashboard out.

    /ws/camera      the phone page sends binary JPEG frames
    /ws/dashboard   messages exactly as in docs/dashboard-brief.md section 4
    /thumb/<object id>.jpg, /snapshot/<sighting id>.jpg   images from the memory store
    /, /phone       the built dashboard and camera page (dashboard/dist)

    python -m server.app --store store/live          # http://localhost:8000
    python -m server.app --video videos/x.mp4 --loop  # a recorded video instead of the phone
    python -m server.app --record videos/desk1        # also save every processed frame, to replay later

Recording can also be switched on and off from the dashboard or the phone page:
each recording is a folder of frames (replayable with --video FOLDER) plus an
.mp4 of it beside the folder, made when the recording stops.

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
import traceback
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from engine.engine import Engine
from engine.rounds import Rounds, compare
from engine.video import FPS, frames

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dashboard" / "dist"
SAVE_EVERY = 10.0        # seconds between memory saves, so a crash loses at most this much
ACK_EVERY = 1.0          # seconds between camera_ack messages to the phone
VIEW_UPDATE_EVERY = 1.0  # seconds; most often a memory message is sent only because the view changed
REPORT_EVERY = 30.0       # seconds between latency lines in the server log
VOICE_URL = os.environ.get("GM_VOICE_URL", "http://127.0.0.1:8010")  # server/voice.py
SEND_TIMEOUT = 2.0        # seconds a socket may take to accept one message before it is dropped


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
ALLOWED_HOSTS = [h for h in ALLOWED_HOSTS if h]


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
        self.rounds = Rounds(store)
        self.round = None                # the "running" round message, for dashboards that join late
        self.cameras = set()
        self.recording = None            # folder the frames are being saved in, or None


live: Live = None  # created at startup, after the store path is known
STORE = "store/live"
VIDEO = None             # a recorded video to play instead of the phone
RECORD = None            # folder to record into from the start (--record)
VIDEOS = ROOT / "videos"  # where recordings started from the dashboard or phone go
LOOP = False


# --- messages ------------------------------------------------------------------

def fps():
    t = live.processed
    return round((len(t) - 1) / (t[-1] - t[0]), 1) if len(t) > 1 and t[-1] > t[0] else 0.0


def memory_message():
    return {"type": "memory", "objects": [
        {"object_id": o.id, "name": o.name, "label": o.label, "last_seen": o.last_seen,
         "where": o.where, "thumb": f"/thumb/{o.id}.jpg?v={o.thumb_version}", "in_view": o.id in live.in_view}
        for o in live.engine.memory.objects.values()]}


async def broadcast(msg, to=None):
    """Send to every socket at once. One that cannot take a message within
    SEND_TIMEOUT (a stalled phone or tab) is dropped, and its page reconnects
    by itself; waiting for it would freeze the video for everyone."""
    data = json.dumps(msg)
    sockets = live.dashboards if to is None else to

    async def send(ws):
        try:
            await asyncio.wait_for(ws.send_text(data), SEND_TIMEOUT)
        except Exception:
            sockets.discard(ws)
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(*(send(ws) for ws in list(sockets)))


def recording_message(video=None):
    return {"type": "recording", "on": live.recording is not None,
            "folder": str(live.recording) if live.recording else None, "video": video}


def run_frame(jpeg, arrived):
    """Decode and process one frame (in a worker thread). Returns the frame
    message, the ids of objects that changed, and the ids now in view."""
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, set(), set()
    ts = time.time()
    folder = live.recording
    if folder:
        # The JPEG exactly as the engine got it, so a replay sees the same frames.
        (folder / f"{ts:.3f}.jpg").write_bytes(jpeg)
    with live.lock:
        detections, changed = live.engine.process(frame, ts)
        stats = live.engine.stats(len(detections))
        for d in detections:
            if d["object_id"] is not None:
                live.rounds.see(live.engine.memory.objects[d["object_id"]])
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
        try:
            msg, changed, in_view = await asyncio.to_thread(run_frame, jpeg, arrived)
        except Exception:
            # One bad frame must not stop the video: say so and take the next.
            print("[live] frame failed:", flush=True)
            traceback.print_exc()
            continue
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


def make_mp4(folder):
    """Turn a folder of recorded frames into folder.mp4, at the rate they
    actually arrived. Returns the video path, or None if it failed."""
    shots = sorted(folder.glob("*.jpg"))
    if len(shots) < 2:
        return None
    first, last = float(shots[0].stem), float(shots[-1].stem)
    rate = max(1.0, (len(shots) - 1) / max(last - first, 1e-3))
    out = folder.with_suffix(".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{rate:.2f}",
           "-pattern_type", "glob", "-i", str(folder / "*.jpg"),
           # H.264 + yuv420p plays everywhere (phones, WhatsApp); it needs even sizes
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[live] could not make {out}: {e}", flush=True)
        return None
    return out


def start_recording(folder=None):
    folder = Path(folder) if folder else VIDEOS / datetime.now().strftime("rec-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    live.recording = folder
    print(f"[live] recording into {folder}", flush=True)


async def stop_recording():
    folder, live.recording = live.recording, None
    await broadcast(recording_message())
    await broadcast(recording_message(), to=live.cameras)
    video = await asyncio.to_thread(make_mp4, folder)
    print(f"[live] recording saved: {folder} and {video}", flush=True)
    done = recording_message(str(video.relative_to(ROOT)) if video and video.is_relative_to(ROOT) else
                             str(video) if video else None)
    await broadcast(done)
    await broadcast(done, to=live.cameras)


async def set_recording(on):
    if on and not live.recording:
        start_recording()
        await broadcast(recording_message())
        await broadcast(recording_message(), to=live.cameras)
    elif not on and live.recording:
        await stop_recording()


@asynccontextmanager
async def lifespan(_app):
    global live
    live = Live(STORE)
    if RECORD:
        start_recording(RECORD)
    # The text model loads on the first question; loading it here keeps the
    # first real question from freezing the video.
    await asyncio.to_thread(live.engine.warm_up)
    tasks = [asyncio.create_task(f()) for f in (process_loop, save_loop, report_loop)]
    if VIDEO:
        tasks.append(asyncio.create_task(video_loop(VIDEO, LOOP)))
    for t in tasks:
        t.add_done_callback(report_crash)
    yield
    for t in tasks:
        t.cancel()
    save()
    if live.recording:
        make_mp4(live.recording)


def report_crash(task):
    """A background task that raises dies silently; print why, so a frozen
    dashboard has a reason in the server log."""
    if not task.cancelled() and task.exception():
        e = task.exception()
        print(f"[live] {task.get_coro().__name__} stopped:", flush=True)
        traceback.print_exception(type(e), e, e.__traceback__)


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
    live.cameras.add(ws)
    await ws.send_text(json.dumps(recording_message()))
    last_ack = time.monotonic()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                live.latest = (msg["bytes"], time.perf_counter())
                live.new_frame.set()
            elif msg.get("text"):
                # The phone's record button; frames are binary, so text is a command.
                try:
                    cmd = json.loads(msg["text"])
                    if isinstance(cmd, dict) and cmd.get("type") in ("record_start", "record_stop"):
                        await set_recording(cmd["type"] == "record_start")
                except ValueError:
                    pass
            if time.monotonic() - last_ack >= ACK_EVERY:
                last_ack = time.monotonic()
                await ws.send_text(json.dumps({"type": "camera_ack", "fps": fps()}))
    except WebSocketDisconnect:
        pass
    finally:
        live.cameras.discard(ws)


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    if not same_origin(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    live.dashboards.add(ws)
    await ws.send_text(json.dumps(memory_message()))
    await ws.send_text(json.dumps(recording_message()))
    if live.round:
        await ws.send_text(json.dumps(live.round))
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
                if not isinstance(msg, dict):
                    raise ValueError("not a JSON object")
                await handle(ws, msg)
            except WebSocketDisconnect:
                raise
            except Exception:
                # A bad message is dropped, not allowed to close the socket.
                print(f"[live] dashboard message failed: {text[:200]!r}", flush=True)
                traceback.print_exc()
    except WebSocketDisconnect:
        pass
    finally:
        live.dashboards.discard(ws)


async def handle(ws, msg):
    kind = msg.get("type")
    if kind == "ask":
        answer = await asyncio.to_thread(asked, live.engine.ask, msg.get("query", ""))
        await ws.send_text(json.dumps(answer))
    elif kind == "ask_object":
        answer = await asyncio.to_thread(asked, live.engine.ask_object, msg.get("object_id"))
        await ws.send_text(json.dumps(answer))
    elif kind == "ask_photo":
        try:
            raw = base64.b64decode(msg.get("jpeg_b64") or "", validate=True)
        except (ValueError, TypeError):
            raw = b""
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if image is None:
            await ws.send_text(json.dumps({"type": "answer", "query": "photo", "found": False,
                                           "object_id": None, "text": "That photo could not be read.",
                                           "snapshot": None, "box": None, "nearby": [],
                                           "sightings": [], "candidates": []}))
            return
        answer = await asyncio.to_thread(asked, live.engine.ask_photo, image)
        await ws.send_text(json.dumps(answer))
    elif kind == "rename":
        oid, name = msg.get("object_id"), (msg.get("name") or "").strip()
        if oid in live.engine.memory.objects and name:
            await asyncio.to_thread(locked, live.engine.memory.rename, oid, name)
            await broadcast(memory_message())
    elif kind == "forget_all":
        await asyncio.to_thread(locked, live.engine.forget_all)
        live.in_view = set()
        with live.lock:
            live.rounds = Rounds(STORE)     # the rounds were in the store that was just wiped
        if live.round:
            ended, live.round = live.round, None
            await broadcast({"type": "round", "state": "ended", "round_id": ended["round_id"],
                             "report": [], "seen": 0, "first": True})
        await broadcast(memory_message())
    elif kind == "round_start" and not live.round:
        with live.lock:
            r = live.rounds.start(str(msg.get("name") or "").strip())
        live.round = {"type": "round", "state": "running", "round_id": r.id,
                      "name": r.name, "started": r.started}
        await broadcast(live.round)
    elif kind == "round_end" and live.round:
        live.round = None
        await broadcast(await asyncio.to_thread(locked, end_round))
    elif kind in ("record_start", "record_stop"):
        await set_recording(kind == "record_start")


def end_round():
    """Finish the round and compare it with the one before (holds the lock)."""
    r, previous = live.rounds.end()
    report = compare(previous, r, live.engine.memory.objects, snapshot_url) if previous else []
    return {"type": "round", "state": "ended", "round_id": r.id, "report": report,
            "seen": len(r.seen), "first": previous is None}


def snapshot_url(sighting_id):
    return f"/snapshot/{sighting_id}.jpg" if live.engine.memory.snapshot_path(sighting_id).exists() else ""


def locked(fn, *args):
    with live.lock:
        return fn(*args)


def asked(fn, *args):
    """Answer a question as of now. The engine's clock is the time of the last
    frame, which stands still while the phone is not sending: "ago" would then
    count from that frame instead of from the question."""
    with live.lock:
        live.engine.now = time.time()
        return fn(*args)


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


@app.post("/speak")
async def speak(request: Request):
    """An answer spoken by the Kokoro voice service (server/voice.py). 503 when
    it is not running, and the dashboard then uses the browser's own speech."""
    text = (await request.body())[:2000]
    try:
        wav = await asyncio.to_thread(voice, text)
    except OSError:
        raise HTTPException(503, "voice service not running")
    return Response(wav, media_type="audio/wav")


def voice(text):
    req = urllib.request.Request(f"{VOICE_URL}/speak", data=text, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


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
    ap.add_argument("--record", metavar="FOLDER",
                    help="save every processed frame here; replay with --video FOLDER or engine.run FOLDER")
    args = ap.parse_args()
    if args.video and not Path(args.video).exists():
        ap.error(f"no such video: {args.video}")
    STORE, VIDEO, LOOP, RECORD = args.store, args.video, args.loop, args.record
    uvicorn.run(app, host=args.host, port=args.port)
