# Brief: live dashboard and phone camera page

You are building the user interface for the live version of glasses-memory.
Another person is building the server (the ML engine) in parallel. You will
never see that server while you work, so you build against a mock server that
speaks the exact protocol in section 4. If the protocol and the real server
disagree later, the protocol in this file wins; ask Sujan before changing it.

## 1. What the product is

glasses-memory, live: a phone held at eye level plays the role of smart
glasses. It streams video to a GPU server. The server detects every object in
view, tracks it, recognises it again when it comes back, and remembers where
each object was last seen. You can ask "where are my keys?" at any moment.

It also compares walk-throughs ("rounds") of the same space: what is missing,
new, or moved since the last round. This mirrors an operator's inspection
round in a treatment plant.

The audience is technical people watching a live demo over a screen share,
and engineers reading the code. It has to look precise and
calm, like a real instrument. Not flashy, not a template.

## 2. What to build

Two pages, served as static files:

1. `/` — the **dashboard**, on a laptop browser. This is what gets screen-shared.
2. `/phone` — the **camera page**, on a phone browser. It only captures and sends.

Stack: Vite + React + TypeScript. The live overlay is drawn on a `<canvas>`
over the video frame, not with DOM boxes (dozens of boxes at 10 fps). No UI
kit; a small hand-written stylesheet. Dark theme only.

Put everything in `dashboard/` in the repo root. Do not touch the Python files.

## 3. The dashboard, region by region

Wide screen, designed for 1920x1080 and still usable at 1366x768.

### 3.1 Live view (left, about 65% width)
- The latest frame from the server, with the overlay drawn on top.
- Per object: a **thin box** (1.5 px), corner-accented rather than a heavy
  rectangle, and a **small label pill** above it: `remote · 0.87 · #12`
  (label, confidence, object id).
- Colour comes from the **object id**, not the label, so the same object keeps
  the same colour all session. Use a fixed palette that reads on video.
- Boxes ease to their new position between frames (short interpolation) so
  they glide instead of jumping.
- Low-confidence detections (< 0.4) are drawn dimmer and dashed.
- When an object is recognised as one seen earlier, its pill briefly shows
  `seen 3m ago`.
- HUD in a corner, small monospace text: server fps, end-to-end latency (ms),
  objects in view, objects in memory, connection state.

### 3.2 Memory panel (right column, top)
- One row per remembered object: thumbnail, name, label, `last seen 2m ago`,
  and where ("desk, next to laptop").
- Newly created objects slide in at the top; objects in view right now are
  marked live.
- Click a name to rename it ("cup" -> "my blue mug"); sends `rename`.
- Search filter over names.

### 3.3 Ask bar (right column, middle)
- Text input plus a microphone button. Voice uses the browser Web Speech API
  (`SpeechRecognition`); hide the mic where the browser lacks it.
- Sends `ask`. The answer card shows: the snapshot (with the object's box or
  mask), the sentence ("Last seen 4 minutes ago on the desk, next to the
  laptop"), nearby objects as chips, and a small timeline of past sightings.
- The answer sentence is also spoken with `speechSynthesis` (toggle to mute).
- "Ask by photo": upload an image; sends `ask_photo`.
- No match: show the server's message and the closest known objects.

### 3.4 Rounds (right column, bottom, or a bottom strip)
- Start round / End round buttons, current round name and elapsed time.
- During a round, **change alerts** appear as they arrive: missing / new /
  moved, each with before and after thumbnails side by side and the reason
  text. Alerts are colour-coded by kind and stack, newest first.
- After End round: a round report listing every change, with counts.

### 3.5 Always present
- A "Forget everything" button (with a confirm step inside the page, not a
  browser `confirm()` dialog) that sends `forget_all`.
- A clear disconnected state with automatic reconnect.

## 4. Protocol (the contract)

All JSON. Times are Unix seconds (float). Boxes are `[x1, y1, x2, y2]` in the
pixel coordinates of the frame they arrived with.

### 4.1 Dashboard WebSocket: `ws(s)://<host>/ws/dashboard`

Server -> dashboard:

```json
{"type": "frame", "seq": 1042, "ts": 1760000000.12, "width": 640, "height": 1138,
 "jpeg_b64": "<base64 JPEG>",
 "detections": [
   {"track_id": 7, "object_id": 12, "label": "remote", "name": "remote",
    "conf": 0.87, "box": [120, 340, 260, 610], "reidentified": false}
 ],
 "stats": {"fps": 11.8, "latency_ms": 180, "in_view": 9, "in_memory": 27}}
```
`object_id` is `null` while a new track is still being identified.
`reidentified: true` on the first frame an object is matched to an earlier
sighting; the payload then also has `"last_seen_ago_s": 184`.

```json
{"type": "memory", "objects": [
  {"object_id": 12, "name": "remote", "label": "remote", "last_seen": 1759999820.5,
   "where": "desk, next to laptop", "thumb": "/thumb/12.jpg", "in_view": true}
]}
```
Sent whole on connect, then whenever an object is created, renamed, or moves.

```json
{"type": "answer", "query": "where's my remote?", "found": true, "object_id": 12,
 "text": "Last seen 4 minutes ago on the desk, next to the laptop.",
 "snapshot": "/snapshot/88.jpg", "box": [120, 340, 260, 610],
 "nearby": ["laptop", "mug"],
 "sightings": [{"start": 1759999700.0, "end": 1759999820.5, "where": "desk"}],
 "candidates": []}
```
When `found` is false, `text` says so and `candidates` lists the closest names.

```json
{"type": "round", "state": "running", "round_id": 3, "name": "Round 3", "started": 1760000000.0}
{"type": "round", "state": "ended", "round_id": 3, "report": [ <change objects> ]}
```

```json
{"type": "change", "round_id": 3, "kind": "missing", "object_id": 31, "name": "multimeter",
 "text": "Last round it was on the shelf, left of the toolbox.",
 "before": "/keyframe/204.jpg", "after": "/keyframe/377.jpg"}
```
`kind` is one of `missing`, `new`, `moved`.

Dashboard -> server:

```json
{"type": "ask", "query": "where's my remote?"}
{"type": "ask_photo", "jpeg_b64": "<base64 JPEG>"}
{"type": "rename", "object_id": 12, "name": "tv remote"}
{"type": "round_start", "name": "Morning round"}
{"type": "round_end"}
{"type": "forget_all"}
```

Images (`thumb`, `snapshot`, `keyframe` paths) are fetched over plain HTTP
from the same host.

### 4.2 Camera WebSocket: `ws(s)://<host>/ws/camera`

The phone page sends **binary** messages, each one a JPEG frame, about 10 per
second, resized so the long side is 960 px, quality 0.7. Nothing else.
The server may send `{"type": "camera_ack", "fps": 11.8}` back; show it.

## 5. The phone page `/phone`

- Big Start/Stop button, rear camera (`facingMode: "environment"`), portrait.
- Shows its own preview, the sending rate, and connection state.
- Keeps the screen awake while streaming (Wake Lock API where available).
- Camera access needs HTTPS; the real setup provides it via Tailscale. Do not
  build any HTTP fallback hacks.

## 6. Mock server (build this first)

Write `dashboard/mock/mock_server.py` (FastAPI + uvicorn) that:
- serves the built dashboard and phone page,
- on `/ws/dashboard`, replays a scripted scene at 10 fps: a test image or a
  short loop of frames you generate (no personal footage), with 5-10 fake
  objects moving smoothly, one leaving and coming back as `reidentified`,
- answers `ask` with a canned answer for known names and `found: false` otherwise,
- emits a few `change` messages during a round,
- accepts `/ws/camera` frames and echoes the latest one back as the `frame`
  image so the phone page can be tested end to end.

The UI is done when every region in section 3 works against the mock.

## 7. Rules (from docs/HANDBOOK.md)

- Work on a branch (`feat/dashboard`), open a pull request; Sujan reviews before merge.
- Conventional commits (`feat:`, `fix:`, `docs:`), body explains why. No co-author trailers.
- No personal footage in the repo, ever.
- Keep it small and readable. Both of us must be able to explain every line
  in an interview; if a piece can't be explained, it isn't done.
- No comments that narrate how the code was written; comments explain the code.
