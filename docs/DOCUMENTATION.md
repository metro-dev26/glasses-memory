# glasses-memory: full documentation

Everything about this project in one place: the problem, both versions, how every
part of the live system works and why, the message protocol, what we measured,
what is still broken, and how to run and test it.

Other docs go deeper on one thing each:
- [`HANDBOOK.md`](HANDBOOK.md): version 1 (the offline SAM 2 pipeline), file by file, and our rules.
- [`design/live-glasses-memory.md`](design/live-glasses-memory.md): the original design for version 2.
- [`dashboard-brief.md`](dashboard-brief.md): the UI spec and the full protocol.
- [`../server/README.md`](../server/README.md): running the server, Tailscale, latency.

Line numbers in this file are for branch `feat/safety-rounds` at the time of writing.

---

## Contents

1. [The problem and the idea](#1-the-problem-and-the-idea)
2. [Two versions at a glance](#2-two-versions-at-a-glance)
3. [Version 2: the whole system](#3-version-2-the-whole-system)
4. [The engine, file by file](#4-the-engine-file-by-file)
5. [The server](#5-the-server)
6. [The phone page and the dashboard](#6-the-phone-page-and-the-dashboard)
7. [The protocol](#7-the-protocol)
8. [Version 1, in short](#8-version-1-in-short)
9. [Measuring: tools and every result so far](#9-measuring-tools-and-every-result-so-far)
10. [Known problems](#10-known-problems)
11. [What's next](#11-whats-next)
12. [Setup, running, testing](#12-setup-running-testing)
13. [Every file in the repo](#13-every-file-in-the-repo)
14. [Rules we follow](#14-rules-we-follow)
15. [Glossary](#15-glossary)

---

## 1. The problem and the idea

People lose everyday things: keys, earphones, a charger, a stapler. You had it a
few minutes ago, but you can't remember where you put it down.

Smart glasses see everything you see. So they could remember it for you. This
project builds that memory: a camera at eye level sees a room, the system
remembers each object and where it was put down, and later you ask "where's my
stapler?" and get the last place it was seen, with a photo of that moment.

Two ideas run through everything:

- **"Last known", not "current".** The memory knows where an object was when the
  camera last saw it. If someone moved it while the camera was elsewhere, the
  memory can't know. Every answer is phrased that way ("Last seen 4 minutes ago,
  on the desk").
- **Memory, not recording.** No video is kept. Per object we keep a small
  thumbnail, a few fingerprints, and one snapshot of the moment it was left
  somewhere. Forgetting everything means deleting one folder.

A second use came later: **safety rounds**. Someone walks the same space twice
(for example an inspection walk in a plant), and the system reports what is
missing, what is new, and what has moved since the last walk.

We don't have glasses, so a phone held at eye level stands in for them.

---

## 2. Two versions at a glance

| | Version 1 (offline) | Version 2 (live) |
|---|---|---|
| Input | a recorded video file | the phone's camera, live (or a recorded video replayed live) |
| Setup per object | draw a box around it once (`register.py`) | none: every object in view is found automatically |
| Finding objects | SAM 2 follows your box through the video | YOLOE detects ~66 kinds of everyday things |
| Same object across frames | SAM 2's video memory | ByteTrack (while in view) + DINOv3 fingerprints (after it leaves and returns) |
| Speed | ~5 min per minute of video (RTX 3050) | ~17 to 35 frames a second, the phone sends 10 |
| Answer | a static HTML page | a live dashboard: text, voice or photo questions |
| Accuracy | 3/5 on the untuned test video | see section 9: re-identification is the weak point |
| Code | `register.py`, `track.py`, `remember.py`, `ask_page.py`, `run.py` | `engine/`, `server/`, `dashboard/` |

Version 1 is accurate because a human tells it exactly which objects matter.
Version 2 needs no human setup and runs live, but it must decide on its own,
every time an object comes back into view, whether it has seen it before.
That decision (**re-identification**) is the core problem of version 2.

---

## 3. Version 2: the whole system

### 3.1 The parts

```
 phone (/phone page)                laptop                                   browser
 ───────────────────        ─────────────────────────────────────          ─────────────
 rear camera                server/app.py (FastAPI)                         dashboard (/)
 -> JPEG, 960px,  ──wss──>  /ws/camera ─> newest frame only                 live view
    10 per second           │                                               memory panel
                            v                                               ask bar
                    engine/engine.py  Engine.process(frame)                 rounds
                      detect.py    YOLOE + ByteTrack                          ^
                      identify.py  DINOv3 fingerprints, matching              │
                      memory.py    objects, sightings, snapshots ─ store/     │
                      ask.py       text / photo questions                     │
                      rounds.py    missing / new / moved                      │
                            │                                                 │
                            └── frame + boxes, memory, answers ──wss──────────┘
                                /ws/dashboard

 Tailscale: `tailscale serve` gives the laptop an HTTPS address inside the
 tailnet, which phone browsers require before they allow the camera.
```

Everything runs on one laptop with one GPU (developed on an RTX 3050, 4 GB).
Nothing goes to the cloud.

### 3.2 One frame, start to finish

1. The phone page draws the camera image onto a canvas, scaled so the long side
   is 960 px, encodes it as a JPEG (quality 0.7) and sends it as a binary
   WebSocket message. About 10 a second.
2. The server stores it as `live.latest`, **replacing** whatever frame was
   waiting. There is no queue.
3. The processing loop wakes up, takes the newest frame and runs it in a worker
   thread: decode the JPEG, then `Engine.process(frame, ts)`.
4. The engine:
   1. **detects and tracks**: YOLOE finds objects, ByteTrack gives each a track
      id that stays the same while it stays in view;
   2. **measures motion**: is each object moving in the world, or only on
      screen because the head moved;
   3. **takes fingerprints** of new tracks (first 5 frames) and, once a second,
      of known ones (to collect new viewpoints);
   4. **identifies** any track that has lasted 5 frames: an object we already
      know, or a new one;
   5. **records a sighting** of every identified object, keeps its clearest
      thumbnail, and saves a snapshot when it is at rest.
5. The server sends a `frame` message to every dashboard: the phone's own JPEG
   (not re-encoded) plus the boxes, names and object ids.
6. If an object was created, renamed or moved, or what's in view changed (at
   most once a second), it also sends the whole memory as a `memory` message.

A question ("where's my bottle?") arrives on the same dashboard WebSocket, is
answered from memory under the same lock, and goes back as an `answer` message.

### 3.3 Why it is built this way

- **Detect + track + fingerprint, not one model.** Detection is per frame and
  knows only *kinds* of things. Tracking keeps identity while the object stays
  in view and costs almost nothing. Fingerprints are needed only when a track is
  new, which is when the real question ("seen before?") comes up.
- **Newest frame only.** If processing is slower than the camera, a queue would
  grow and the delay with it. Dropping old frames keeps the delay at one frame.
- **One lock around the engine.** The detector's tracker, the GPU and the memory
  are not safe to use from two threads. Frames, questions, renames, saves and
  rounds all take `live.lock`.
- **A fixed list of words.** You ask by name, so names have to be right. The
  prompt-free detector called a helmet "alarm" and a router "phonebook".

---

## 4. The engine, file by file

The engine is plain Python with no web code. The same `Engine` class runs on a
recorded video (`engine/run.py`) and behind the live server. Tests replace the
detector and the embedder with stand-ins, so they run without a GPU.

### 4.1 `engine/detect.py`: finding objects

`Detector` wraps **YOLOE** (`models/yoloe-11l-seg.pt`, the large segmentation
model). YOLOE is open-vocabulary: you give it words, it finds those things.

| Constant | Value | Why |
|---|---|---|
| `MODEL` | `yoloe-11l-seg.pt` | the large model, ~23 ms a frame on the 3050; it named the helmet and power bank that the small one missed |
| `IMGSZ` | 640 | input size; 960 was slower for little gain (section 9.1) |
| `MIN_CONF` | 0.25 | ultralytics' default; the dashboard dims anything under 0.4 |
| `QUANTIZE` | 16 on GPU, none on CPU | fp16 halves GPU time; on CPU fp16 is ~40x slower |
| `MAX_AREA` | 0.4 | boxes covering more of the frame are walls, the room, the bed you sit on |
| `VOCAB` | ~66 words + "person", "hand" | everyday things people lose; see below |
| `IGNORE` | people, body parts, walls, rooms… | dropped, never stored |
| `SURFACES` | table, desk, bed, shelf… | things objects rest on, used to say *where* |

**`VOCAB`** (`detect.py:28`). The prompt-free model (`yoloe-*-pf.pt`, ~4,500
built-in words) was tried first and named things badly. A short list of things
people actually lose names them far better. Two consequences:
- An object missing from the list is **not detected at all**. Add it to `VOCAB`.
- Anything off the list is forced into the nearest word: a door became "shelf",
  a trimmer and a stapler became "pen". Renaming in the dashboard fixes the name
  for good (section 6).
- "person" and "hand" are in the list only so they can be dropped. Without them,
  a face was forced into the nearest word ("cabinet") and stored.

**`Detector.__call__(frame)`** (`detect.py:51`) runs
`model.track(..., tracker="bytetrack.yaml", persist=True, agnostic_nms=True)`.
- `persist=True` keeps ByteTrack's state between calls, so track ids carry over.
- `agnostic_nms=True`: YOLOE can put two boxes with two labels on one object
  ("remote" and "phone"); suppressing overlaps across labels keeps one.
- It returns `(objects, surfaces)`. Each is `{track_id, label, conf, box}`.
  A surface can also be an object (a book is both), so surfaces are collected
  before the `IGNORE`/`MAX_AREA` filter drops anything.
- It also keeps each track's **segmentation outline** in `self.outlines`, for
  fingerprinting only the object's pixels (but see section 10, problem 1).

**`Detector.boxes(image)`** runs `predict`, not `track`, on a single photo (for
ask-by-photo), so a photo never disturbs the live tracks.

**`Detector.reset()`** clears ByteTrack's state, e.g. before a new video.

### 4.2 `engine/identify.py`: fingerprints and matching

**The fingerprint.** `Embedder` runs **DINOv3** ViT-S/16
(`vit_small_patch16_dinov3.lvd1689m`, timm's copy, since Meta's Hugging Face
repo is gated). DINOv3 is trained without labels to tell images apart, which
makes it better at "*this* remote vs *a* remote" than CLIP (in version 1, CLIP
scored the wrong sunglasses 0.74).

For each box (`identify.py:40`):
1. optionally grey out everything outside the object's outline (`only_object`);
2. crop the box, grown by 10% (`PAD`) so the object's edges stay in;
3. resize to 224x224, normalise with ImageNet mean/std;
4. run the model in fp16 on GPU (fp32 on CPU);
5. L2-normalise. The result is a unit vector, so a dot product of two
   fingerprints is their cosine similarity (1 = identical, ~0 = unrelated).

Two choices here were measured, not guessed:
- **The CLS token** (`global_pool="token"`), not timm's default average of the
  patch tokens. The average gives every crop a large shared component: two
  different objects scored a median 0.64 with it, 0.28 with CLS.
- **Greying out the background** (`only_object`, `identify.py:56`). Crops of an
  object in the hand are mostly hand and desk; on a live test two different held
  objects scored 0.78, more than most pairs of views of the same object. Grey
  is the ImageNet mean colour, which the model sees as zero after normalising.

**The gallery.** Each object keeps up to `GALLERY_SIZE` = 12 fingerprints from
different viewpoints. `add_view` (`identify.py:103`) adds a fingerprint only if
it is less than `NEW_VIEW` = 0.80 similar to every view already stored (so it
really is a new angle), and keeps the newest 12.

**Matching.** `best_match(fingerprint, galleries, exclude)` (`identify.py:80`):
1. Score every object by its **closest** gallery view (one view from the same
   side is enough), skipping objects in `exclude`.
2. The best must score at least `MATCH` = 0.70...
3. ...and beat the second best by `MARGIN` = 0.05.
4. Otherwise: no match, it's a new object.

Two details:
- **`exclude`** is every object visible right now under another track. One
  object can't be in two places at once.
- **The margin ignores look-alike runners-up.** A runner-up that looks like the
  winner (its gallery scores ≥ 0.70 against the winner's) is usually the same
  object stored twice. Counting it would make every duplicate cause more
  duplicates: on sample3, 20 of 116 new objects were created that way.

`MATCH` and `MARGIN` were calibrated on `sample3` only (section 9.2): 70% of
returning tracks re-identified, 7% wrong merges. We lean high on purpose: a
wrong merge gives a **wrong** answer, a missed one only a duplicate object.

### 4.3 `engine/engine.py`: one frame in, boxes out

| Constant | Value | Meaning |
|---|---|---|
| `MIN_TRACK_AGE` | 5 frames | a new track must survive this long before it is identified; flickers die sooner |
| `VIEW_EVERY` | 10 frames | how often a known object's gallery gets a new fingerprint (1 s at 10 fps) |
| `TRACK_TTL` | 30 frames | a track not seen for this long is forgotten (ByteTrack also keeps 30) |
| `STILL` | 0.03 | moving less than 3% of the box diagonal per frame = not moving |
| `STILL_FRAMES` | 3 | still for 3 frames in a row = put down |
| `REST_SNAPSHOT_EVERY` | 0.5 s | snapshot rate while at rest |
| `MOVING_SNAPSHOT_EVERY` | 1.0 s | fallback snapshot rate before it has ever rested |

A `Track` (`engine.py:27`) is the engine's note about one ByteTrack id: its
label, age in frames, the fingerprints collected while new, the object id once
identified, its last centre, and how many frames it has been still.

**`process(frame, ts)`** (`engine.py:49`), in order:
1. `detector(frame)` → objects and surfaces.
2. Create or age a `Track` for every object.
3. `update_motion` (below).
4. `fingerprint`: embed tracks that are new (age ≤ 5) or due a gallery update
   (every 10th frame once identified). All crops of a frame go in one batch.
5. For each object: if its track has no object id and is 5 frames old,
   `identify_track`; if it has one, `observe`.
6. `forget_old_tracks`; update a smoothed fps (how fast the engine *could* go).

Each detection returned has `object_id` (`None` while still being identified),
`name` (the user's name if renamed), and `reidentified` (true on the frame an
old object is recognised, with `last_seen_ago_s`).

**`update_motion`** (`engine.py:84`). The camera is on a head, so it never
stands still and everything moves on screen. Each object's on-screen shift
minus the **median shift of the other objects in view** (an estimate of the
camera's own movement) is its movement in the world. If that is under `STILL`
of its box diagonal, its still-count goes up, otherwise it resets. With only
one object in view there is nothing to subtract, so camera motion counts as
object motion.

**`identify_track`** (`engine.py:119`). Averages the track's first fingerprints
(re-normalised) into one query and calls `best_match`, excluding objects already
visible. No match → `memory.create(...)` with a gallery built from those
fingerprints. Match → the track takes that object's id, `reidentified` is set.

**`observe`** (`engine.py:138`). For an identified object, every frame:
- extend or start a **sighting** (section 4.4);
- maybe replace the thumbnail with a clearer view;
- save a snapshot when due: every 0.5 s while it's at rest, or every 1 s if this
  sighting has never seen it at rest (so an object only ever seen moving still
  gets a picture). The snapshot overwrites the sighting's previous one, so each
  sighting ends up with the **last** picture of it, i.e. where it was left.

**`describe`** (`engine.py:180`) turns a box into words:
- the **surface**: the smallest surface box that contains the object's "foot"
  (bottom-centre of its box) → "on the desk";
- the two **closest other identified objects** by centre distance → "next to
  the mug and the laptop".

**`forget_all`** wipes memory and open tracks together, so a half-identified
track can't carry an old object id into the new memory. **`warm_up`** loads the
text model at startup so the first question isn't slow.

### 4.4 `engine/memory.py`: what is remembered

**Data model.**
- `Object`: `id`, `name` (what the user calls it), `label` (what the detector
  called it), `gallery` (fingerprints), `sightings`, thumbnail bookkeeping.
  `last_seen` and `where` come from its newest sighting.
- `Sighting`: one continuous stretch of seeing an object. `start`, `end`, last
  `box`, `where` in words, `nearby` names, whether it ever `rested`, and when its
  snapshot was taken (`snapshot_at`, -1 = none).

**Sightings.** `sighting(obj, ts, box)` (`memory.py:97`) extends the open
sighting if the object was seen within `SIGHTING_GAP` = 2 s, otherwise starts a
new one. So looking away for a moment doesn't split a sighting.

**Thumbnails.** `update_thumb` (`memory.py:78`) keeps the clearest view. The
first view is usually the object entering the frame, half cut off. A later view
replaces it only if the box doesn't touch the frame edge (within `THUMB_EDGE` =
4 px) and is at least `THUMB_GROWTH` = 1.3x larger. Thumbnails are at most
160 px. Each replacement bumps `thumb_version`, which the dashboard adds to the
URL (`?v=`) so the browser reloads it.

**Snapshots** are the whole frame with a yellow box drawn on the object.

**Storage.** One folder per memory ("store"):
```
store/<name>/
  memory.db              SQLite: objects, sightings (+ rounds, round_seen from rounds.py)
  galleries/<id>.npy     one NumPy array of fingerprints per object
  thumbs/<id>.jpg
  snapshots/<sighting id>.jpg
```
`save()` rewrites the `objects` and `sightings` tables in full (the server calls
it every 10 s and on shutdown). `load()` reads them back on startup. `rename`
changes only `name`. `forget_all` deletes the folder and starts empty.

### 4.5 `engine/ask.py`: answering questions

**By text** (`by_text`, `ask.py:32`):
1. Lower-case, split into words, drop stop words ("where", "my", "the",
   "leave"…).
2. Embed what's left with a small sentence model (`all-MiniLM-L6-v2`, on CPU).
3. Score every object by the better of its **name** and its **label**.
4. Below `ASK_MIN` = 0.50: "I don't remember seeing …", plus the 3 closest names.
   (On a hand-made word list, absent things scored ≤ 0.43 and synonyms
   0.71–0.93; "shades" finds "sunglasses" at 0.71, but "clicker" scores only
   0.42 against "remote control".)
5. Objects within `TIE` = 0.05 of the best are treated as the same thing stored
   twice; the answer is whichever was **seen last**.

**By photo** (`by_photo`, `ask.py:52`): detect objects in the photo, take the
biggest box (or the whole photo if none), fingerprint it with its outline, and
match against every gallery. Below `MATCH` = 0.70, not found.

**The answer** (`answer`, `ask.py:69`) uses the newest sighting that has a
snapshot: "Last seen 4 minutes ago on the desk, next to the mug." plus the
snapshot URL, the box, nearby names and the last 5 sightings for the timeline.
`ago()` says "just now" under 10 s, then seconds, minutes, hours.

### 4.6 `engine/rounds.py`: safety rounds

A **round** is one walk through a space. While it runs, every identified object
in view is noted with its latest sighting (`Rounds.see`, called by the server for
every frame). Ending a round stores it in `memory.db` (tables `rounds`,
`round_seen`) and compares it with the round before (`compare`, `rounds.py:62`):

| Kind | Rule |
|---|---|
| **missing** | seen last round, not this round |
| **new** | seen this round, not last round |
| **moved** | seen in both, and the **surface** under it changed ("desk" → "bed"); both rounds must have seen a surface |
| **probably moved** | one missing and one new object with the **same label**, the only one of that label on each side: reported as one move, not two alarms |

The last rule exists because re-identification isn't reliable: an object that
was picked up and put down often comes back as a new object.

A rule "its neighbours changed" was tried and **dropped**: neighbour names come
from the same unsure labels ("next to the bottle" with six bottles), and on a
replay of desk1 it flagged 13 objects that had not moved (`moved`, `rounds.py:48`).

Changes are sorted missing → moved → new, each with before/after snapshot URLs.
The first round has nothing to compare with, so its report is empty.

### 4.7 `engine/video.py` and `engine/run.py`

`frames(path)` yields `(seconds, frame)` from a video file, sampled at 10 fps
and resized to a 960 px long side, i.e. exactly what the phone sends. Given a
**folder** of frames recorded by the server (`<unix time>.jpg`), it replays every
frame at its recorded time.

`python -m engine.run VIDEO_OR_FOLDER --store store/x --fresh [--ask "remote" ...]`
runs the whole engine on it, saves the memory, and prints fps, object count,
re-identifications and the memory (or the answers). This is how changes get
scored without the phone.

---

## 5. The server

`server/app.py`, FastAPI + uvicorn. It joins the phone, the engine and the
dashboard.

### 5.1 State and threads

`Live` (`app.py:76`) holds everything shared: the `Engine`, the lock, the newest
frame, the connected dashboards and cameras, a frame counter, what's in view,
recent finish times (for fps) and latencies, the `Rounds` store, the running
round, and the recording folder.

The event loop handles sockets. Heavy work (a frame, a question, a save, ending
a round) runs in a worker thread via `asyncio.to_thread`, under `live.lock`.

### 5.2 Background tasks (started in `lifespan`, `app.py:296`)

| Task | What it does |
|---|---|
| `process_loop` | waits for a new frame, runs `run_frame`, broadcasts the `frame` message, and the `memory` message when needed |
| `save_loop` | `memory.save()` every 10 s, so a crash loses at most 10 s |
| `report_loop` | logs fps and server latency (p50, p95) every 30 s |
| `video_loop` | only with `--video`: plays a file or a recorded folder into the same path as the phone, at real speed and 10 fps |

A task that crashes would otherwise die silently; `report_crash` prints why, so a
frozen dashboard has a reason in the log. A frame that throws is logged and
skipped, never stops the loop.

### 5.3 Sending to many pages

`broadcast` (`app.py:119`) sends to every socket at once. A socket that can't
take a message within `SEND_TIMEOUT` = 2 s (a stalled phone or a tab in the
background) is dropped; its page reconnects by itself. Before this, one stalled
socket blocked the whole frame loop.

The `memory` message carries the whole memory, so it is sent only when it
matters: at once when an object is created, renamed or moves, but at most once
a second (`VIEW_UPDATE_EVERY`) when only "what's in view" changed, since that
changes nearly every frame.

### 5.4 Recording

The Record button (dashboard or phone), or `--record FOLDER` at launch, saves
every processed JPEG as `<unix time>.jpg`, exactly as the engine got it. When
recording stops, `make_mp4` turns the folder into an H.264 `.mp4` beside it, at
the rate frames actually arrived (needs `ffmpeg`). The folder replays exactly
with `--video FOLDER` or `python -m engine.run FOLDER`, which is how a live
session becomes a repeatable test.

### 5.5 Security

The server listens on `127.0.0.1` only; `tailscale serve` is what makes it
reachable, over HTTPS, inside the tailnet.
- **Host check** (`TrustedHostMiddleware`, `ALLOWED_HOSTS`): only `localhost`,
  `127.0.0.1`, `::1` and this machine's own Tailscale name (more via
  `GM_ALLOW_HOSTS=a,b`). This blocks DNS rebinding, where some website points
  its own name at 127.0.0.1.
- **Origin check** on both WebSockets (`same_origin`, `app.py:333`): browsers let
  any page open a WebSocket to any host, so without this a web page open on the
  laptop could watch the camera or forget everything through localhost.
- **There is no login.** Anyone on the tailnet can open the dashboard (section 10).

### 5.6 Endpoints and flags

| Path | What |
|---|---|
| `/` | the built dashboard (`dashboard/dist`) |
| `/phone` | the camera page |
| `/ws/camera` | binary JPEG frames in; text = `record_start`/`record_stop`; sends `camera_ack` (fps) once a second and `recording` |
| `/ws/dashboard` | everything else (section 7) |
| `/thumb/<object id>.jpg`, `/snapshot/<sighting id>.jpg` | images from the store |
| `/keyframe/<name>` | not built, always 404 |

```
python -m server.app [--store store/live] [--host 127.0.0.1] [--port 8000]
                     [--video FILE_OR_FOLDER [--loop]] [--record FOLDER]
```

---

## 6. The phone page and the dashboard

`dashboard/`: React 19 + TypeScript + Vite, no UI library. Two pages from one
build: `index.html` (dashboard) and `phone/index.html`.

**`src/socket.ts`.** `useSocket(path, onMessage)`: a WebSocket to the same host
the page came from (`wss:` on HTTPS), which reconnects by itself after 0.5, 1, 2,
then every 4 s. The message handler is read through a ref, so re-rendering never
tears the socket down.

**`src/protocol.ts`.** TypeScript types for every message in section 7.

**Phone page, `src/phone/Phone.tsx`.** Start/Stop, rear camera at 1280x720 ideal,
a preview, the sending rate, the server's fps, and a Record button.
- Every 100 ms (`SEND_FPS` = 10): draw the video onto a canvas at a 960 px long
  side, `toBlob` as JPEG 0.7, send. It **skips** a frame if the previous one is
  still encoding or if more than 512 KB is waiting in the socket's buffer
  (`MAX_BUFFERED`), so a slow network drops frames instead of building delay.
- Keeps the screen awake with the Wake Lock API, and re-takes it when the page
  becomes visible again.
- Camera access needs HTTPS (browser rule), which Tailscale provides.

**Dashboard, `src/dashboard/`.**
- `App.tsx`: one socket, routes each message type to the right state. A question
  that gets no answer in 10 s stops showing "Searching".
- `LiveView.tsx`: draws frames on a canvas. Frames go from the socket to the
  canvas through a `FrameBus`, **outside React state**, because re-rendering the
  page 10 times a second would be wasted work. Boxes glide to their new place
  over 120 ms, low-confidence ones (< 0.4) are dim and dashed, a re-identified
  object shows "seen 3m ago" for 2.5 s, and 2 s without a frame shows the camera
  as stopped. `?clock` adds a millisecond clock for measuring glass-to-glass
  latency (point the phone at the screen, compare the two clocks).
- `MemoryPanel.tsx`: every object with thumbnail, name, when and where last
  seen, a "live" mark if in view; new ones highlighted for 4 s; filter by name;
  **click a name to rename it**.
- `AskBar.tsx`: text box, microphone (browser speech recognition, hidden where
  unsupported), ask-by-photo, and the answer card: snapshot with the box, the
  sentence, nearby chips, a timeline of past sightings. Answers are spoken aloud
  (`speechSynthesis`) unless muted.
- `Rounds.tsx`: Start/End round, live change alerts, the end-of-round report.
- `Forget.tsx`: "Forget everything", with an in-page confirm step.

**Mock server.** `dashboard/mock/mock_server.py` plays a scripted scene so the UI
can be built and tested with no GPU and no footage (`npm run mock`).

---

## 7. The protocol

All JSON; times are Unix seconds; boxes are `[x1, y1, x2, y2]` in the pixels of
the frame they came with. Full examples: [`dashboard-brief.md`](dashboard-brief.md) section 4.

**Server → dashboard**

| type | when | carries |
|---|---|---|
| `frame` | every processed frame | `seq`, `ts`, `width`, `height`, `jpeg_b64`, `detections[]` (`track_id`, `object_id`, `label`, `name`, `conf`, `box`, `reidentified`, `last_seen_ago_s`), `stats` (`fps`, `latency_ms`, `in_view`, `in_memory`) |
| `memory` | on connect, on any change | `objects[]`: `object_id`, `name`, `label`, `last_seen`, `where`, `thumb`, `in_view` |
| `answer` | reply to `ask` / `ask_photo` | `found`, `text`, `object_id`, `snapshot`, `box`, `nearby`, `seen_at`, `sightings[]`, `candidates[]` |
| `round` | round started / ended | `state` running: `round_id`, `name`, `started`; ended: `report[]`, `seen`, `first` |
| `change` | defined, for live alerts during a round | `kind` missing/new/moved, `object_id`, `name`, `text`, `before`, `after` |
| `recording` | recording toggled | `on`, `folder`, `video` (the .mp4, once made) |

**Dashboard → server:** `ask {query}`, `ask_photo {jpeg_b64}`,
`rename {object_id, name}`, `round_start {name}`, `round_end`, `forget_all`,
`record_start`, `record_stop`. A malformed message is logged and dropped; it
never closes the socket.

**Phone ↔ server:** binary JPEG frames in; `camera_ack {fps}` and `recording`
out; `record_start`/`record_stop` as text in.

---

## 8. Version 1, in short

Still in the repo and still works (`python run.py video.mp4`). Full detail:
[`HANDBOOK.md`](HANDBOOK.md).

1. **Register** (`register.py`): scrub to a close-up of each object, drag a box,
   type its name → `registry/<video>.json`.
2. **Track** (`track.py`): SAM 2 (base+) follows every registered object
   through the video in one pass, at 10 fps, 1024 px. `LazyFrames` decodes
   frames on demand because SAM 2's own loader needs ~8 GB of RAM for one minute.
3. **Sleep** (`remember.py`): group frames into episodes, drop glitches shorter
   than 0.3 s, drop episodes whose crop looks less than 0.70 like the registered
   object (CLIP), pick the last clear frame as "where it was left".
4. **Ask** (`ask_page.py`): one offline HTML page, keyword matching, the
   snapshot and the clip around the moment it was put down.

How we got there, each step fixing the last one's failure: YOLO-World per frame
(0/5 on video 3) → OWLv2 image search (scores didn't separate look-alikes) →
SAM 2 tracking (3/5) → all objects in one pass (6x faster) → SAM 2 base+ →
the sleep step (4/5) → validation on untouched video 4: **3/5 correct, 1 partial,
1 wrong.** The old attempts are kept in `experiments/`.

Version 2 exists because version 1 needs a human to register every object and
takes 5 minutes per minute of video: fine for a demo, not for glasses.

---

## 9. Measuring: tools and every result so far

### 9.1 Speed (`eval/phase0_speed.py`, `eval/phase0_results.md`)

GPU time only, RTX 3050, sample3, prompt-free models:

| Detector | DINOv3 | total p50 | fps |
|---|---|---|---|
| yoloe-11s-pf, 640 | ViT-S | 15.9 ms | 55.9 |
| yoloe-11s-pf, 960 | ViT-S | 20.5 ms | 44.3 |
| yoloe-11m-pf, 640 | ViT-B | 19.4 ms | 47.1 |
| yoloe-11l-pf, 640 | ViT-B | 24.3 ms | 38.6 |

Whole engine (detect, track, identify, memory, snapshots): 35 fps on sample3 with
the small model; on sample5, 23.7 fps before and ~17.5 fps after moving to the
large model with the word list. The phone sends 10, so speed is not the
constraint. GPU memory stays under ~1.1 GB of 4.

### 9.2 Identity thresholds (`eval/calibrate_identity.py`)

Needs no hand labels: the tracker gives free ground truth (one track = one
object). Each long track is split; the first half builds a gallery, five frames
from the second half are the "returning" object. With its own gallery present a
match should pick it (genuine); with it removed, any match is a wrong merge
(impostor). Errors are upper bounds, since two tracks of one object count as
different. Run only on calibration videos.

Result on sample3 at MATCH 0.70 / MARGIN 0.05: 70% re-identified, 7% wrong
merges. Changes it caused: the CLS token (different objects 0.64 → 0.28), and
margin against different-looking runners-up only (116 → 103 objects).

### 9.3 "Where" answers (`eval/where.py`, `eval/phase1_results.md`)

sample4 (test video, nothing tuned on it), prompt-free detector: **on time by
photo 1/5, by text 0/5** (version 1: 3/5). In 4 of 5 the photo found the right
object but an older sighting: the later sighting had been stored as a separate
object. Text failed on wrong labels ("toy gun", "bookmark", "pen").

### 9.4 sample5 (answer key in [`demo/README.md`](demo/README.md))

| Version | Objects stored | Re-IDs | Answer key |
|---|---|---|---|
| master (small, prompt-free) | 83 | 90 | 0/4 |
| + word list + large model | 59 | 38 | 1/4 (perfume) |
| + masked fingerprints | 54 | 43 | 1/4 |
| + "just vanished" rule (later reverted) | 50 | 47 | 1/4 |

Fewer objects is **not** proof of better: sample5 has no ground truth for
wrong merges.

Why new objects get created (sample5, masked): 97 identifications, 43 matched,
50 new. 47 of the 50 new ones scored under 0.70 (mostly 0.5–0.7); only 3 were
blocked by the margin.

### 9.5 Live tests (phone, a real room)

- ~10 real things → **50 objects in 57 s**. Duplicates: a syrup bottle ×4, a
  glass cleaner ×4, a tripod ×~5, a stapler ×2.
- Moved-object test: about **1–2 of 5** answered right; the same with a slow pan
  and a fast one, so filming speed isn't the cause.

### 9.6 Can a threshold fix it? No.

Fingerprint similarity on `store/slow-test`, 9 hand-labelled groups: **same
object median 0.60 (min 0.35), different objects median 0.37 (max 0.78).** The
two ranges overlap; at ≥ 0.6 we'd catch half the duplicates and make 13 wrong
pairs out of 297. Most objects had only 1–2 gallery views.

### 9.7 Is the tracker the problem? No.

On sample5: ByteTrack gives 193 track ids / 50 objects; ByteTrack with conf 0.1
(re-enabling its low-score stage) 189 / 54; BoT-SORT with camera-motion
compensation 193 / 57. The ~190 tracks are real exits and re-entries, so the
weak step is the fingerprint match. That's also why the "just vanished" rule
(a new track near where an object just disappeared needs only 0.50) was
reverted: it assumed the tracker loses objects briefly, and it recovered ~4.

### 9.8 Names (desk1 replay)

26 of 45 judgeable objects named right. Part of the reason: the label is fixed
from a track's **first** detection (`engine.py:59`), often a blurry, half-in-frame
view, while the thumbnail is the clearest one.

---

## 10. Known problems

1. **Live fingerprints are not masked right now.** `only_object` exists and
   ask-by-photo uses it, but `Engine.fingerprint` (`engine.py:111`) passes no
   outlines. The engine call was added together with the "just vanished" rule
   and removed when that rule was reverted (`43ec298`). Fix: pass
   `[self.detector.outlines.get(o["track_id"]) for o in todo]` again, and
   re-measure.
2. **Re-identification** (sections 9.4–9.7). The same object scores ~0.6
   against itself; different objects can reach 0.78. This causes duplicates,
   "last seen" answers that stop too early, and moves reported as missing + new.
3. **Identifying after 5 frames is early.** Those are often the blurriest,
   most cut-off views.
4. **Labels come from the first detection** and can flip between frames
   (bottle / deodorant / laptop). A majority vote over the track would be cheap.
5. **Closed vocabulary.** Off-list things are missed or misnamed. Agreed
   direction: names come from the user (rename) plus thumbnails as a picture wall.
6. **No login.** Tailscale is the only gate.
7. **Rounds "moved" needs a surface** under the object in both rounds; objects
   with no surface detected under them can only show as missing/new.
8. **Live `change` messages** are in the protocol and the UI, but the server
   only sends the report when a round ends.
9. **`/keyframe`** is a stub.
10. **Small things:** the README's live test count is out of date (37 now);
    `pytest` isn't in `requirements-live.txt`; Android may kill the Tailscale
    app in the background (set its battery mode to Unrestricted).

---

## 11. What's next

Agreed order, each change scored on the desk1 replay (objects stored vs the real
count, answer-key accuracy, wrong merges checked on thumbnails):

1. **A ground truth for live sessions.** Record a session with `--record`, write
   down the object list and the moves → an answer key. Everything else gets
   measured against it.
2. **Fingerprint from clear views**, not the first 5 frames.
3. **A merge pass**: later, join objects that look alike. "Never seen at the same
   time" and "same label" are strong hints, not hard rules (same-object pairs
   *were* seen together in slow-test; labels flip).
4. **Top-k gallery scoring** instead of the single closest view.

Also untested: DINOv3 base/large instead of small.

---

## 12. Setup, running, testing

**Needs:** Python 3.11 (newer Pythons lack the torch wheels), Node 20+, ideally an
NVIDIA GPU (4 GB is enough; on CPU it runs at ~4 fps).

```bash
git clone https://github.com/metro-dev26/glasses-memory && cd glasses-memory
python3.11 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or .../whl/cpu
pip install -r requirements-live.txt pytest
cd dashboard && npm install && npm run build && cd ..
```
YOLOE, DINOv3 and the text model download themselves on first run.
(Version 1 needs `requirements.txt` and `./setup.sh` for the SAM 2 checkpoints.)

**Live, with the phone:**
```bash
python -m server.app --store store/live      # dashboard at http://localhost:8000
tailscale serve --bg 8000                     # once per boot; prints the HTTPS address
```
On the phone (Tailscale on, same tailnet): `https://<machine>.<tailnet>.ts.net/phone`.
One-time Tailscale setup is in [`server/README.md`](../server/README.md).

**A recorded video or recording, through the dashboard:**
`python -m server.app --video videos/x.mp4 --loop`

**Engine only, printing results:**
`python -m engine.run videos/x.mp4 --store store/x --fresh --ask "stapler"`

**Tests:** `python -m pytest -q` (37 tests, ~11 s; detector and embedder are
stand-ins, so no GPU needed). They cover detection filtering, matching and
galleries, memory save/load, answers, rounds, video reading, and the
camera-relative stillness rule.

**UI without a GPU:** `cd dashboard && npm run mock`.

Footage (`videos/`) and memories (`store/`) are git-ignored and never committed.

---

## 13. Every file in the repo

**Version 2**
- `engine/detect.py`: YOLOE + ByteTrack, the word list, what to ignore, surfaces.
- `engine/identify.py`: DINOv3 fingerprints, galleries, matching.
- `engine/engine.py`: the per-frame loop, motion, snapshots, "where" in words.
- `engine/memory.py`: objects, sightings, thumbnails, SQLite storage.
- `engine/ask.py`: questions by text and by photo.
- `engine/rounds.py`: safety rounds and their comparison.
- `engine/video.py`: read a video or a recorded folder like the phone sends it.
- `engine/run.py`: run the engine on a video from the command line.
- `server/app.py`: the live server. `server/README.md`: how to run it.
- `dashboard/src/…`: the dashboard and phone page (section 6).
  `dashboard/mock/mock_server.py`: fake server for UI work.

**Version 1**
- `register.py`, `track.py`, `remember.py`, `ask_page.py`, `run.py`, `setup.sh`,
  `requirements.txt`, `registry/*.json` (boxes for sample3 and sample4).
- `experiments/`: YOLO-World, OWLv2, SAM 2 one-at-a-time.

**Measuring**
- `eval/phase0_speed.py` + `phase0_results.md`: speed.
- `eval/calibrate_identity.py`: identity thresholds from a calibration video.
- `eval/where.py` + `eval/truth/sample4.json` + `phase1_results.md`: "where" answers.
- `docs/demo/README.md`: sample5 and its answer key.

**Tests:** `tests/` (`conftest.py` holds the stand-ins).

**Docs:** this file, `HANDBOOK.md`, `design/live-glasses-memory.md`,
`dashboard-brief.md`, `plans/2026-10-09-live-engine-phase-0-1.md` (the
step-by-step build plan for phases 0–1), `demo.gif`.

---

## 14. Rules we follow

From [`HANDBOOK.md`](HANDBOOK.md) section 8, in short:
- Never tune on the test video. Report the real number, even when it's worse.
- Personal footage never gets committed.
- Every change gets a before/after number.
- Both of us must be able to explain every line.
- Branch → pull request → the other person reviews → merge. `master` stays working.
- Conventional commits (`feat:`, `fix:`, `docs:`), the body says why. No
  co-author trailers.
- Small, readable code: short functions, a module docstring with a usage line,
  constants at the top with a comment saying why that value.

---

## 15. Glossary

- **Open-vocabulary detection**: finding objects from words you give it, not a
  fixed class list (YOLOE, YOLO-World).
- **Tracking / track id**: linking boxes across frames so the same object keeps
  the same number while it stays in view (ByteTrack).
- **Re-identification (re-ID)**: recognising an object again after it left the
  frame and got a new track id.
- **Fingerprint / embedding**: a vector describing an image. **Cosine
  similarity** of two unit vectors is their dot product; near 1 = look alike.
- **Gallery**: the set of fingerprints stored per object, one per viewpoint.
- **CLS token**: the one output of a vision transformer meant to summarise the
  whole image.
- **Segmentation outline / mask**: which pixels belong to the object.
- **Sighting**: one continuous stretch of seeing an object (gaps ≤ 2 s).
- **Snapshot**: the saved frame showing where an object was during a sighting.
- **Surface**: something objects rest on (desk, bed, shelf), used for "where".
- **Round**: one walk through a space; compared with the previous one.
- **Calibration vs test video**: thresholds are set on calibration videos only;
  test videos are for honest numbers.
- **Glass-to-glass latency**: delay from light hitting the phone's camera to the
  picture on the dashboard screen.
- **Tailscale / tailnet**: a private network between your devices; `tailscale
  serve` gives the laptop an HTTPS address inside it.
- **DNS rebinding**: a website pointing its own name at 127.0.0.1 to reach
  servers on your machine; blocked by the host check.
