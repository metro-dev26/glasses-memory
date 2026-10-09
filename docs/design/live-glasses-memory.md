# Live glasses-memory: design

Status: agreed design, 2026-10-09. Section 7 (evaluation) and the phase plan
are proposed and still open for review. Every number marked "unmeasured" is a
target, not a result.

Read this first, then `docs/HANDBOOK.md` (how version 1 works and the rules we
follow), then `docs/dashboard-brief.md` (the UI and the server protocol).

---

## 1. What we are building

Version 1 of glasses-memory works on a **recorded** video: you register each
object by drawing a box, SAM 2 tracks it offline (about 5 minutes per minute of
video on an RTX 3050), and a static page answers "where's my remote?". On the
untuned validation video it got 3 of 5 objects right.

Version 2 makes it **live** and removes the manual registration:

- A phone held at eye level plays the role of smart glasses and streams video.
- A GPU server **detects every object** in view automatically, tracks it, and
  **recognises it again** when it comes back, even minutes later.
- A dashboard shows the live view with small boxes and labels on every object,
  a memory panel that fills up as objects are seen, and an ask box.
- You ask by voice or text, "where are my keys?", and get the last place it was
  seen: a snapshot, the time, and the nearby objects ("on the desk, next to the
  laptop").
- **Rounds:** you walk the same space twice and it tells you what is
  **missing, new, or moved** since the last walk.
- Nothing is kept as video: only object records, feature vectors, and small
  thumbnails, and one button forgets everything.

## 2. Why this shape

**The task is an open research problem.** Ego4D's Episodic Memory "Visual
Queries" benchmark (Meta) is exactly "where did I last see X?" in first-person
video. The *online* version, where each frame is seen once and only a compact
memory is kept, was defined in ESOM (WACV 2026). The best published online
result is about **4% success**. ESOM's own analysis: with perfect object
discovery and perfect tracking it would reach about 82%, so discovery and
tracking are where the gains are. Our design attacks exactly those two parts.

**The industry is heading here.** Ray-Ban Meta glasses already answer "where
did I leave my keys?" for things the user asked them to remember, and Meta's
reported next prototype watches continuously and keeps metadata instead of raw
footage. We build the open, measurable version of that idea.

**The use case beyond the home is inspection rounds.** In industrial sites such
as water and wastewater treatment plants, operators walk the same inspection
route every shift. The rounds feature ("what changed since the last round?")
is the version of our memory that matters there: spotting change is what an
experienced operator does. Gauge reading is planned for later (section 10).

## 3. Decisions already made, and why

| Decision | Choice | Reason |
|---|---|---|
| What "live" means | Live camera, not upload-and-wait | It is what glasses do, and it is the hard ML problem |
| Where the live view is shown | Laptop browser dashboard; the phone is only the camera | It can be screen-shared on a call; room for the memory panel |
| Compute | Abhinav's laptop, RTX 3050 4 GB, reached over Tailscale | Available now; Sujan's laptop has no NVIDIA GPU |
| Engine approach | **A: detect every frame, link with a tracker, re-identify with DINOv3** | Real-time systems work this way; cost does not grow per object the way mask trackers do |
| Rejected: B | Sparse detection + EdgeTAM mask tracking (ESOM's design) | A SAM-style tracker's cost grows with each object followed; dozens of objects on a 3050 will not be live |
| Rejected: C | Send keyframes to a large cloud vision-language model | Not our ML, costs per frame, adds latency, sends video off-device |
| Inspection features | Rounds / change detection and object memory now; gauge reading later | Change detection only exists because we have memory; it is what an inspection round needs |
| UI | Built in parallel by Abhinav's Claude from `docs/dashboard-brief.md` | Clear contract, so both halves connect |

## 4. System layout

```
 PHONE (the "glasses")         ABHINAV'S LAPTOP (RTX 3050)               SUJAN'S LAPTOP
 /phone page, rear camera  --> server: detect -> track -> identify   --> dashboard  /
 ~10 JPEG frames/sec           -> memory -> rounds / change detection    live view, memory,
 over /ws/camera                                                      <-- ask, rounds
                                                                         over /ws/dashboard
```

- **Networking:** the three devices are on different home networks. Tailscale
  puts them on one private network and provides the HTTPS certificate that
  phone browsers require before they allow camera access.
- **Protocol:** all messages are defined in `docs/dashboard-brief.md`
  section 4. That file is the contract between the server and the UI.
- **Voice:** speech recognition and text-to-speech run in the dashboard's
  browser (Web Speech API), so they cost no GPU.

## 5. The engine, step by step

Per incoming frame:

1. **Detect: YOLOE, prompt-free mode.** Finds every object with a box, a label
   from its built-in vocabulary, and a confidence. Labels are a hint, not
   identity: the same remote may be called "remote" then "phone".
2. **Track: ByteTrack.** Links boxes across consecutive frames into tracks with
   stable track ids. Nearly free compared with detection.
3. **Identify: DINOv3** (small ViT). Run only when a track is new or comes back,
   not every frame. Produces an identity fingerprint of the object crop.
4. **Memory:** record the sighting (object, box, time, keyframe, thumbnail).
5. **Keyframes:** save one whenever the view has changed enough (about every
   2 s while walking), for rounds.
6. Send boxes, labels, ids and memory updates to the dashboard.

EdgeTAM (a 22x faster SAM 2 variant, CVPR 2025) is kept for one job: drawing a
precise mask on the snapshot of the object you asked about. It is not in the
per-frame loop.

### 5.1 Identity: the same object after it leaves and returns

Two kinds of id:

| | Track id | Object id |
|---|---|---|
| Comes from | ByteTrack | DINOv3 matching |
| Lives | while the object stays in view | for the whole memory |
| Breaks when | the object leaves the frame | two objects look nearly identical |

When a new track appears:
1. Wait until it survives about 5 frames, so flickers are ignored.
2. Take DINOv3 fingerprints from several of those frames.
3. Compare with every known object's **gallery**: each object keeps several
   fingerprints from different viewpoints (a remote from the side and from
   above look different). New views are added when they differ enough.
4. **Match** only if similarity clears a threshold **and** beats the
   second-best candidate by a margin. Otherwise create a new object. The margin
   rule is what stops look-alikes merging (version 1's sunglasses failure).

Known limit: two truly identical objects (two of the same pen) cannot be told
apart by appearance. The system says "I know two of these" instead of guessing.

### 5.2 What each object remembers

Name (auto label, renamable: "cup" -> "my blue mug"), label, fingerprint
gallery, and every sighting: time span, box, keyframe, thumbnail.

### 5.3 Answering "where"

- **Where it was left:** the last frame of the last sighting in which the box
  **stopped moving**. (Version 1 used area, and the snapshot sometimes still
  showed the object in a hand.)
- **Where, in words, without 3D:** the objects near it in that frame: "on the
  desk, next to the laptop".
- **By text:** the question is matched to object names with a text embedding,
  not keyword overlap, so "clicker" can find "remote".
- **By photo:** upload a photo; it is matched by DINOv3 fingerprint.

### 5.4 Privacy

No continuous video is stored. Kept: object records, fingerprints, small
keyframe thumbnails used for answers and rounds. "Forget everything" wipes
them. Storage: SQLite plus NumPy arrays for fingerprints.

## 6. Rounds: "what changed since last round?"

1. **Start round / End round** in the dashboard. Each round is saved and
   compared with the previous one.
2. **Keyframes** in a round hold: a DINOv3 summary of the whole view (place
   fingerprint), the objects in view with positions, a thumbnail.
3. **Place match:** each new keyframe is matched to the most similar keyframe
   of the last round. It counts only if similar enough **and** the two views
   share some of the same objects, so one white wall does not match another.
4. **Object diff** inside matched views:
   - **missing:** in view last round, not now
   - **new:** here now, not last round
   - **moved:** same object, different place, judged **relative to nearby
     objects that did not move**, not raw pixels (the camera angle is never the same)
5. **No false alarms from a passing hand:** a change is reported only if it
   holds across at least two matched keyframes, and "missing" only when the old
   spot is clearly in view and not blocked.
6. Alerts appear live while walking; End round produces a report.

Known limit: only **objects the detector can see**. A puddle, leak or stain is
not an object to YOLOE; detecting it needs pixel comparison between aligned
views from a moving camera. Named as future work, not promised.

## 7. Evaluation (proposed, open for review)

Rules carried over from the handbook, non-negotiable:
- Thresholds are calibrated on **calibration videos** only. **Test videos are
  never used for tuning.**
- Ground truth is **written down before** the system is run on a video.
- Report the real number even when it is worse.

| What | How | Metric |
|---|---|---|
| Speed | Recorded phone video replayed on the 3050 | fps, GPU memory, end-to-end latency (phone to dashboard) |
| Identity | Videos where objects leave and return; ground truth list of returns | correct re-identifications, wrong merges, wrong splits |
| "Where" answers | At least 3 new test videos, at least 20 objects in total, truth written before recording | correct / partial / wrong, same scoring as version 1 |
| Rounds | Two walks; between them stage changes (e.g. move 5, remove 3, add 2), written down before walk 2 | precision and recall of reported changes |
| Benchmark | Ego4D Episodic Memory VQ2D, online setting (license approval takes about 48 hours) | the same success metric ESOM reports, on the subset our compute allows, stated as such |

## 8. Phases

0. **Feasibility test (before anything else).** On the 3050, run YOLOE +
   ByteTrack + DINOv3 on a recorded phone video. Measure fps and GPU memory.
   Target about 10 fps or better (unmeasured). If it falls short: smaller model
   sizes, lower input resolution, DINOv3 on fewer frames. Needs Tailscale + SSH
   to Abhinav's laptop first.
1. **Engine on recorded video:** detect, track, identify, memory, answers, plus
   the evaluation harness for speed, identity and "where".
2. **Live:** server with `/ws/camera` and `/ws/dashboard` per the protocol;
   phone page; dashboard connected (built in parallel against the mock).
3. **Rounds and change detection,** with its evaluation.
4. **Ego4D benchmark run.**
5. Later: gauge reading, spatial (3D) answers.

## 9. Repository layout (proposed)

```
glasses-memory/
  (version 1 files stay as they are: register.py, track.py, remember.py, ...)
  experiments/   earlier approaches (version 1 history)
  engine/        detect, track, identify, memory, rounds  (Python package)
  server/        FastAPI app: websockets, image endpoints
  dashboard/     React + TypeScript UI, and dashboard/mock/ (see the brief)
  eval/          evaluation scripts and ground-truth files (no footage)
  docs/          HANDBOOK.md, dashboard-brief.md, design/
```

## 10. Later

- **Gauge reading:** read analog pressure gauges during a
  round and log the value with time and place. Needle and scale detection is a
  separate, well-studied ML problem. Test with a cheap hardware-shop gauge.
- **Spatial answers:** a 3D map from ordinary video (as in SpatialMem,
  ConceptGraphs) for answers like "on the shelf, left of the door".

## 11. Risks and unknowns

- **Real-time on 4 GB is unmeasured.** Phase 0 exists to find out before design
  work goes further.
- **Round-trip latency** phone -> Abhinav's house -> Sujan's laptop over home
  internet. Boxes may lag the picture; measured in phase 0/2.
- **Open-vocabulary labels are sometimes wrong** ("cup" for a candle). Shown
  with confidence; identity never depends on the label.
- **The demo depends on Abhinav's laptop being on.** Every demo also gets a
  recorded backup.

## 12. References

- ESOM, Online Episodic Memory Visual Query Localization (WACV 2026): https://arxiv.org/abs/2411.16934
- EAGLE, unified 2D-3D visual query localization: https://arxiv.org/abs/2511.08007
- Ego4D dataset and license: https://github.com/facebookresearch/Ego4d
- YOLOE: https://docs.ultralytics.com/models/yoloe
- EdgeTAM: https://arxiv.org/abs/2501.07256
- DINOv3 overview: https://www.lightly.ai/blog/dinov3
- SpatialMem: https://arxiv.org/abs/2601.14895
