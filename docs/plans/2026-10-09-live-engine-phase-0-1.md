# Live engine, phases 0 and 1: implementation plan

> **Superseded.** Phases 0 and 1 were built and measured in `engine/` and
> `eval/` (see `eval/phase0_results.md`, `eval/phase1_results.md`), with a
> different module layout from the one below. Kept for its test designs, which
> the next plan reuses; do not execute it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether YOLOE + ByteTrack + DINOv3 run live on an RTX 3050 (phase 0), and build the engine that turns a recorded video into an object memory with lasting identities, "where it was left" answers, and an evaluation harness (phase 1).

**Architecture:** A new `engine/` package. Each frame: YOLOE (prompt-free) detects, ultralytics' built-in ByteTrack links boxes into tracks, DINOv3 fingerprints new tracks, an `IdentityResolver` maps tracks to lasting object ids, open sightings track where each object came to rest, and a SQLite `MemoryStore` keeps the result. `Engine` wires these together with injected components, so the pipeline is tested with fakes and no models. `eval/` holds the phase 0 speed test and the answer scorer.

**Tech Stack:** Python 3.11, ultralytics >= 8.4.0 (YOLOE, ByteTrack), transformers >= 4.56 (DINOv3), sentence-transformers (question matching), OpenCV, NumPy, SQLite (stdlib), pytest.

**Spec:** `docs/design/live-glasses-memory.md` (sections 5, 7, 8). Version 1 rules: `docs/HANDBOOK.md` section 8.

## Global Constraints

- Python **3.11** (newer Pythons lack the torch wheels).
- Match the repo's hand: no type hints except where a dataclass requires them; a module docstring with a usage line; constants at the top of each module with a comment saying why that value; short functions.
- `print()` is fine in CLI entry points; library modules do not print.
- No comments that narrate how the code was written. Comments explain the code.
- **No personal footage in the repo.** Videos and memory output stay gitignored.
- **Thresholds are calibrated on calibration videos only. Test videos are never used for tuning.** Ground truth is written down before the system is run on a video.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`), body explains why. **No co-author trailers.**
- Version 1 files (`register.py`, `track.py`, `remember.py`, `ask_page.py`, `run.py`) are not modified.
- Tests that need downloaded model weights carry `@pytest.mark.models`; the fast suite is `pytest -m "not models"`.

## Review Focus

1. **No detections, or the tracker has not assigned ids yet** (blank frame, first frame): `Detector.detect` returns `[]`, nothing downstream crashes. Test in Task 2.
2. **Boxes at the frame edge, or tiny boxes:** crops are clamped to the frame; boxes under 16 px get no fingerprint and the track stays undecided. Test in Task 3.
3. **Two look-alike objects in view at the same time** must never be merged into one object id. Test in Task 6.
4. **Asking before anything is remembered, or about an unknown object:** a "not found" answer with closest names, never a crash. Test in Task 10.
5. **Unreadable or missing video file:** a clear `ValueError`, not the division by zero version 1's `LazyFrames` would hit when the reported fps is 0. Test in Task 1.

## Not in this plan

From spec section 5, deferred to the phase 2 plan, where the dashboard first needs them:
- **Ask by photo** (match an uploaded photo by DINOv3 fingerprint).
- **EdgeTAM mask** on the answer snapshot (this plan draws the box only).

## File structure

| File | Responsibility |
|---|---|
| `requirements-live.txt` | Dependencies for the live engine (version 1 keeps `requirements.txt`) |
| `pytest.ini` | Test paths, `models` marker, repo root on the import path |
| `engine/__init__.py` | Package marker |
| `engine/frames.py` | `VideoFrames`: read a video at 10 fps, one frame at a time |
| `engine/detect.py` | `Detection`, `Detector`: YOLOE prompt-free + ByteTrack |
| `engine/embed.py` | `crop()`, `Embedder`: DINOv3 fingerprints |
| `engine/identity.py` | `IdentityResolver`: track id -> lasting object id |
| `engine/where.py` | `nearby()`, `describe()`: "on the desk, next to the laptop" |
| `engine/sighting.py` | `OpenSighting`: one stretch of visibility and where it came to rest |
| `engine/store.py` | `MemoryStore`: SQLite objects and sightings, snapshot files |
| `engine/ask.py` | `QueryMatcher`: question text -> remembered object name |
| `engine/pipeline.py` | `Engine`, `Seen`: wire everything per frame; answer questions |
| `engine/run.py` | CLI: build a memory from a video |
| `engine/answer.py` | CLI: ask a built memory a question |
| `eval/__init__.py` | Package marker |
| `eval/speed.py` | Phase 0: per-stage timing, fps, GPU memory |
| `eval/score.py` | Score answers against a ground-truth file |
| `eval/truth/README.md` | Ground-truth file format and the before-you-run rule |
| `tests/...` | One test file per engine module |
| `docs/design/phase0-results.md` | Phase 0 measurements (written in Task 5) |
| `docs/design/phase1-results.md` | Phase 1 measurements (written in Task 12) |

---

### Task 1: Project setup and `VideoFrames`

**Files:**
- Create: `requirements-live.txt`, `pytest.ini`, `engine/__init__.py`, `engine/frames.py`, `tests/conftest.py`, `tests/test_frames.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: `VideoFrames(path, fps=10)`, iterable of `(index, time_seconds, bgr_frame)`; raises `ValueError` for an unreadable file. Fixture `synthetic_video(tmp_path)` returning the path of a 3 s, 30 fps, 320x240 test video.

- [ ] **Step 1: Create the virtual environment and dependency files**

`requirements-live.txt`:
```
# Install PyTorch first, as for version 1 (CPU build, or the CUDA build matching the GPU driver).
ultralytics>=8.4.0
lap>=0.5.12
transformers>=4.56
sentence-transformers>=3.0
opencv-python
numpy
pytest
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    models: needs downloaded model weights (slow); skip with -m "not models"
```

Append to `.gitignore`:
```
# live engine output and downloaded weights
memory_live/
*.pt
```

`engine/__init__.py`: empty file.

Run (PowerShell, repo root):
```
py -3.11 -m venv venv-live
venv-live\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-live.txt
```
Expected: installs without errors. (On the GPU machine use `--index-url https://download.pytorch.org/whl/cu121` instead.)

- [ ] **Step 2: Write the failing tests**

`tests/conftest.py`:
```python
import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """3 s of 30 fps, 320x240 video: a white square sliding right on black."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(90):
        frame = np.zeros((240, 320, 3), np.uint8)
        x = 10 + i * 2
        frame[100:140, x:x + 40] = 255
        writer.write(frame)
    writer.release()
    return path
```

`tests/test_frames.py`:
```python
import pytest

from engine.frames import VideoFrames


def test_samples_at_ten_fps(synthetic_video):
    frames = list(VideoFrames(synthetic_video))
    assert 29 <= len(frames) <= 31
    indexes = [i for i, _, _ in frames]
    assert indexes == list(range(len(frames)))
    times = [t for _, t, _ in frames]
    assert times[0] == 0.0
    assert all(0.09 <= b - a <= 0.11 for a, b in zip(times, times[1:]))
    assert frames[0][2].shape == (240, 320, 3)


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read video"):
        VideoFrames(tmp_path / "nope.mp4")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.frames'`

- [ ] **Step 4: Implement `engine/frames.py`**

```python
"""Read a video at a fixed sampling rate, one frame at a time.

    for index, t, frame in VideoFrames("videos/desk.mp4"):
        ...
"""
import cv2

FPS = 10  # same rate as version 1: enough to follow a hand moving an object


class VideoFrames:
    def __init__(self, path, fps=FPS):
        self.cap = cv2.VideoCapture(str(path))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened() or self.source_fps <= 0:
            raise ValueError(f"cannot read video: {path}")
        self.step = self.source_fps / fps

    def __iter__(self):
        read, index, next_pick = 0, 0, 0.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            if read >= next_pick:
                yield index, round(read / self.source_fps, 2), frame
                index += 1
                next_pick += self.step
            read += 1
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_frames.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add requirements-live.txt pytest.ini .gitignore engine/__init__.py engine/frames.py tests/conftest.py tests/test_frames.py
git commit -m "feat: live engine package with a fixed-rate video reader

Phase 1 of the live design reads recorded video at 10 fps, the rate
version 1 used. An unreadable file raises a clear error instead of
dividing by a zero frame rate."
```

---

### Task 2: `Detector` (YOLOE prompt-free + ByteTrack)

**Files:**
- Create: `engine/detect.py`, `tests/test_detect.py`

**Interfaces:**
- Produces: dataclass `Detection(track_id: int, label: str, conf: float, box: list)` with `box = [x1, y1, x2, y2]` ints in frame pixels; `Detector(weights=WEIGHTS, imgsz=640)` with `detect(bgr_frame) -> list[Detection]`. Tracker state persists across calls on one `Detector`; use a new `Detector` per video.

- [ ] **Step 1: Write the failing tests**

`tests/test_detect.py`:
```python
import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from engine.detect import Detector


@pytest.fixture(scope="module")
def detector():
    return Detector()


@pytest.mark.models
def test_finds_objects_in_a_street_photo(detector):
    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    detections = detector.detect(frame)
    assert len(detections) >= 3
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert 0.0 < d.conf <= 1.0
        assert d.label
    assert len({d.track_id for d in detections}) == len(detections)


@pytest.mark.models
def test_blank_frame_gives_no_detections(detector):
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.detect'`

- [ ] **Step 3: Implement `engine/detect.py`**

```python
"""Find every object in a frame and keep a track id on it across frames.

YOLOE in prompt-free mode needs no list of classes: it names objects from a
built-in vocabulary of several thousand. ByteTrack (built into ultralytics)
links boxes between frames.

    detector = Detector()
    for d in detector.detect(frame):
        print(d.track_id, d.label, d.conf, d.box)
"""
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "yoloe-26s-seg-pf.pt"  # small model; phase 0 decides if a larger one fits
IMGSZ = 640       # detector input size; phase 0 also tries 480
MIN_CONF = 0.25   # below this, prompt-free labels are mostly guesses


@dataclass
class Detection:
    track_id: int
    label: str
    conf: float
    box: list  # x1, y1, x2, y2 in frame pixels


class Detector:
    def __init__(self, weights=WEIGHTS, imgsz=IMGSZ):
        self.model = YOLOE(str(weights))
        self.imgsz = imgsz

    def detect(self, frame):
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  imgsz=self.imgsz, conf=MIN_CONF, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        return [Detection(int(i), result.names[int(c)], float(p), [int(v) for v in b])
                for i, c, p, b in zip(boxes.id.tolist(), boxes.cls.tolist(),
                                      boxes.conf.tolist(), boxes.xyxy.tolist())]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_detect.py -v`
Expected: 2 passed (the first run downloads `yoloe-26s-seg-pf.pt` into `models/`). If ultralytics saves the weights elsewhere, move the file into `models/` and rerun.

- [ ] **Step 5: Commit**

```bash
git add engine/detect.py tests/test_detect.py
git commit -m "feat: detect and track every object with YOLOE prompt-free

Removes the per-object registration step: the detector names objects
from its own vocabulary and ByteTrack keeps a track id on each."
```

---

### Task 3: `crop()` and `Embedder` (DINOv3 fingerprints)

**Files:**
- Create: `engine/embed.py`, `tests/test_embed.py`

**Interfaces:**
- Produces: `crop(frame, box, pad=0.1)` -> BGR array or `None` when either side is under 16 px; `Embedder(model=MODEL)` with `embed(list_of_bgr_crops) -> np.ndarray` of shape `(n, dim)`, each row unit length, and `dim` attribute.

- [ ] **Step 1: Accept the DINOv3 license (one time, per machine)**

Open https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m, sign in, accept the terms. Then:
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
Paste a read token from https://huggingface.co/settings/tokens. Expected: "Login successful".

- [ ] **Step 2: Write the failing tests**

`tests/test_embed.py`:
```python
import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from engine.embed import Embedder, crop


def test_crop_adds_margin_inside_the_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop(frame, [50, 20, 150, 80]).shape == (72, 120, 3)


def test_crop_clamps_at_the_frame_edge():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop(frame, [0, 0, 100, 100]).shape == (100, 110, 3)


def test_tiny_box_has_no_crop():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop(frame, [10, 10, 20, 60]) is None


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.mark.models
def test_same_object_scores_higher_than_noise(embedder):
    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    a = frame[230:740, 50:800]
    shifted = frame[235:745, 58:808]
    noise = np.random.default_rng(0).integers(0, 255, a.shape, dtype=np.uint8)
    fa, fs, fn = embedder.embed([a, shifted, noise])
    assert abs(np.linalg.norm(fa) - 1) < 1e-4
    assert fa @ fs > fa @ fn
    assert fa @ fs > 0.8


@pytest.mark.models
def test_no_crops_gives_empty_array(embedder):
    assert embedder.embed([]).shape == (0, embedder.dim)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.embed'`

- [ ] **Step 4: Implement `engine/embed.py`**

```python
"""Identity fingerprints: a DINOv3 vector per object crop.

DINOv3 is trained to tell individual objects apart, not just categories, which
is what version 1's CLIP check could not do (the false sunglasses episode).

    embedder = Embedder()
    vectors = embedder.embed([crop(frame, d.box) for d in detections])
"""
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"  # smallest DINOv3; gated, accept the license first
MIN_SIDE = 16  # px; a smaller crop is blur, not identity
PAD = 0.1      # grow the box by 10% so the object's outline is in the crop


def crop(frame, box, pad=PAD):
    """Cut a box out of a frame with a margin, clamped to the frame; None if too small."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    dx, dy = int(w * pad), int(h * pad)
    height, width = frame.shape[:2]
    return frame[max(0, y1 - dy):min(height, y2 + dy), max(0, x1 - dx):min(width, x2 + dx)]


class Embedder:
    def __init__(self, model=MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def embed(self, crops):
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        images = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vectors = self.model(**inputs).pooler_output.float()
        return torch.nn.functional.normalize(vectors, dim=1).cpu().numpy()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_embed.py -v`
Expected: 5 passed. If loading fails with a 401/403 "gated repo" error, Step 1 was not completed on this machine.

- [ ] **Step 6: Commit**

```bash
git add engine/embed.py tests/test_embed.py
git commit -m "feat: DINOv3 identity fingerprints for object crops

Instance-level features are the fix the handbook names for version 1's
category-level CLIP check. Crops under 16 px are skipped rather than
fingerprinted from blur."
```

---

### Task 4: Phase 0 speed test

**Files:**
- Create: `eval/__init__.py` (empty), `eval/speed.py`, `tests/test_speed.py`

**Interfaces:**
- Consumes: `VideoFrames`, `Detector`, `Embedder`, `crop`.
- Produces: `summarize(milliseconds)` -> `{"median_ms", "p90_ms", "fps"}`; CLI `python -m eval.speed VIDEO [--frames N] [--weights PATH] [--imgsz N]`.

- [ ] **Step 1: Write the failing test**

`tests/test_speed.py`:
```python
from eval.speed import summarize


def test_summarize_reports_median_p90_and_fps():
    s = summarize([10.0] * 9 + [20.0])
    assert s["median_ms"] == 10.0
    assert s["p90_ms"] == 10.0
    assert round(s["fps"], 1) == 90.9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_speed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.speed'`

- [ ] **Step 3: Implement `eval/speed.py`**

```python
"""Phase 0: can detection, tracking and fingerprints keep up live on this machine?

Times each stage per frame. The fingerprint stage embeds every detection on
every frame, the worst case; the engine itself only embeds new tracks.

    python -m eval.speed videos/desk.mp4 [--frames 300] [--weights models/yoloe-26m-seg-pf.pt] [--imgsz 480]
"""
import argparse
import statistics
import time

import torch

from engine.detect import IMGSZ, WEIGHTS, Detector
from engine.embed import Embedder, crop
from engine.frames import VideoFrames

WARMUP = 10  # first frames include model compilation and caching; not counted


def summarize(milliseconds):
    ms = sorted(milliseconds)
    return {"median_ms": statistics.median(ms),
            "p90_ms": ms[int(0.9 * (len(ms) - 1))],
            "fps": 1000 / statistics.mean(ms)}


def measure(video, frames, weights, imgsz):
    detector, embedder = Detector(weights, imgsz), Embedder()
    stages = {"detect+track": [], "fingerprint (all boxes)": [], "total": []}
    for index, _, frame in VideoFrames(video):
        if index >= frames + WARMUP:
            break
        start = time.perf_counter()
        detections = detector.detect(frame)
        mid = time.perf_counter()
        embedder.embed([c for c in (crop(frame, d.box) for d in detections) if c is not None])
        end = time.perf_counter()
        if index >= WARMUP:
            stages["detect+track"].append((mid - start) * 1000)
            stages["fingerprint (all boxes)"].append((end - mid) * 1000)
            stages["total"].append((end - start) * 1000)
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    args = parser.parse_args()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    stages = measure(args.video, args.frames, args.weights, args.imgsz)
    print(f"device {device} | weights {args.weights} | imgsz {args.imgsz} | frames {len(stages['total'])}")
    for name, ms in stages.items():
        s = summarize(ms)
        print(f"  {name:24} median {s['median_ms']:6.1f} ms  p90 {s['p90_ms']:6.1f} ms  ~{s['fps']:5.1f} fps")
    if torch.cuda.is_available():
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_speed.py -v`
Expected: 1 passed

- [ ] **Step 5: Smoke-run on the CPU laptop**

Record a 1-minute phone video of a desk with 5-10 objects, copy it to `videos/desk.mp4` (gitignored).
Run: `python -m eval.speed videos/desk.mp4 --frames 30`
Expected: three timing lines and `device CPU`. CPU numbers are not the phase 0 result; they only prove the script runs.

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/speed.py tests/test_speed.py
git commit -m "feat: phase 0 speed test for detection, tracking and fingerprints

Measures per-stage time and GPU memory so the live design can be checked
against the 3050 before more of it is built."
```

---

### Task 5: Run phase 0 on the RTX 3050 (manual, needs Tailscale)

**Files:**
- Create: `docs/design/phase0-results.md`

**Interfaces:**
- Consumes: `eval/speed.py` from Task 4, pushed to GitHub.

- [ ] **Step 1: Connect** (prerequisite: both laptops on the same Tailscale network, OpenSSH Server running on Abhinav's laptop)

From Sujan's laptop: `ssh <abhinav-windows-user>@<abhinav-tailscale-machine-name>`
Expected: a PowerShell prompt on Abhinav's machine. Ask Abhinav before every heavy run.

- [ ] **Step 2: Set up the repo on the GPU machine**

```
git clone https://github.com/metro-dev26/glasses-memory glasses-memory-live
cd glasses-memory-live
py -3.11 -m venv venv-live
venv-live\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-live.txt
huggingface-cli login
```
Then from Sujan's laptop: `scp videos\desk.mp4 <user>@<machine>:glasses-memory-live/videos/desk.mp4` (create `videos\` on the remote first).

- [ ] **Step 3: Measure four configurations**

```
python -m eval.speed videos/desk.mp4 --weights models/yoloe-26s-seg-pf.pt --imgsz 640
python -m eval.speed videos/desk.mp4 --weights models/yoloe-26s-seg-pf.pt --imgsz 480
python -m eval.speed videos/desk.mp4 --weights models/yoloe-26m-seg-pf.pt --imgsz 640
python -m eval.speed videos/desk.mp4 --weights models/yoloe-26n-seg-pf.pt --imgsz 640
```
Expected: each prints `device NVIDIA GeForce RTX 3050 ...`, timings, peak memory under 4096 MB.

- [ ] **Step 4: Record the results and the decision**

`docs/design/phase0-results.md`: a table of the four runs (weights, imgsz, detect+track median, fingerprint median, total fps, peak MB), the date, the GPU name, and one decision line. Rule from the spec: **total about 10 fps or better** means approach A holds with that configuration; set `WEIGHTS`/`IMGSZ` in `engine/detect.py` to it. If no configuration reaches it, record that and stop: the design needs revisiting before phase 2, not a quieter threshold.

- [ ] **Step 5: Commit**

```bash
git add docs/design/phase0-results.md engine/detect.py
git commit -m "docs: phase 0 speed results on the RTX 3050

Records which detector size and input resolution keep the pipeline live,
and sets that configuration as the default."
```

---

### Task 6: `IdentityResolver`

**Files:**
- Create: `engine/identity.py`, `tests/test_identity.py`

**Interfaces:**
- Produces: `IdentityResolver(match_sim=MATCH_SIM, margin=MATCH_MARGIN)` with `needs_embedding(track_id) -> bool`, `observe(track_id, unit_vector) -> (object_id or None, reidentified: bool)`, `object_for(track_id) -> object_id or None`, `end_frame(active_track_ids)`. Object ids are ints from 1. Module constants `MIN_TRACK_FRAMES`, `REFRESH_FRAMES`, `MATCH_SIM`.

- [ ] **Step 1: Write the failing tests**

`tests/test_identity.py`:
```python
import numpy as np

from engine.identity import MIN_TRACK_FRAMES, REFRESH_FRAMES, IdentityResolver


def unit(seed):
    v = np.random.default_rng(seed).normal(size=64)
    return v / np.linalg.norm(v)


def near(v, seed, noise=0.05):
    w = v + np.random.default_rng(seed).normal(size=v.size) * noise
    return w / np.linalg.norm(w)


def show(ids, track_id, vector, frames=MIN_TRACK_FRAMES, active=None):
    """Feed one track for `frames` frames, with noise unique to the track; returns the last observe() result."""
    result = None
    for i in range(frames):
        ids.end_frame(active if active is not None else {track_id})
        result = ids.observe(track_id, near(vector, track_id * 100 + i))
    return result


def test_undecided_until_enough_frames():
    ids = IdentityResolver()
    assert show(ids, 1, unit(0), frames=MIN_TRACK_FRAMES - 1) == (None, False)
    assert ids.object_for(1) is None


def test_new_track_becomes_object_one():
    ids = IdentityResolver()
    assert show(ids, 1, unit(0)) == (1, False)
    assert ids.object_for(1) == 1


def test_returning_object_is_reidentified():
    ids = IdentityResolver()
    remote = unit(0)
    show(ids, 1, remote)
    ids.end_frame(set())
    assert show(ids, 2, remote) == (1, True)


def test_different_object_gets_a_new_id():
    ids = IdentityResolver()
    show(ids, 1, unit(0))
    ids.end_frame(set())
    assert show(ids, 2, unit(1)) == (2, False)


def test_ambiguous_match_creates_a_new_object():
    ids = IdentityResolver()
    a, b = unit(0), unit(1)
    show(ids, 1, a)
    ids.end_frame(set())
    show(ids, 2, b)
    ids.end_frame(set())
    halfway = (a + b) / np.linalg.norm(a + b)
    object_id, reidentified = show(ids, 3, halfway)
    assert object_id == 3 and not reidentified


def test_lookalike_in_view_at_the_same_time_is_not_merged():
    ids = IdentityResolver()
    pen = unit(0)
    show(ids, 1, pen)
    result = None
    for i in range(MIN_TRACK_FRAMES):
        ids.end_frame({1, 2})
        result = ids.observe(2, near(pen, 200 + i))
    assert result == (2, False)
    assert ids.object_for(1) == 1


def test_flicker_track_is_forgotten():
    ids = IdentityResolver()
    show(ids, 1, unit(0), frames=2)
    ids.end_frame(set())
    assert show(ids, 1, unit(0), frames=MIN_TRACK_FRAMES - 1) == (None, False)


def test_identified_track_is_refingerprinted_periodically():
    ids = IdentityResolver()
    show(ids, 1, unit(0))
    assert not ids.needs_embedding(1)
    for _ in range(REFRESH_FRAMES):
        ids.end_frame({1})
    assert ids.needs_embedding(1)
    assert ids.observe(1, unit(0)) == (1, False)
    assert not ids.needs_embedding(1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_identity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.identity'`

- [ ] **Step 3: Implement `engine/identity.py`**

```python
"""Give every tracked object a lasting identity, so it is recognised when it comes back.

ByteTrack's track ids last only while an object stays in view. Each new track
is matched by DINOv3 fingerprint against every object seen before.

    ids = IdentityResolver()
    ids.end_frame(active_track_ids)
    if ids.needs_embedding(track_id):
        object_id, reidentified = ids.observe(track_id, fingerprint)
"""
import numpy as np

MIN_TRACK_FRAMES = 5  # fingerprints collected before deciding; shorter tracks are flicker
MATCH_SIM = 0.6       # starting value; calibrated on calibration videos only (Task 12)
MATCH_MARGIN = 0.05   # the best match must beat the runner-up by this much, or look-alikes merge
NEW_VIEW_SIM = 0.85   # a fingerprint less similar than this to every stored view is a new viewpoint
GALLERY_SIZE = 8      # views kept per object; the first view is never dropped
REFRESH_FRAMES = 15   # an identified track is fingerprinted again this often, to collect new views


def _unit(v):
    return v / np.linalg.norm(v)


class IdentityResolver:
    def __init__(self, match_sim=MATCH_SIM, margin=MATCH_MARGIN):
        self.match_sim = match_sim
        self.margin = margin
        self.galleries = {}      # object id -> list of unit vectors
        self.bound = {}          # track id -> object id
        self.pending = {}        # track id -> fingerprints collected while undecided
        self.since_refresh = {}  # track id -> frames since its last fingerprint
        self.next_id = 1

    def object_for(self, track_id):
        return self.bound.get(track_id)

    def needs_embedding(self, track_id):
        if track_id not in self.bound:
            return True
        return self.since_refresh[track_id] >= REFRESH_FRAMES

    def observe(self, track_id, fingerprint):
        """Feed one fingerprint. Returns (object_id, reidentified); object_id is None while undecided."""
        if track_id in self.bound:
            object_id = self.bound[track_id]
            self._add_view(object_id, fingerprint)
            self.since_refresh[track_id] = 0
            return object_id, False
        views = self.pending.setdefault(track_id, [])
        views.append(fingerprint)
        if len(views) < MIN_TRACK_FRAMES:
            return None, False
        del self.pending[track_id]
        object_id, reidentified = self._resolve(_unit(np.mean(views, axis=0)))
        for view in views:
            self._add_view(object_id, view)
        self.bound[track_id] = object_id
        self.since_refresh[track_id] = 0
        return object_id, reidentified

    def end_frame(self, active_track_ids):
        """Forget tracks that left the view and count frames towards the next fingerprint."""
        for track_id in list(self.bound):
            if track_id in active_track_ids:
                self.since_refresh[track_id] += 1
            else:
                del self.bound[track_id], self.since_refresh[track_id]
        for track_id in list(self.pending):
            if track_id not in active_track_ids:
                del self.pending[track_id]

    def _resolve(self, query):
        in_view = set(self.bound.values())
        scores = sorted(((max(np.array(g) @ query), object_id)
                         for object_id, g in self.galleries.items() if object_id not in in_view),
                        reverse=True)
        if scores:
            best, object_id = scores[0]
            runner_up = scores[1][0] if len(scores) > 1 else -1.0
            if best >= self.match_sim and best - runner_up >= self.margin:
                return object_id, True
        object_id, self.next_id = self.next_id, self.next_id + 1
        return object_id, False

    def _add_view(self, object_id, fingerprint):
        gallery = self.galleries.setdefault(object_id, [])
        if gallery and max(np.array(gallery) @ fingerprint) >= NEW_VIEW_SIM:
            return
        gallery.append(fingerprint)
        if len(gallery) > GALLERY_SIZE:
            gallery.pop(1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_identity.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add engine/identity.py tests/test_identity.py
git commit -m "feat: lasting object identities across tracks

Matches each new track to earlier objects by fingerprint, with a margin
over the runner-up so look-alikes are not merged, and never matches an
object that is already in view under another track."
```

---

### Task 7: Where it was left (`where.py`, `sighting.py`)

**Files:**
- Create: `engine/where.py`, `engine/sighting.py`, `tests/test_where.py`, `tests/test_sighting.py`

**Interfaces:**
- Produces: `nearby(box, others)` with `others` a list of `(label, box)` -> `(surface_label or None, [up to 2 neighbour labels])`; `describe(surface, neighbours)` -> str; `OpenSighting(object_id)` with `add(t, box, where_text, snapshot_jpeg_bytes)`, attributes `start`, `end`, and `resting()` -> `(t, box, where_text, snapshot)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_where.py`:
```python
from engine.where import describe, nearby


def test_surface_is_the_smallest_big_box_holding_the_base():
    remote = [100, 100, 140, 120]
    others = [("room", [0, 0, 1000, 1000]), ("desk", [50, 110, 400, 300]), ("mug", [150, 95, 175, 120])]
    surface, neighbours = nearby(remote, others)
    assert surface == "desk"
    assert neighbours == ["mug"]


def test_far_objects_are_not_neighbours():
    surface, neighbours = nearby([100, 100, 140, 120], [("lamp", [900, 900, 950, 950])])
    assert (surface, neighbours) == (None, [])


def test_describe():
    assert describe("desk", ["laptop", "mug"]) == "on the desk, next to the laptop and the mug"
    assert describe(None, ["laptop"]) == "next to the laptop"
    assert describe(None, []) == "with nothing recognisable nearby"
```

`tests/test_sighting.py`:
```python
from engine.sighting import STILL_FRAMES, OpenSighting


def test_rest_is_the_last_still_moment_before_it_is_picked_up():
    s = OpenSighting(1)
    t = 0.0
    for x in range(0, 100, 20):  # moving in a hand
        s.add(t, [x, 0, x + 30, 30], "moving", b"m")
        t += 0.1
    for _ in range(STILL_FRAMES + 2):  # put down on the desk
        s.add(t, [200, 0, 230, 30], "on the desk", b"still")
        t += 0.1
    last_still = t - 0.1
    for x in range(300, 500, 40):  # picked up again and carried out of view
        s.add(t, [x, 0, x + 30, 30], "moving", b"m")
        t += 0.1
    rest_t, box, where, snapshot = s.resting()
    assert abs(rest_t - last_still) < 1e-9
    assert (box, where, snapshot) == ([200, 0, 230, 30], "on the desk", b"still")
    assert s.start == 0.0 and abs(s.end - (t - 0.1)) < 1e-9


def test_never_still_falls_back_to_the_last_sample():
    s = OpenSighting(1)
    for i in range(5):
        s.add(i * 0.1, [i * 50, 0, i * 50 + 30, 30], f"spot {i}", b"x")
    assert s.resting()[2] == "spot 4"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_where.py tests/test_sighting.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `engine/where.py`**

```python
"""Say where an object is in words, from the other objects in the same frame.

No 3D map: the surface is the smallest much larger box that holds the object's
base ("on the desk"), and neighbours are nearby boxes ("next to the laptop").

    surface, neighbours = nearby(remote.box, [(d.label, d.box) for d in others])
    describe(surface, neighbours)   # "on the desk, next to the laptop"
"""
from math import hypot

NEAR = 1.5     # neighbours within 1.5 object-diagonals, centre to centre, count as "next to"
SURFACE = 2.0  # a box at least twice the object's area that holds its base is what it sits on
MAX_NEIGHBOURS = 2


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def nearby(box, others):
    base_x, base_y = (box[0] + box[2]) / 2, box[3]
    reach = NEAR * hypot(box[2] - box[0], box[3] - box[1])
    cx, cy = _centre(box)
    surfaces, neighbours = [], []
    for label, other in others:
        holds_base = other[0] <= base_x <= other[2] and other[1] <= base_y <= other[3]
        if holds_base and _area(other) >= SURFACE * _area(box):
            surfaces.append((_area(other), label))
            continue
        ox, oy = _centre(other)
        distance = hypot(ox - cx, oy - cy)
        if distance <= reach:
            neighbours.append((distance, label))
    surface = min(surfaces)[1] if surfaces else None
    labels = []
    for _, label in sorted(neighbours):
        if label not in labels:
            labels.append(label)
    return surface, labels[:MAX_NEIGHBOURS]


def describe(surface, neighbours):
    parts = []
    if surface:
        parts.append(f"on the {surface}")
    if neighbours:
        parts.append("next to the " + " and the ".join(neighbours))
    return ", ".join(parts) or "with nothing recognisable nearby"
```

- [ ] **Step 4: Implement `engine/sighting.py`**

```python
"""One stretch of time an object was in view, and the moment it came to rest.

The resting moment is the last frame of a run where the box stopped moving:
where the object was put down, not where it looked biggest (often in a hand).

    s = OpenSighting(object_id)
    s.add(t, box, "on the desk", jpeg_bytes)   # every frame it is seen
    t, box, where, snapshot = s.resting()
"""
from math import hypot

STILL_PX = 8      # centre moving less than this between frames (at 10 fps) counts as still
STILL_FRAMES = 5  # 0.5 s of stillness means put down, not a pause mid-move


def _shift(a, b):
    return hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


class OpenSighting:
    def __init__(self, object_id):
        self.object_id = object_id
        self.start = self.end = None
        self.prev_box = None
        self.still_run = 0
        self.last = self.rest = None

    def add(self, t, box, where, snapshot):
        if self.start is None:
            self.start = t
        self.end = t
        still = self.prev_box is not None and _shift(self.prev_box, box) <= STILL_PX
        self.still_run = self.still_run + 1 if still else 0
        self.prev_box = box
        self.last = (t, box, where, snapshot)
        if self.still_run >= STILL_FRAMES:
            self.rest = self.last

    def resting(self):
        return self.rest or self.last
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_where.py tests/test_sighting.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add engine/where.py engine/sighting.py tests/test_where.py tests/test_sighting.py
git commit -m "feat: where an object was left, in words and as a moment

The resting moment is the last still frame, which fixes the handbook's
snapshot-in-hand weakness; the description comes from the surface and
neighbours already detected in the same frame."
```

---

### Task 8: `MemoryStore`

**Files:**
- Create: `engine/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `OpenSighting` (Task 7).
- Produces: `MemoryStore(root, fresh=True)`; `add_object(object_id, label)` (idempotent), `rename(object_id, name)`, `save_sighting(open_sighting)`, `objects()` -> list of dicts `{"id", "name", "label", "last_seen", "where"}` newest first, `sightings(object_id)` -> list of dicts `{"start", "end", "rest_time", "rest_box", "where", "snapshot"}` newest first, `forget_all()`. Snapshots are written to `root/snapshots/<sighting id>.jpg`; `snapshot` is that path relative to `root`.

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:
```python
from engine.sighting import OpenSighting
from engine.store import MemoryStore


def sighting(object_id, t0, where):
    s = OpenSighting(object_id)
    for i in range(3):
        s.add(t0 + i * 0.1, [10, 10, 50, 50], where, b"\xff\xd8jpeg")
    return s


def test_objects_and_sightings_newest_first(tmp_path):
    store = MemoryStore(tmp_path)
    store.add_object(1, "remote")
    store.add_object(1, "phone")
    store.save_sighting(sighting(1, 0.0, "on the desk"))
    store.save_sighting(sighting(1, 30.0, "on the bed"))
    [remote] = store.objects()
    assert remote["name"] == "remote" and remote["label"] == "remote"
    assert remote["last_seen"] == 30.2 and remote["where"] == "on the bed"
    rows = store.sightings(1)
    assert [r["where"] for r in rows] == ["on the bed", "on the desk"]
    assert (tmp_path / rows[0]["snapshot"]).read_bytes() == b"\xff\xd8jpeg"


def test_rename_and_reopen(tmp_path):
    store = MemoryStore(tmp_path)
    store.add_object(1, "cup")
    store.save_sighting(sighting(1, 0.0, "on the desk"))
    store.rename(1, "my blue mug")
    assert MemoryStore(tmp_path, fresh=False).objects()[0]["name"] == "my blue mug"


def test_object_never_seen_still_listed(tmp_path):
    store = MemoryStore(tmp_path)
    store.add_object(2, "pen")
    assert store.objects() == [{"id": 2, "name": "pen", "label": "pen", "last_seen": None, "where": None}]
    assert store.sightings(2) == []


def test_forget_all(tmp_path):
    store = MemoryStore(tmp_path)
    store.add_object(1, "remote")
    store.save_sighting(sighting(1, 0.0, "on the desk"))
    store.forget_all()
    assert store.objects() == []
    assert list((tmp_path / "snapshots").iterdir()) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.store'`

- [ ] **Step 3: Implement `engine/store.py`**

```python
"""The object memory on disk: SQLite for facts, JPEG files for snapshots.

No video is kept: one snapshot per sighting, at the moment the object came to rest.

    store = MemoryStore("memory_live")            # fresh=True wipes an old memory
    store.add_object(object_id, "remote")
    store.save_sighting(open_sighting)
    store.objects(); store.sightings(object_id)
"""
import json
import shutil
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, label TEXT, name TEXT);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, object_id INTEGER, start REAL, end REAL,
    rest_time REAL, rest_box TEXT, where_text TEXT, snapshot TEXT);
"""


class MemoryStore:
    def __init__(self, root, fresh=True):
        self.root = Path(root)
        if fresh:
            shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "memory.db")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add_object(self, object_id, label):
        self.db.execute("INSERT OR IGNORE INTO objects VALUES (?, ?, ?)", (object_id, label, label))
        self.db.commit()

    def rename(self, object_id, name):
        self.db.execute("UPDATE objects SET name = ? WHERE id = ?", (name, object_id))
        self.db.commit()

    def save_sighting(self, sighting):
        rest_time, box, where, snapshot = sighting.resting()
        cursor = self.db.execute(
            "INSERT INTO sightings (object_id, start, end, rest_time, rest_box, where_text)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (sighting.object_id, sighting.start, round(sighting.end, 2), rest_time, json.dumps(box), where))
        path = f"snapshots/{cursor.lastrowid}.jpg"
        (self.root / path).write_bytes(snapshot)
        self.db.execute("UPDATE sightings SET snapshot = ? WHERE id = ?", (path, cursor.lastrowid))
        self.db.commit()

    def objects(self):
        rows = self.db.execute(
            "SELECT o.id, o.name, o.label, s.end AS last_seen, s.where_text AS 'where' FROM objects o"
            " LEFT JOIN sightings s ON s.id = (SELECT id FROM sightings WHERE object_id = o.id"
            " ORDER BY end DESC LIMIT 1) ORDER BY s.end IS NULL, s.end DESC, o.id")
        return [dict(r) for r in rows]

    def sightings(self, object_id):
        rows = self.db.execute(
            "SELECT start, end, rest_time, rest_box, where_text AS 'where', snapshot FROM sightings"
            " WHERE object_id = ? ORDER BY end DESC", (object_id,))
        return [dict(r, rest_box=json.loads(r["rest_box"])) for r in rows]

    def forget_all(self):
        self.db.executescript("DELETE FROM sightings; DELETE FROM objects;")
        for f in (self.root / "snapshots").iterdir():
            f.unlink()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add engine/store.py tests/test_store.py
git commit -m "feat: SQLite object memory with one snapshot per sighting

Keeps facts and a resting snapshot instead of video, as the design's
privacy section requires, and can be wiped in one call."
```

---

### Task 9: `QueryMatcher`

**Files:**
- Create: `engine/ask.py`, `tests/test_ask.py`

**Interfaces:**
- Produces: `QueryMatcher(model=MODEL)` with `match(query, names)` -> `(best_name or None, [up to 3 closest names])`; `(None, [])` when `names` is empty.

- [ ] **Step 1: Write the failing tests**

`tests/test_ask.py`:
```python
import pytest

from engine.ask import QueryMatcher, strip

NAMES = ["sunglasses", "remote", "coffee mug", "stapler"]


def test_strip_removes_question_words():
    assert strip("Where did I leave my sunglasses?") == "sunglasses"


@pytest.fixture(scope="module")
def matcher():
    return QueryMatcher()


@pytest.mark.models
def test_finds_the_named_object(matcher):
    assert matcher.match("where are my sunglasses?", NAMES)[0] == "sunglasses"


@pytest.mark.models
def test_finds_by_related_words(matcher):
    assert matcher.match("where did I put my glasses", NAMES)[0] == "sunglasses"
    assert matcher.match("where's my mug", NAMES)[0] == "coffee mug"


@pytest.mark.models
def test_unknown_object_is_not_found_but_closest_names_come_back(matcher):
    best, closest = matcher.match("where's my bicycle?", NAMES)
    assert best is None
    assert len(closest) == 3 and set(closest) <= set(NAMES)


@pytest.mark.models
def test_empty_memory(matcher):
    assert matcher.match("where's my remote?", []) == (None, [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ask.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.ask'`

- [ ] **Step 3: Implement `engine/ask.py`**

```python
"""Match a question to a remembered object by meaning, not exact words.

Version 1 matched shared keywords; a sentence embedding also connects
"glasses" to "sunglasses" and "mug" to "coffee mug".

    best, closest = QueryMatcher().match("where's my mug?", ["coffee mug", "remote"])
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80 MB, fast on CPU
MIN_SIM = 0.5   # below this the closest name is a guess, so the answer is "not found"
CLOSEST = 3
STOP = set("where where's wheres is are am my the a an did i leave left put keep kept last see seen saw".split())


def strip(query):
    words = re.sub(r"[^a-z' ]", " ", query.lower()).split()
    return " ".join(w for w in words if w not in STOP)


class QueryMatcher:
    def __init__(self, model=MODEL):
        self.model = SentenceTransformer(model)

    def match(self, query, names):
        if not names:
            return None, []
        q = self.model.encode([strip(query) or query], normalize_embeddings=True)[0]
        sims = self.model.encode(names, normalize_embeddings=True) @ q
        order = np.argsort(-sims)
        best = names[order[0]] if sims[order[0]] >= MIN_SIM else None
        return best, [names[i] for i in order[:CLOSEST]]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ask.py -v`
Expected: 5 passed. If a related-word test fails, print the similarities for that query and report them; do not lower `MIN_SIM` to make it pass without recording why.

- [ ] **Step 5: Commit**

```bash
git add engine/ask.py tests/test_ask.py
git commit -m "feat: match questions to remembered objects by meaning

A sentence embedding replaces version 1's keyword overlap, with a
similarity floor so an unknown object gets 'not found', not a guess."
```

---

### Task 10: `Engine` and the command-line tools

**Files:**
- Create: `engine/pipeline.py`, `engine/run.py`, `engine/answer.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Detection` (Task 2), `crop` (Task 3), `IdentityResolver` (Task 6), `nearby`/`describe` (Task 7), `OpenSighting` (Task 7), `MemoryStore` (Task 8), `QueryMatcher` (Task 9).
- Produces: dataclass `Seen(detection, object_id, reidentified)`; `Engine(detector, embedder, store, resolver=None, matcher=None)` with `process(frame, t) -> list[Seen]`, `finish()`, `ask(query) -> dict` shaped like the protocol's `answer` message plus `rest_time`: `{"query", "found", "object_id", "text", "snapshot", "box", "rest_time", "sightings", "candidates"}`. CLIs `python -m engine.run VIDEO [--out DIR] [--match-sim X]` and `python -m engine.answer DIR "question"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
import numpy as np

from engine.detect import Detection
from engine.pipeline import Engine
from engine.store import MemoryStore

RED, BLUE = (0, 0, 255), (255, 0, 0)


class FakeDetector:
    """Scripted scene: a red remote visible, gone for 2 s, back under a new track; a blue mug throughout."""

    def detect(self, frame):
        self.t = getattr(self, "t", -1) + 1
        detections = [Detection(9, "mug", 0.9, [200, 100, 260, 160])]
        if self.t < 20:
            detections.append(Detection(1, "remote", 0.8, [100, 100, 160, 160]))
        elif self.t >= 40:
            detections.append(Detection(2, "remote", 0.8, [120, 100, 180, 160]))
        return detections


class FakeEmbedder:
    """Fingerprint = the crop's mean colour, so equal colours mean the same object."""

    def embed(self, crops):
        if not crops:
            return np.zeros((0, 3))
        vectors = np.array([c.reshape(-1, 3).mean(axis=0) + 1 for c in crops], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeMatcher:
    def match(self, query, names):
        hits = [n for n in names if n in query]
        return (hits[0] if hits else None), names[:3]


def frame_at(t):
    frame = np.zeros((240, 400, 3), np.uint8)
    frame[100:160, 200:260] = BLUE
    if t < 20:
        frame[100:160, 100:160] = RED
    elif t >= 40:
        frame[100:160, 120:180] = RED
    return frame


def run_scene(tmp_path):
    engine = Engine(FakeDetector(), FakeEmbedder(), MemoryStore(tmp_path), matcher=FakeMatcher())
    seen = [engine.process(frame_at(i), round(i * 0.1, 2)) for i in range(60)]
    engine.finish()
    return engine, seen


def test_remote_returning_on_a_new_track_keeps_its_identity(tmp_path):
    engine, seen = run_scene(tmp_path)
    remote_ids = {s.object_id for frame in seen for s in frame
                  if s.detection.label == "remote" and s.object_id is not None}
    assert len(remote_ids) == 1
    assert any(s.reidentified for frame in seen for s in frame)
    names = sorted(o["name"] for o in engine.store.objects())
    assert names == ["mug", "remote"]


def test_gap_splits_the_remote_into_two_sightings(tmp_path):
    engine, _ = run_scene(tmp_path)
    remote = next(o for o in engine.store.objects() if o["name"] == "remote")
    assert len(engine.store.sightings(remote["id"])) == 2


def test_answer_for_a_known_object(tmp_path):
    engine, _ = run_scene(tmp_path)
    answer = engine.ask("where's my remote?")
    assert answer["found"] and answer["rest_time"] >= 4.0
    assert "next to the mug" in answer["text"]
    assert (tmp_path / answer["snapshot"]).exists()
    assert len(answer["sightings"]) == 2


def test_unknown_object_and_empty_memory(tmp_path):
    engine, _ = run_scene(tmp_path)
    answer = engine.ask("where's my bicycle?")
    assert not answer["found"] and answer["candidates"]
    empty = Engine(FakeDetector(), FakeEmbedder(), MemoryStore(tmp_path / "empty"), matcher=FakeMatcher())
    answer = empty.ask("where's my remote?")
    assert not answer["found"] and answer["candidates"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.pipeline'`

- [ ] **Step 3: Implement `engine/pipeline.py`**

```python
"""One frame in, an updated object memory out.

    engine = Engine(Detector(), Embedder(), MemoryStore("memory_live"))
    for _, t, frame in VideoFrames(video):
        engine.process(frame, t)
    engine.finish()
    engine.ask("where's my remote?")
"""
from dataclasses import dataclass

import cv2

from engine.embed import crop
from engine.identity import IdentityResolver
from engine.sighting import OpenSighting
from engine.where import describe, nearby

CLOSE_AFTER_S = 1.0  # an object unseen this long has left; its sighting is saved
JPEG_QUALITY = 80


@dataclass
class Seen:
    detection: object
    object_id: object  # None while the track is still being identified
    reidentified: bool


def _clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Engine:
    def __init__(self, detector, embedder, store, resolver=None, matcher=None):
        self.detector, self.embedder, self.store = detector, embedder, store
        self.resolver = resolver or IdentityResolver()
        self._matcher = matcher
        self.open = {}       # object id -> OpenSighting
        self.last_seen = {}  # object id -> time

    def process(self, frame, t):
        detections = self.detector.detect(frame)
        self.resolver.end_frame({d.track_id for d in detections})
        reidentified = self._fingerprint(frame, detections)
        snapshot = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        seen = []
        for d in detections:
            object_id = self.resolver.object_for(d.track_id)
            if object_id is not None:
                self._record(object_id, d, detections, t, snapshot)
            seen.append(Seen(d, object_id, d.track_id in reidentified))
        self._close_stale(t)
        return seen

    def finish(self):
        for sighting in self.open.values():
            self.store.save_sighting(sighting)
        self.open.clear()

    def ask(self, query):
        objects = self.store.objects()
        names = [o["name"] for o in objects]
        best, closest = self.matcher.match(query, names)
        answer = {"query": query, "found": False, "object_id": None, "text": "",
                  "snapshot": None, "box": None, "rest_time": None, "sightings": [], "candidates": closest}
        same_name = [o for o in objects if o["name"] == best and o["last_seen"] is not None]
        if not same_name:
            answer["text"] = "I don't remember anything like that." if best is None else \
                f"I was shown the {best}, but never saw it clearly."
            return answer
        target = same_name[0]
        sightings = self.store.sightings(target["id"])
        last = sightings[0]
        text = f"Last seen at {_clock(last['rest_time'])} into the recording, {last['where']}."
        if len(same_name) > 1:
            text += f" I know {len(same_name)} of these; this is the most recently seen."
        answer.update(found=True, object_id=target["id"], text=text, snapshot=last["snapshot"],
                      box=last["rest_box"], rest_time=last["rest_time"], candidates=[],
                      sightings=[{"start": s["start"], "end": s["end"], "where": s["where"]} for s in sightings])
        return answer

    @property
    def matcher(self):
        if self._matcher is None:
            from engine.ask import QueryMatcher
            self._matcher = QueryMatcher()
        return self._matcher

    def _fingerprint(self, frame, detections):
        pairs = [(d, crop(frame, d.box)) for d in detections if self.resolver.needs_embedding(d.track_id)]
        pairs = [(d, c) for d, c in pairs if c is not None]
        reidentified = set()
        vectors = self.embedder.embed([c for _, c in pairs])
        for (d, _), vector in zip(pairs, vectors):
            if self.resolver.observe(d.track_id, vector)[1]:
                reidentified.add(d.track_id)
        return reidentified

    def _record(self, object_id, detection, detections, t, snapshot):
        self.store.add_object(object_id, detection.label)
        others = [(o.label, o.box) for o in detections if o is not detection]
        where = describe(*nearby(detection.box, others))
        self.open.setdefault(object_id, OpenSighting(object_id)).add(t, detection.box, where, snapshot)
        self.last_seen[object_id] = t

    def _close_stale(self, t):
        for object_id in [i for i in self.open if t - self.last_seen[i] > CLOSE_AFTER_S]:
            self.store.save_sighting(self.open.pop(object_id))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Implement the CLIs**

`engine/run.py`:
```python
"""Build an object memory from a recorded video.

    python -m engine.run videos/desk.mp4 [--out memory_live] [--match-sim 0.6]
"""
import argparse

from engine.detect import Detector
from engine.embed import Embedder
from engine.frames import VideoFrames
from engine.identity import MATCH_SIM, IdentityResolver
from engine.pipeline import Engine
from engine.store import MemoryStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default="memory_live")
    parser.add_argument("--match-sim", type=float, default=MATCH_SIM)
    args = parser.parse_args()
    engine = Engine(Detector(), Embedder(), MemoryStore(args.out),
                    resolver=IdentityResolver(match_sim=args.match_sim))
    for index, t, frame in VideoFrames(args.video):
        engine.process(frame, t)
        if index % 50 == 0:
            print(f"  {t:6.1f}s  {len(engine.store.objects())} objects", flush=True)
    engine.finish()
    for o in engine.store.objects():
        print(f"#{o['id']:<3} {o['name']:20} last seen {o['last_seen']}s, {o['where']}")


if __name__ == "__main__":
    main()
```

`engine/answer.py`:
```python
"""Ask a memory built by engine.run where something is.

    python -m engine.answer memory_live "where's my remote?"
"""
import sys

from engine.pipeline import Engine
from engine.store import MemoryStore


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    engine = Engine(None, None, MemoryStore(sys.argv[1], fresh=False))
    answer = engine.ask(sys.argv[2])
    print(answer["text"])
    if answer["found"]:
        print(f"snapshot: {sys.argv[1]}/{answer['snapshot']}")
    elif answer["candidates"]:
        print("I know: " + ", ".join(answer["candidates"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the engine on the desk video**

Run: `python -m engine.run videos/desk.mp4` then `python -m engine.answer memory_live "where's my <an object in the video>?"`
Expected: an object list, then an answer sentence and a snapshot path that opens to the right moment. On CPU this is slow (minutes); that is expected.

- [ ] **Step 7: Run the fast suite and commit**

Run: `pytest -m "not models" -v`
Expected: all pass.

```bash
git add engine/pipeline.py engine/run.py engine/answer.py tests/test_pipeline.py
git commit -m "feat: engine that builds an object memory from video and answers questions

Wires detection, identities, resting moments and the store per frame,
with components injected so the pipeline is tested without models."
```

---

### Task 11: Ground truth and scoring

**Files:**
- Create: `eval/score.py`, `eval/truth/README.md`, `tests/test_score.py`

**Interfaces:**
- Consumes: `Engine.ask` answer dicts (Task 10).
- Produces: `score(truth, ask)` where `truth` is the parsed truth JSON and `ask(query) -> answer dict`; returns `{"rows": [{"query", "verdict", "rest_time", "truth_rest", "coverage"}], "correct", "partial", "wrong"}`. CLI `python -m eval.score TRUTH.json MEMORY_DIR`.

- [ ] **Step 1: Write the truth-format README**

`eval/truth/README.md`:
````markdown
# Ground truth

One JSON file per test video, **written before the system is run on that video**.
Footage stays out of the repo; truth files are committed.

```json
{
  "video": "videos/test1.mp4",
  "split": "test",
  "objects": [
    {"query": "where's my remote?", "rest_time": 38.0, "visible": [[2.0, 9.5], [30.1, 38.5]]}
  ]
}
```

- `split`: `calibration` (thresholds may be tuned on it) or `test` (never tuned on).
- `rest_time`: second at which the object was last put down, from the recording.
- `visible`: every stretch the object is clearly in view, in seconds.

Verdicts: **correct** when the answer's rest time is within 2 s of `rest_time`;
**partial** when the right object was found (its sightings overlap `visible`)
but the rest time is off; **wrong** when nothing was found or the sightings do
not overlap `visible` (a different object). `coverage` is the share of
`visible` time covered by the answered object's sightings; low coverage means
the object was split into several identities.
````

- [ ] **Step 2: Write the failing tests**

`tests/test_score.py`:
```python
from eval.score import score

TRUTH = {"objects": [
    {"query": "remote", "rest_time": 38.0, "visible": [[30.0, 38.5]]},
    {"query": "stapler", "rest_time": 30.0, "visible": [[20.0, 30.5]]},
    {"query": "sunglasses", "rest_time": 24.0, "visible": [[10.0, 24.5]]},
    {"query": "keys", "rest_time": 5.0, "visible": [[1.0, 5.5]]},
]}

ANSWERS = {
    "remote": {"found": True, "rest_time": 38.4, "sightings": [{"start": 30.0, "end": 38.5}]},
    "stapler": {"found": True, "rest_time": 27.0, "sightings": [{"start": 20.0, "end": 30.6}]},
    "sunglasses": {"found": True, "rest_time": 54.0, "sightings": [{"start": 50.0, "end": 54.0}]},
    "keys": {"found": False, "rest_time": None, "sightings": []},
}


def test_verdicts_and_totals():
    result = score(TRUTH, ANSWERS.__getitem__)
    assert [r["verdict"] for r in result["rows"]] == ["correct", "partial", "wrong", "wrong"]
    assert (result["correct"], result["partial"], result["wrong"]) == (1, 1, 1 + 1)


def test_coverage_is_share_of_visible_time():
    result = score(TRUTH, ANSWERS.__getitem__)
    assert result["rows"][0]["coverage"] == 1.0
    assert result["rows"][2]["coverage"] == 0.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.score'`

- [ ] **Step 4: Implement `eval/score.py`**

```python
"""Score the memory's answers against ground truth written before the run.

    python -m eval.score eval/truth/test1.json memory_live
"""
import json
import sys

REST_TOL = 2.0  # seconds; within this of the true put-down moment counts as correct


def _overlap(spans, visible):
    return sum(max(0.0, min(e, ve) - max(s, vs)) for s, e in spans for vs, ve in visible)


def _verdict(answer, truth, covered):
    if not answer["found"] or covered == 0:
        return "wrong"
    return "correct" if abs(answer["rest_time"] - truth["rest_time"]) <= REST_TOL else "partial"


def score(truth, ask):
    rows = []
    for item in truth["objects"]:
        answer = ask(item["query"])
        spans = [(s["start"], s["end"]) for s in answer["sightings"]]
        covered = _overlap(spans, item["visible"])
        total = sum(e - s for s, e in item["visible"])
        rows.append({"query": item["query"], "verdict": _verdict(answer, item, covered),
                     "rest_time": answer["rest_time"], "truth_rest": item["rest_time"],
                     "coverage": round(min(1.0, covered / total), 2)})
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "partial", "wrong")}
    return {"rows": rows, **counts}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    from engine.pipeline import Engine
    from engine.store import MemoryStore
    truth = json.loads(open(sys.argv[1], encoding="utf-8").read())
    engine = Engine(None, None, MemoryStore(sys.argv[2], fresh=False))
    result = score(truth, engine.ask)
    for r in result["rows"]:
        print(f"{r['verdict']:8} {r['query']:30} answered {r['rest_time']}  truth {r['truth_rest']}  coverage {r['coverage']}")
    print(f"{result['correct']} correct, {result['partial']} partial, {result['wrong']} wrong"
          f"  ({truth.get('split', 'unknown')} split)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_score.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add eval/score.py eval/truth/README.md tests/test_score.py
git commit -m "feat: score memory answers against ground truth

Automates version 1's hand scoring with the same correct, partial and
wrong verdicts, plus coverage to expose objects split across identities."
```

---

### Task 12: Calibrate and measure (manual)

**Files:**
- Create: `eval/truth/calib1.json`, `eval/truth/test1.json`, `eval/truth/test2.json`, `eval/truth/test3.json`, `docs/design/phase1-results.md`
- Modify: `engine/identity.py` (`MATCH_SIM` only)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Record four videos (phone at eye level, about 1 minute each, no other people in frame)**

One calibration video and three test videos, at least 20 objects across the three test videos. In each: put objects down, leave and come back so objects leave the view and return, move some things. **Write each truth file while recording or straight after, before running anything.** Commit the four truth files on their own:

```bash
git add eval/truth/*.json
git commit -m "test: ground truth for phase 1 videos, written before any run"
```

- [ ] **Step 2: Calibrate `MATCH_SIM` on the calibration video only**

```
python -m engine.run videos/calib1.mp4 --out memory_calib --match-sim 0.5
python -m eval.score eval/truth/calib1.json memory_calib
```
Repeat with `--match-sim 0.6` and `0.7`. Pick the value with the most correct answers and the highest coverage; set `MATCH_SIM` in `engine/identity.py` to it.

- [ ] **Step 3: Run each test video once**

```
python -m engine.run videos/test1.mp4 --out memory_test1
python -m eval.score eval/truth/test1.json memory_test1
```
Same for test2 and test3. Do not change any threshold after seeing these numbers.

- [ ] **Step 4: Record the results**

`docs/design/phase1-results.md`: the calibration sweep table, the chosen `MATCH_SIM`, and per test video the score table and totals, with the version 1 baseline (3 of 5 on video 4) for comparison. Report the real numbers even when worse.

- [ ] **Step 5: Commit**

```bash
git add engine/identity.py docs/design/phase1-results.md
git commit -m "docs: phase 1 results on three held-out videos

MATCH_SIM was set on the calibration video only; the test numbers are
from a single run each."
```
