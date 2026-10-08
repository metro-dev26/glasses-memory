# Handbook

Everything you need to work on glasses-memory: what it does, why it is built this way,
how every file works, what we measured, what is still broken, and the rules we follow.
Read it top to bottom once. Then use it as a reference.

---

## 1. The idea in one paragraph

Smart glasses see everything you see. So they could answer "where did I leave my keys?".
This project does that for a recorded first-person video. You **register** each object once
(a box around it on a close-up frame). The system **tracks** every object through the whole
video, **consolidates** the raw tracking into a few memorable "episodes" (like sleep turns a
day into memories), and **remembers** each object as a small markdown file. Then you ask
"where's my remote?" and get the moment you put it down, with a snapshot and the video clip.

Answers are always **last known, not current**: the memory knows where the object was when
the camera last saw it, not where it is now.

## 2. Why it is built this way (the path we took)

Each step fixed a failure of the previous one. Knowing this is the best way to understand
the code, and it is what an interviewer will ask about.

1. **YOLO-World, per frame** (`experiments/memory.py`). An open-vocabulary detector: you type
   "yellow stapler", it finds yellow staplers in each frame. Failed: on video 3 it got
   0 of 5 objects right. Problem: it looks at each frame on its own. When the object is
   blurry, half-hidden, or at an odd angle for a few frames, it is simply missed, and it also
   fires on other yellow things. It knows *kinds* of things, not *your* thing.
2. **OWLv2 image-guided search** (`experiments/owl_test*.py`). Give it a crop of your
   object, it searches for that crop. Weak: scores barely separated the real object from
   lookalikes. (Note: its sigmoid scores all came out as 1.00, so we had to read the raw logits.)
3. **SAM 2 video tracking** (`experiments/sam_test.py`, then `track.py`). The big jump. SAM 2 (Meta's
   Segment Anything 2) takes one box on one frame and follows that exact object through the
   video, using a *memory* of what it looked like in earlier frames. Continuity is the point:
   it doesn't re-decide from scratch on each frame. Video 3 went to 3/5.
4. **All objects in one pass.** Tracking all objects together instead of one at a time was
   6x faster with the same accuracy. Bonus: objects compete for pixels, so one track can't
   jump onto another registered object.
5. **SAM 2 base+ instead of tiny.** The bigger model re-finds objects better after they leave
   the frame and come back.
6. **The "sleep" step** (`remember.py`). Raw tracks are noisy. Group frames into episodes,
   drop tiny glitches, and drop episodes that don't look like the registered object.
   Video 3 went to 4/5 with 1 partial.
7. **Validation on video 4** with new objects and *nothing tuned on it*: 3/5 correct,
   1 partial, 1 wrong. Lower, as expected. This is the honest number.

The human-memory idea came from how people (and note files) organise memory:
episodes, consolidation during sleep, an index of what you know, one card per thing,
and the habit of saying "last known" instead of pretending to know the present.

## 3. Setup

Needs an NVIDIA GPU (developed on an RTX 3050 with 4 GB VRAM) and **Python 3.11**
(newer Pythons lack the torch wheels).

```bash
git clone https://github.com/metro-dev26/glasses-memory && cd glasses-memory
python3.11 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
./setup.sh        # downloads SAM 2.1 tiny + base_plus checkpoints into models/
```

Known install trap: SAM 2 from git can fail with build errors ("invalid command bdist_wheel").
Fix:

```bash
pip install wheel setuptools
SAM2_BUILD_CUDA=0 pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git
```

CLIP downloads its own weights (~340 MB) into `models/weights/clip/` the first time it runs.

**Videos are not in the repo.** They are personal footage of Abhi's room (another person
appears in one). Ask Abhi for them privately, or record your own (see section 9).

## 4. Running it

One command:

```bash
python run.py videos/myvideo.mp4            # base_plus model (accurate, ~5 min per video minute)
python run.py videos/myvideo.mp4 tiny       # faster, less accurate
```

If the video has no registry yet, the register window opens first. Then it tracks,
consolidates, builds the ask page and opens it in the browser.

Or step by step (useful while developing):

```bash
python register.py videos/sample4.mp4                       # -> registry/sample4.json
python track.py videos/sample4.mp4 base_plus                # -> out/sample4.base_plus.tracks.json
python remember.py out/sample4.base_plus.tracks.json        # -> memory/
python ask_page.py                                          # -> memory/index.html
```

Re-running `track.py` is the slow part. When you only change the consolidation logic,
re-run `remember.py` on the existing tracks file. That takes seconds.

## 5. Every file, explained

### `register.py`: show the memory what each object looks like
Opens the video in a window at 10 fps steps. Keys: `a`/`d` = 0.5 s back/forward,
`A`/`D` = 3 s, `space` = drag a box (then Enter) and type the name in the terminal,
`u` = undo last, `q` = save. Writes `registry/<video stem>.json`:

```json
{ "remote": [7.0, [130, 170, 380, 854]] }
```
= object name -> [time in seconds of the close-up frame, box x1, y1, x2, y2 in original-video pixels].

Pick a frame where the object is **big, sharp and alone**. That one box is all SAM 2 knows
about the object, and its crop is also the reference for the look-alike check.

### `track.py`: follow every object with SAM 2
- `FPS = 10`: the video is sampled at 10 frames per second. Enough to follow hands moving
  objects, and 3x cheaper than 30 fps.
- `IMAGE_SIZE = 1024`: SAM 2's input resolution. Each frame is resized and normalised
  (ImageNet mean/std) before going in.
- `MODELS`: tiny vs base_plus config + checkpoint names.
- **`LazyFrames`**: SAM 2's own loader decodes the *whole* video into RAM up front
  (~8 GB for one minute at 1024x1024 float). The laptop had ~5 GB free. `LazyFrames`
  decodes one frame only when SAM 2 asks for it. We swap it in by replacing
  `sam2.sam2_video_predictor.load_video_frames` (a monkeypatch). `raw(i)` gives the original
  BGR frame (used for snapshots and crops), `__getitem__` gives the normalised tensor SAM 2 wants.
- **`track()`**: builds the predictor, adds one box prompt per object *on that object's own
  registration frame* (objects can be registered at different times), then
  `propagate_in_video` walks the whole video. Runs under `torch.inference_mode()` and bf16
  autocast, with `offload_state_to_cpu=True`, so it fits in 4 GB VRAM.
- Output per object, per frame where its mask is non-empty: `frame`, `time`, mask `area`
  (pixels), and the mask's bounding `box`. **The masks come back at the original video
  resolution** (480x854 here), not 1024. Don't rescale the boxes (we did once, and every
  box was wrong).

### `remember.py`: the "sleep" step
Turns noisy per-frame tracks into a few trustworthy memories.

- **`episodes(track, registered_at)`**: keep frames with mask area >= `MIN_AREA` (300 px;
  smaller masks are tracking crumbs). Consecutive frames with gaps <= `MAX_GAP` (5 frames
  = 0.5 s) form one episode. Episodes shorter than `MIN_EPISODE` (3 frames = 0.3 s) are
  one-off glitches and are dropped. **Frames before the object's registration are ignored**:
  SAM 2 also "guesses" backwards in time before the prompt, and you can't remember something
  you hadn't been shown yet.
- **`Lookalike`**: CLIP ViT-B/32. Embeds the registration crop of each object. For each
  episode, it takes the frame with the biggest mask, crops it, and compares
  (cosine similarity). Below `MIN_LOOKALIKE = 0.70`, the tracker probably drifted onto
  something else, so the episode is dropped. 0.70 was tuned on video 3 only.
- **`resting_frame(ep)`**: which frame to show as "where it was left". Not the biggest
  frame (that's usually the object in your hand, close to the camera). It is the **last
  frame of the episode whose area is at least 10% of the episode's biggest area**: the last
  clear view before it left the frame. (30% was too strict: it threw away the steel stapler
  sitting on the desk.)
- **`main()`** wipes `memory/` and writes:
  - `memory/MEMORY.md`: the index. One line per object: when it was last seen, how many episodes.
  - `memory/<object>.md`: one card per object: frontmatter (name, source, recorded),
    episodes newest first, each with a snapshot.
  - `memory/snapshots/*.jpg`: the resting frame with a yellow box.
  - `memory/memory.json`: the same facts as data, for the ask page.

### `ask_page.py`: the "where's my X?" page
Builds `memory/index.html`, a single offline page. The memory JSON is **embedded into the
HTML** because browsers block a `file://` page from reading other local files.
- Matching is simple keyword overlap: stop-words ("where", "my", "leave"…) are removed,
  and the object whose name shares the most words wins. "glasses" finds "sunglasses".
- Answer = latest episode: its resting time, snapshot, and the video playing from
  4 s before the resting moment to 1 s after (`video.mp4#t=from,to`).
- Unknown object: "I don't remember anything like that", plus the list of known objects.
- `video_src` is computed relative to `memory/`, so videos can live anywhere.

### `run.py`
register (only if there's no registry) -> track -> free GPU memory -> remember -> ask page
-> open the browser.

### `experiments/`: old approaches (kept on purpose, they are the story)
`memory.py` (YOLO-World + CLIP), `owl_test.py` / `owl_test2.py` (OWLv2), `sam_test.py`
(SAM 2 via ultralytics, one object at a time). Run them from the repo root; they need
`pip install ultralytics transformers`, which the pipeline does not.

### Data folders (all gitignored)
`videos/` footage · `out/` tracks JSON · `memory/` generated memory · `models/` checkpoints.
`registry/*.json` **is** committed (just names, times and boxes).

## 6. Results so far

**Video 3** (used for tuning): 4 correct, 1 partial (the pen: tracked while it was moving,
lost before it was set down), 0 wrong.

**Video 4** (validation, nothing tuned on it):

| Object | What really happened | What the memory says | |
|---|---|---|---|
| Remote | moved to the bed | on the bed, 0:38.0 | ✅ |
| Hand gripper | moved to the side table | side table, 0:47.1 | ✅ |
| Yellow stapler | out of the cupboard, onto the table | on the table, 0:59.8 | ✅ |
| Steel stapler | put on the desk by the laptop, ~0:30 | episode ends 0:30.6, but the snapshot is still in the hand (0:28.1) | 🟡 |
| Sunglasses | left to right on the notebook, ~0:24 | a false episode in the cupboard at 0:54 (look-alike 0.74, passed 0.70) | ❌ |

## 7. What is weak, and what's next

1. **The look-alike check is category-level.** CLIP knows "sunglasses-ish thing", not
   "*these* sunglasses". The false sunglasses episode scored 0.74. Next: an instance-level
   embedding such as **DINOv2**, and/or checking several crops per episode instead of one.
2. **Validate on a fresh video 5.** Any fix found on video 4 is tuned on video 4.
   It only counts once it holds on a video nobody has looked at.
3. **Snapshot timing** (steel stapler): the resting frame can still catch the hand.
   An idea: prefer frames where the box stops moving.
4. **Speed**: base_plus is ~5 min per video minute on a 3050. Fine for a demo, not for glasses.
5. **Registration** still needs a close-up per object. Ideas: register from a separate photo,
   or from a short "show the object" clip.

## 8. Rules we follow

- **Never retune on the validation video.** If you change a threshold because of video 4,
  video 4 is no longer a test. Find the fix, then prove it on a new video. Report the real
  number even when it is worse.
- **Personal footage never gets committed.** `videos/`, `out/`, `memory/` stay gitignored.
  The demo GIF only uses clips without other people in them.
- **Every change gets a before/after number.** The scoring is by hand: for each object,
  compare the memory's answer with what really happened (ask whoever recorded the video).
- **Both of us must be able to explain every line.** This is an internship application.
  If either of us can't defend a piece of code in an interview, it isn't done yet.
  When something is changed, explain the why in the commit message and walk the other
  person through it.
- **Workflow**: work on a branch, open a pull request, the other person reads it before merge.
  `master` stays working.
- **Commit messages**: conventional (`feat:`, `fix:`, `docs:`), body explains why.
  No co-author trailers.
- Keep code small and readable, matching the style already there: short functions,
  module docstring with a usage line, constants at the top with a comment saying why
  that value.

## 9. How to record a good test video

- Hold the phone at eye level (it pretends to be glasses). Portrait is fine.
- **First, show each object up close**, alone, for 1–2 seconds. That's the registration frame.
- Then move things around normally. Put each one down and **let the camera see it resting**
  for a second before moving on.
- Write down the truth as you go ("remote -> bed"). That's your ground truth for scoring.
- Keep other people out of frame.
- 1 minute is plenty.

## 10. Glossary

- **Egocentric video**: first-person video, recorded from the wearer's eyes.
- **Open-vocabulary detection**: detecting objects from a text prompt instead of a fixed class list.
- **Segmentation mask**: per-pixel "this is the object" map. SAM 2 outputs masks; we keep
  their area and bounding box.
- **Prompt (SAM 2)**: the hint telling SAM 2 which object to follow. Here, a box.
- **Propagation**: SAM 2 carrying a mask from the prompt frame through the rest of the video.
- **Embedding / cosine similarity**: a vector describing an image; similarity near 1 = looks alike.
- **Episode**: a continuous stretch where an object was clearly seen.
- **Consolidation**: turning raw per-frame data into episodes and memory files.
- **Validation set**: data held back to measure honestly, never used for tuning.
