# Phase 1 results: the engine on recorded video

Measured 2026-10-09. Calibration video: `sample3` (all thresholds set there).
Test video: `sample4`, run once, nothing tuned on it. Truth: `eval/truth/sample4.json`,
written for version 1 before version 2 existed.

## Speed

`sample3`, 690 frames at 10 fps: 35 fps for the whole engine (detect, track,
identify, memory, snapshots) on the RTX 3050.

## Identity (sample3, `eval/calibrate_identity.py`)

At MATCH 0.70, MARGIN 0.05: 70% of returning tracks re-identified, 7% wrong
merges, up to 30% of never-seen objects merged into something (upper bound).
Two calibration findings changed the code:
- DINOv3's CLS token, not timm's default patch average (different objects:
  median similarity 0.28 instead of 0.64).
- The margin is measured only against runners-up that look different from the
  winner. Before this, every split caused more splits: 116 objects -> 103.

## "Where" answers (sample4, test, `eval/where.py`)

| Object | Truth | By photo | By text |
|---|---|---|---|
| remote | bed, 38.0 s | right object, earlier sighting (desk, 7.6 s) | right object, same earlier sighting |
| hand gripper | side table, 47.1 s | right object (labelled "toy gun"), earlier sighting (33.8 s) | not found |
| yellow stapler | table, 59.8 s | **on time** (labelled "bookmark") | not found |
| steel stapler | desk by laptop, 30.0 s | right object (labelled "pen"), earlier sighting (22.3 s) | not found |
| sunglasses | notebook, 24.0 s | wrong: a fingertip | not found |

**On time: photo 1/5, text 0/5.** Version 1, with a hand-drawn box per object, got 3/5.

Snapshots were checked by eye: in 4 of 5 the photo answer is the **right
object** but an **older sighting**. The later sighting was stored as a separate
object (a missed re-identification), so "last seen" stops too early.
Text fails because prompt-free labels are wrong for these objects ("toy gun",
"bookmark", "pen"); renaming an object once fixes that in the product, but it
was not done here.

## Next, to be proven on a new video (not on sample4)

1. Fewer missed re-identifications: the main loss above.
2. Answer by photo from the most recent of all objects that match the photo
   (the text path already does this for ties).
3. sample4 has been used now; the next number needs a fresh test video with
   truth written before recording.
