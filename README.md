# glasses-memory

"Where did I leave my keys?" for first-person (smart-glasses style) video.

Register an object once, follow it through the video, remember where it was put down.

## How it works
1. **Register**: one box per object on its close-up frame (`REGISTRY` in `track.py`).
2. **Track** (`track.py`): one SAM 2 pass follows every object together, at 10 fps.
3. **Sleep** (`remember.py`): turn per-frame tracks into episodes. Drop glitches (<0.3 s)
   and episodes that don't look like the registered object (CLIP similarity < 0.70).
4. **Remember**: write `memory/MEMORY.md` plus one markdown card per object with snapshots.
   Answers are "last known, not current".

## Results on test video 3 (5 moved objects, ground truth from the recorder)
| Approach | Correct | Partial | Wrong |
|---|---|---|---|
| YOLO-World + CLIP filter (`memory.py`) | 0 | 2 | 3 |
| SAM 2 tiny, one object at a time | 3 | 1 | 1 |
| SAM 2 tiny, all objects in one pass (6x faster) | 3 | 1 | 1 |
| SAM 2 base+, one pass, + look-alike check | 4 | 1 | 0 |

Partial = the pen: tracked while being moved, lost before it was set down.

## Validation: video 4 (new objects, nothing tuned on it)
| Object | Truth | Memory says | |
|---|---|---|---|
| Remote | moved to the bed | on the bed, 38.0s | ✅ |
| Hand gripper | moved to the side table | on the side table, 47.1s | ✅ |
| Yellow stapler | out of the cupboard, onto the table | on the table, 59.8s | ✅ |
| Steel stapler | (recorder forgot; footage shows the desk by the laptop, ~30s) | episode ends 30.6s, snapshot still in hand at 28.1s | 🟡 |
| Sunglasses | left → right on the notebook (~24s) | a false episode in the cupboard at 54s (look-alike 0.74 > 0.70) | ❌ |

**3 / 5 correct, 1 partial, 1 wrong.** Lower than video 3 (4/5), as expected for an
untuned video. The weak link is the look-alike check: CLIP knows "kind of thing",
not "this exact thing". Next: an instance-level embedding (e.g. DINOv2) or several
crops per episode, validated on a 5th video.

Fixes found on video 4 (design fixes, not threshold tuning):
- snapshot = the last clear frame of an episode (where it was left), not the biggest
- ignore tracks before an object is registered (SAM 2 guesses before its prompt)

Earlier experiments: `memory.py` (YOLO-World), `owl_test*.py` (OWLv2 one-shot), `sam_test.py`.
