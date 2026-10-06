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
The 0.70 cutoff was tuned on this one video, so it needs a fresh video to validate.

Earlier experiments: `memory.py` (YOLO-World), `owl_test*.py` (OWLv2 one-shot), `sam_test.py`.
