# glasses-memory

"Where did I leave my keys?" for first-person (smart-glasses style) video.

Register an object once, follow it through the video, remember where it was put down.

## Status (2026-10-06)
- `memory.py`: YOLO-World + CLIP per-frame detection with a similarity and persistence filter (v1). Fails on recall: the detector misses objects in their final spots.
- `owl_test*.py`: OWLv2 one-shot (image-guided) search. Weak present/absent separation.
- `sam_test.py`: SAM 2 tracking from one box per object. 3/5 correct on video 3.

Next: one multi-object SAM 2 pass, put-down events, markdown memory files.
