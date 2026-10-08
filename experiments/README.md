# Earlier experiments

The approaches tried before SAM 2 tracking, kept because they explain the design
(see [HANDBOOK section 2](../docs/HANDBOOK.md#2-why-it-is-built-this-way-the-path-we-took)).
None of them is part of the pipeline.

| File | Approach | Result on video 3 |
|---|---|---|
| `memory.py` | YOLO-World per frame + CLIP filter | 0 correct of 5 |
| `owl_test.py`, `owl_test2.py` | OWLv2 image-guided one-shot search | scores barely separated the object from look-alikes |
| `sam_test.py` | SAM 2 via ultralytics, one object at a time | 3 correct of 5 |

Run them from the repository root, e.g. `python experiments/memory.py ask videos/sample3.mp4 "yellow stapler"`.
They need two packages the pipeline does not:

```bash
pip install ultralytics transformers
```

The OWLv2 and SAM tests read `videos/sample3.mp4` and `refs_crops/`, which are personal footage and not in the repository.
