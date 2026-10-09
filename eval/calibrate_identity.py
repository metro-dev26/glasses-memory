"""Pick the identity thresholds from a calibration video, with no hand labels.

The tracker gives free ground truth: one track is one object. Each long track
is split in two. Its first half builds a gallery the way the engine does; five
frames from its second half (after a 0.5 s gap) are the "returning" object.
  genuine  - the returning track scored against every gallery, its own included:
             a match should pick its own gallery
  impostor - the same, with its own gallery removed: a match is a wrong merge,
             the case of an object the memory has never seen
Two tracks on one physical object count as different here, so the error
rates are upper bounds. Only ever run this on calibration videos.

    python -m eval.calibrate_identity videos/sample3.mp4
"""
import sys
from collections import defaultdict

import numpy as np

import engine.identify as identify
from engine.detect import Detector
from engine.identify import Embedder
from engine.video import frames

MIN_TRACK = 25           # frames (2.5 s) a track needs to be split into gallery + query
QUERY = 5                # frames averaged into the returning fingerprint, as in the engine


def collect(video):
    detector, embedder = Detector(), Embedder()
    by_track = defaultdict(list)
    for _, frame in frames(video):
        objects, _ = detector(frame)
        for o, f in zip(objects, embedder(frame, [o["box"] for o in objects])):
            by_track[o["track_id"]].append(f)
    return [np.array(fs) for fs in by_track.values() if len(fs) >= MIN_TRACK]


def split(tracks):
    galleries, queries = {}, {}
    for i, fs in enumerate(tracks):
        half = len(fs) // 2
        gallery = np.zeros((0, fs.shape[1]), dtype=fs.dtype)
        for f in fs[:half - QUERY]:
            gallery = identify.add_view(gallery, f)
        q = fs[half:half + QUERY].mean(0)
        galleries[i], queries[i] = gallery, q / np.linalg.norm(q)
    return galleries, queries


def rates(galleries, queries):
    correct = wrong = impostor = 0
    for i, q in queries.items():
        oid, _ = identify.best_match(q, galleries)
        correct += oid == i
        wrong += oid is not None and oid != i
        impostor += identify.best_match(q, galleries, exclude={i})[0] is not None
    n = len(queries)
    return correct / n, wrong / n, impostor / n


def main(video):
    tracks = collect(video)
    galleries, queries = split(tracks)
    print(f"{len(tracks)} tracks of >= {MIN_TRACK} frames\n")
    print("MATCH  MARGIN | re-identified | wrong merge | unseen object merged")
    for match in (0.5, 0.6, 0.7, 0.8):
        for margin in (0.0, 0.05, 0.1):
            identify.MATCH, identify.MARGIN = match, margin
            c, w, imp = rates(galleries, queries)
            print(f" {match:.2f}   {margin:.2f}  |    {c:6.1%}     |   {w:6.1%}    |   {imp:6.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
