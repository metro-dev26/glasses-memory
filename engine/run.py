"""Run the engine over a recorded video, then answer questions from its memory.

    python -m engine.run videos/sample4.mp4 --ask "remote" "sunglasses"
    python -m engine.run videos/sample4.mp4 --store store/sample4 --fresh

Times are seconds from the start of the video. Without --ask, it prints the memory.
"""
import argparse
import shutil
import time

from engine.engine import Engine
from engine.video import frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--store", default="store")
    ap.add_argument("--fresh", action="store_true", help="forget everything first")
    ap.add_argument("--ask", nargs="*", default=[])
    args = ap.parse_args()

    if args.fresh:
        shutil.rmtree(args.store, ignore_errors=True)
    engine = Engine(args.store)
    started, n, reids = time.perf_counter(), 0, 0
    for ts, frame in frames(args.video):
        detections, _ = engine.process(frame, ts)
        reids += sum(d["reidentified"] for d in detections)
        n += 1
    engine.memory.save()
    took = time.perf_counter() - started

    objects = sorted(engine.memory.objects.values(), key=lambda o: -o.last_seen)
    print(f"{n} frames in {took:.0f} s ({n / took:.1f} fps), {len(engine.tracks)} live tracks at the end")
    print(f"{len(objects)} objects in memory, {reids} re-identifications\n")
    if not args.ask:
        for o in objects:
            print(f"  #{o.id:<3} {o.name:20s} {len(o.sightings):2d} sightings, "
                  f"last {o.last_seen:5.1f}s  {o.where}")
    for q in args.ask:
        a = engine.ask(q)
        print(f"{q!r}: {a['text']}")
        if a["found"]:
            print(f"   object #{a['object_id']}, snapshot {args.store}{a['snapshot'].replace('/snapshot', '/snapshots')} "
                  f"at {a['seen_at']:.1f}s")
        else:
            print(f"   closest: {a['candidates']}")


if __name__ == "__main__":
    main()
