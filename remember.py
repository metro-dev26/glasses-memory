"""Consolidate raw tracks into human-readable memory (the "sleep" step).

Raw tracks are per-frame noise. Like a night's sleep, this keeps only the
episodes worth remembering and writes them as markdown you can open and read:

    memory/MEMORY.md            one line per object: where/when last seen
    memory/<object>.md          the object's card + its episodes, newest first
    memory/snapshots/*.jpg      the best frame of each episode

    python remember.py out/sample3.base_plus.tracks.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import clip
import cv2
import torch
from PIL import Image

from track import FPS, REGISTRY, LazyFrames

ROOT = Path(__file__).parent
MEMORY = ROOT / "memory"
MIN_AREA = 300      # pixels; smaller masks are tracking crumbs
MAX_GAP = 5         # frames (0.5 s); a longer break starts a new episode
MIN_EPISODE = 3     # frames (0.3 s); shorter episodes are one-off glitches
MIN_LOOKALIKE = 0.70  # CLIP similarity to the registration close-up; below = the
                      # tracker drifted onto something else (tuned on video 3 only)


def episodes(track):
    """Group confident frames into episodes, dropping one-off glitches."""
    runs = []
    for p in (p for p in track if p["area"] >= MIN_AREA):
        if runs and p["frame"] - runs[-1][-1]["frame"] <= MAX_GAP:
            runs[-1].append(p)
        else:
            runs.append([p])
    return [r for r in runs if len(r) >= MIN_EPISODE]


class Lookalike:
    """Does this crop look like the object as it was registered?"""

    def __init__(self, frames):
        self.frames = frames
        self.model, self.prep = clip.load("ViT-B/32", device="cuda", download_root=str(ROOT / "models" / "weights" / "clip"))
        self.refs = {}
        for name, (t, (x1, y1, x2, y2)) in REGISTRY.items():
            self.refs[name] = self.embed(frames.raw(int(t * FPS))[y1:y2, x1:x2])

    def embed(self, bgr):
        with torch.no_grad():
            e = self.model.encode_image(self.prep(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda())
        return (e / e.norm()).float().cpu().numpy()[0]

    def score(self, name, p):
        x1, y1, x2, y2 = p["box"]
        return float(self.embed(self.frames.raw(p["frame"])[y1:y2 + 1, x1:x2 + 1]) @ self.refs[name])


def slug(name):
    return name.replace(" ", "-")


def main(tracks_file):
    data = json.loads(Path(tracks_file).read_text())
    frames = LazyFrames(data["video"])
    lookalike = Lookalike(frames)
    (MEMORY / "snapshots").mkdir(parents=True, exist_ok=True)
    recorded = datetime.fromtimestamp(Path(data["video"]).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    index = []

    for name, track in data["tracks"].items():
        eps = [ep for ep in episodes(track)
               if lookalike.score(name, max(ep, key=lambda p: p["area"])) >= MIN_LOOKALIKE]
        lines = [f"---\nname: {name}\nsource: {data['video']}\nrecorded: {recorded}\n---\n",
                 f"# {name}\n", "## Episodes (newest first)\n"]
        for i, ep in enumerate(reversed(eps)):
            best = max(ep, key=lambda p: p["area"])   # clearest view of the episode
            img = frames.raw(best["frame"])
            x1, y1, x2, y2 = best["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            shot = f"snapshots/{slug(name)}-{best['time']:.1f}s.jpg"
            cv2.imwrite(str(MEMORY / shot), img)
            tag = " **← last seen**" if i == 0 else ""
            lines.append(f"- {ep[0]['time']:.1f}s – {ep[-1]['time']:.1f}s{tag}  \n  ![]({shot})")
        if not eps:
            lines.append("- never seen clearly")
        (MEMORY / f"{slug(name)}.md").write_text("\n".join(lines) + "\n")

        if eps:
            last = eps[-1]
            index.append(f"- [{name}]({slug(name)}.md) — last seen {last[0]['time']:.1f}–{last[-1]['time']:.1f}s "
                         f"into the {recorded} recording ({len(eps)} episodes)")
        else:
            index.append(f"- [{name}]({slug(name)}.md) — never seen clearly")

    (MEMORY / "MEMORY.md").write_text(
        "# Object memory\n\nLast known, not current: things may have moved since the recording.\n\n"
        + "\n".join(index) + "\n")
    print((MEMORY / "MEMORY.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1])
