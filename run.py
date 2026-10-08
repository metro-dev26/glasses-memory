"""One command: register (if needed) -> track -> remember -> open the ask page.

    python run.py videos/myvideo.mp4 [tiny|base_plus]
"""
import sys
import webbrowser
from pathlib import Path

import torch

import ask_page
import register
import remember
import track

ROOT = Path(__file__).parent


def main(video, size="base_plus"):
    if not (ROOT / "registry" / f"{Path(video).stem}.json").exists():
        print("No objects registered for this video yet. Opening the register window.")
        register.main(video)
    print(f"Tracking with SAM 2 ({size})...")
    track.track(video, size)
    torch.cuda.empty_cache()
    print("Consolidating into memory...")
    remember.main(ROOT / "out" / f"{Path(video).stem}.{size}.tracks.json")
    page = ask_page.build()
    webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
