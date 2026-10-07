"""Track every registered object through a video in ONE SAM 2 pass.

Each object gets a box on its own close-up frame (registry/<video>.json). Tracking them together means
objects compete for pixels, so one object's track can't jump onto another.

    python track.py videos/sample3.mp4 [tiny|base_plus]
writes out/<video>.tracks.json: per object, per frame -> mask area + box.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import sam2.sam2_video_predictor as svp
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).parent
FPS = 10              # frames per second fed to SAM 2
IMAGE_SIZE = 1024     # SAM 2's input resolution
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

MODELS = {  # size -> (config, checkpoint)
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
}

def load_registry(video):
    """registry/<video>.json: object -> [close-up time (s), box (x1, y1, x2, y2)]."""
    return json.loads((ROOT / "registry" / f"{Path(video).stem}.json").read_text())


class LazyFrames:
    """Hands SAM 2 one frame at a time instead of the whole video in RAM
    (the stock loader would need ~8 GB for a one-minute clip)."""

    def __init__(self, video):
        self.cap = cv2.VideoCapture(str(video))
        self.step = self.cap.get(cv2.CAP_PROP_FPS) / FPS
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.step)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def __len__(self):
        return self.n

    def raw(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, round(i * self.step))
        return self.cap.read()[1]

    def __getitem__(self, i):
        rgb = cv2.cvtColor(cv2.resize(self.raw(i), (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (img - MEAN) / STD


def track(video, size="tiny"):
    frames = LazyFrames(video)
    svp.load_video_frames = lambda **kw: (frames, frames.h, frames.w)

    config, ckpt = MODELS[size]
    predictor = build_sam2_video_predictor(config, str(ROOT / "models" / ckpt), device="cuda")
    registry = load_registry(video)
    names = list(registry)
    tracks = {name: [] for name in names}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video), offload_state_to_cpu=True)
        for obj_id, name in enumerate(names):
            t, box = registry[name]
            predictor.add_new_points_or_box(state, frame_idx=int(t * FPS), obj_id=obj_id,
                                            box=np.array(box, dtype=np.float32))
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            for obj_id, logits in zip(obj_ids, mask_logits):
                mask = (logits[0] > 0).cpu().numpy()
                if mask.sum() == 0:
                    continue
                ys, xs = np.nonzero(mask)
                tracks[names[obj_id]].append({
                    "frame": frame_idx,
                    "time": round(frame_idx / FPS, 1),
                    "area": int(mask.sum()),
                    "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                })
            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}/{len(frames)}", flush=True)

    out = ROOT / "out" / f"{Path(video).stem}.{size}.tracks.json"
    out.write_text(json.dumps({"video": str(video), "fps": FPS, "tracks": tracks}))
    print(f"wrote {out}")


if __name__ == "__main__":
    track(*sys.argv[1:])
