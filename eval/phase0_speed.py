"""Phase 0 feasibility: can detect -> track -> identify run live on this GPU?

Replays a recorded video as fast as the pipeline allows and measures
per-stage time, end-to-end fps and peak GPU memory. No accuracy is measured
here, only speed.

    python eval/phase0_speed.py videos/sample3.mp4 --det yoloe-11s-seg-pf.pt --imgsz 640
"""

import argparse
import statistics
import time

import cv2
import numpy as np
import timm
import torch
from ultralytics import YOLOE

MIN_TRACK_AGE = 5          # frames a track must survive before it is identified
EMBED_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_long_side(frame, long_side):
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def crops_to_batch(frame, boxes):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = []
    for x1, y1, x2, y2 in boxes:
        crop = rgb[int(y1):int(y2), int(x1):int(x2)]
        crop = cv2.resize(crop, (EMBED_SIZE, EMBED_SIZE)).astype(np.float32) / 255
        batch.append((crop - MEAN) / STD)
    return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)


def ms(samples):
    if not samples:
        return "n/a"
    s = sorted(samples)
    return f"p50 {statistics.median(s):6.1f}  p95 {s[int(len(s) * 0.95)]:6.1f}  mean {statistics.mean(s):6.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--det", default="yoloe-11s-seg-pf.pt")
    ap.add_argument("--embed", default="vit_small_patch16_dinov3.lvd1689m")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--long-side", type=int, default=960, help="what the phone page sends")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    device = "cuda"
    detector = YOLOE(args.det)
    embedder = timm.create_model(args.embed, pretrained=True, num_classes=0).to(device).half().eval()

    cap = cv2.VideoCapture(args.video)
    track_age = {}
    identified = set()
    t_det, t_emb, t_total, n_embedded, n_boxes = [], [], [], [], []

    torch.cuda.reset_peak_memory_stats()
    i = 0
    while i < args.frames + args.warmup:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_long_side(frame, args.long_side)

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = detector.track(frame, persist=True, tracker="bytetrack.yaml",
                                imgsz=args.imgsz, half=True, verbose=False)[0]
        torch.cuda.synchronize()
        after_det = time.perf_counter()

        to_embed = []
        if result.boxes.id is not None:
            for box, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.int().tolist()):
                track_age[tid] = track_age.get(tid, 0) + 1
                if track_age[tid] == MIN_TRACK_AGE and tid not in identified:
                    to_embed.append(box)
                    identified.add(tid)
        if to_embed:
            with torch.inference_mode():
                feats = embedder(crops_to_batch(frame, to_embed).to(device).half())
                feats = torch.nn.functional.normalize(feats, dim=1).cpu()
        torch.cuda.synchronize()
        end = time.perf_counter()

        if i >= args.warmup:
            t_det.append((after_det - start) * 1000)
            if to_embed:
                t_emb.append((end - after_det) * 1000)
            t_total.append((end - start) * 1000)
            n_embedded.append(len(to_embed))
            n_boxes.append(len(result.boxes))
        i += 1

    frames = len(t_total)
    print(f"video        {args.video}  ({frames} timed frames, long side {args.long_side})")
    print(f"detector     {args.det} @ imgsz {args.imgsz}, fp16")
    print(f"embedder     {args.embed}, fp16")
    print(f"boxes/frame  mean {statistics.mean(n_boxes):.1f}  max {max(n_boxes)}")
    print(f"tracks       {len(track_age)} seen, {len(identified)} identified")
    print(f"detect+track {ms(t_det)} ms")
    print(f"identify     {ms(t_emb)} ms  (on {len(t_emb)} frames, max {max(n_embedded)} crops)")
    print(f"total        {ms(t_total)} ms")
    print(f"fps          {1000 / statistics.mean(t_total):.1f}")
    print(f"gpu peak     {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved")


if __name__ == "__main__":
    main()
