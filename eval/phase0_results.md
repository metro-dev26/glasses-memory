# Phase 0 results: speed on the RTX 3050 (4 GB)

Measured 2026-10-09 with `eval/phase0_speed.py` on `videos/sample3.mp4`
(480x854 phone video, 600 timed frames after 20 warm-up frames).
Pipeline per frame: YOLOE prompt-free detect + ByteTrack, then DINOv3 on each
track that reaches 5 frames. fp16, torch 2.5.1, ultralytics 8.4.173.
GPU time only: no JPEG decode, no network, no memory writes.

| Detector | imgsz | DINOv3 | Detect+track p50 | Total p50 / p95 | fps | GPU reserved |
|---|---|---|---|---|---|---|
| yoloe-11s-seg-pf | 640 | ViT-S/16 | 15.6 ms | 15.9 / 28.2 ms | 55.9 | 412 MiB |
| yoloe-11s-seg-pf | 960 | ViT-S/16 | 20.0 ms | 20.5 / 33.1 ms | 44.3 | 1012 MiB |
| yoloe-11m-seg-pf | 640 | ViT-B/16 | 19.2 ms | 19.4 / 31.5 ms | 47.1 | 526 MiB |
| yoloe-11l-seg-pf | 640 | ViT-B/16 | 24.0 ms | 24.3 / 36.3 ms | 38.6 | 534 MiB |

Identify adds about 11 ms on the frames where it runs (1 to 3 crops).

## What this means

- The 10 fps target is met with 4x headroom even with the largest models.
  Speed is not the constraint; the phone's ~10 fps send rate is.
- GPU memory is not a constraint either (under 1.1 GB of 4 GB).
- **Tracks break often:** about 120 to 180 track ids in 20 seconds of video
  with about 6 objects per frame. Each break is a re-identification decision,
  so identity (section 5.1 of the design) is where the accuracy work is,
  which matches ESOM's analysis.

Not yet measured: end-to-end latency phone -> server -> dashboard over Tailscale.
