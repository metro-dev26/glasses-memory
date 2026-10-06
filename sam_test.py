"""Track each registered object with SAM 2 from its close-up to the end of the video.
Where the track ends (or where the object last sits) = where it was left."""
import json, subprocess, sys
import cv2, numpy as np
from ultralytics.models.sam import SAM2VideoPredictor

VIDEO = "videos/sample3.mp4"
FPS = 10
REFS = {  # close-up time (s), box (x1,y1,x2,y2) in the 480x854 frame
    "white deodorant can": (20, (100, 140, 360, 780)),
    "yellow stapler":      (16, (90, 220, 480, 500)),
    "steel stapler":       (12, (110, 250, 340, 740)),
    "bracelet":            (4,  (140, 0, 280, 854)),
    "red pen":             (14, (180, 30, 330, 770)),
}

summary = {}
for label, (t0, box) in REFS.items():
    clip = f"clips/{label.replace(' ', '_')}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-i", VIDEO,
                    "-vf", f"fps={FPS}", "-an", clip], check=True)
    predictor = SAM2VideoPredictor(overrides=dict(model="models/sam2.1_t.pt", imgsz=1024, conf=0.25, verbose=False, save=False))
    track = []  # (time, area, cx, cy)
    last_img = None
    for i, r in enumerate(predictor(source=clip, bboxes=[list(box)], stream=True)):
        t = t0 + i / FPS
        if r.masks is not None and r.masks.data.sum() > 0:
            m = r.masks.data[0].cpu().numpy() > 0.5
            ys, xs = np.nonzero(m)
            if len(xs):
                track.append((round(t, 1), int(m.sum()), int(xs.mean()), int(ys.mean())))
                last_img = r.plot(boxes=False)
                last_t = t
    if last_img is not None:
        cv2.putText(last_img, f"{label} last @ {last_t:.1f}s", (8, 36), 0, 0.9, (0, 255, 255), 2)
        cv2.imwrite(f"out/sam_{label.replace(' ', '_')}.jpg", last_img)
    summary[label] = track
    seen = [p[0] for p in track]
    print(f"{label:20} tracked in {len(track):3} frames, first {seen[0] if seen else '-'}s, last {seen[-1] if seen else '-'}s")
json.dump(summary, open("out/sam_tracks.json", "w"))
