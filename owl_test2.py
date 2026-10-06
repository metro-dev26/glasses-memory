"""OWLv2 one-shot search, with the query picked by a box on the full close-up frame
(the OWL paper's protocol) instead of a padded crop."""
import cv2, torch
from PIL import Image
from torchvision.ops import box_iou
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(MODEL)
model = Owlv2ForObjectDetection.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
cap = cv2.VideoCapture("videos/sample3.mp4")

def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    return Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

@torch.no_grad()
def features(img):
    px = proc(images=img, return_tensors="pt")["pixel_values"].to("cuda").half()
    fmap = model.image_embedder(pixel_values=px)[0]
    b, h, w, d = fmap.shape
    return fmap.reshape(b, h * w, d), fmap

@torch.no_grad()
def query_embedding(t, box):
    img = frame_at(t)
    side = max(img.size)                       # processor pads to a square
    feats, fmap = features(img)
    boxes = model.box_predictor(feats, fmap)[0]              # cx,cy,w,h in 0..1
    corners = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)
    target = torch.tensor([box], device="cuda", dtype=corners.dtype) / side
    best = box_iou(target.float(), corners.float())[0].argmax()
    _, class_embeds = model.class_predictor(feats)
    return class_embeds[0, best][None, None]                 # [1,1,dim]

@torch.no_grad()
def best_logit(q, t):
    feats, _ = features(frame_at(t))
    logits, _ = model.class_predictor(feats, q)
    return logits.max().item()

refs = {  # close-up time, box (x1,y1,x2,y2) in the 480x854 frame
    "white deodorant can": (20, (100, 140, 360, 780)),
    "yellow stapler":      (16, (90, 220, 480, 500)),
    "steel stapler":       (12, (110, 250, 340, 740)),
    "bracelet":            (4,  (140, 0, 280, 854)),
    "red pen":             (14, (180, 30, 330, 770)),
}
cases = {
    "white deodorant can": ([32], [60, 66]),
    "yellow stapler":      ([60], [64, 68]),
    "steel stapler":       ([66, 68], [32, 60]),
    "bracelet":            ([66, 48], [32, 60]),
    "red pen":             ([42, 44], [60, 66]),
}
for lab, (pos, neg) in cases.items():
    q = query_embedding(*refs[lab])
    p = "  ".join(f"{t}s={best_logit(q, t):.1f}" for t in pos)
    n = "  ".join(f"{t}s={best_logit(q, t):.1f}" for t in neg)
    print(f"{lab:20} PRESENT {p:24} ABSENT {n}")
