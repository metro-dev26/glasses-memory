"""Quick check: does OWLv2 image-guided detection find each object in its final spot?"""
import cv2, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(MODEL)
model = Owlv2ForObjectDetection.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

cap = cv2.VideoCapture("videos/sample3.mp4")
def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    return Image.fromarray(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

def best_score(query_img, target_img):
    inputs = proc(images=target_img, query_images=query_img, return_tensors="pt").to("cuda")
    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.image_guided_detection(**inputs)
    return out.logits.max().item()  # raw logit; sigmoid saturates

cases = {  # object: (frames where it IS visible, frames where it is NOT)
    "white deodorant can": ([32], [60, 66]),
    "yellow stapler":      ([60], [64, 68]),
    "steel stapler":       ([66, 68], [32, 60]),
    "bracelet":            ([66, 48], [32, 60]),
    "red pen":             ([42, 44], [60, 66]),
}
for lab, (pos, neg) in cases.items():
    q = Image.open(f"refs_crops/{lab}.jpg").convert("RGB")
    p = [f"{t}s={best_score(q, frame_at(t)):.1f}" for t in pos]
    n = [f"{t}s={best_score(q, frame_at(t)):.1f}" for t in neg]
    print(f"{lab:20} PRESENT {'  '.join(p):24} ABSENT {'  '.join(n)}")
print("vram MB", torch.cuda.max_memory_allocated() // 2**20)
