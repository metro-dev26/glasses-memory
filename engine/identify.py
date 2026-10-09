"""Identity fingerprints: is this crop the same object we saw before?

DINOv3 is trained without labels to tell images apart, so its features
separate *this* remote from *a* remote better than CLIP did in version 1
(CLIP put the wrong sunglasses at 0.74).
"""
import cv2
import numpy as np
import timm
import torch

MODEL = "vit_small_patch16_dinov3.lvd1689m"   # timm's copy; Meta's HF repo is gated
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD = 0.1                # grow each box by 10% so the crop keeps the object's edges

# Calibrated on videos/sample3.mp4 only, by eval/calibrate_identity.py: at these
# values 70% of returning tracks were re-identified with 7% wrong merges. Higher
# values trade re-identifications for fewer merges; a wrong merge gives a wrong
# "where" answer, a missed one only a duplicate object, so we lean high.
MATCH = 0.70             # best gallery similarity must reach this to be the same object
MARGIN = 0.05            # ...and beat the second-best object by this much
NEW_VIEW = 0.80          # a fingerprint less similar than this to the gallery is a new viewpoint
GALLERY_SIZE = 12        # views kept per object


class Embedder:
    def __init__(self, model=MODEL):
        # global_pool="token" returns the CLS token. timm's default averages the patch
        # tokens, which gives every crop a large shared component: on sample3 two
        # different objects then scored a median 0.64, against 0.28 with CLS.
        self.model = timm.create_model(model, pretrained=True, num_classes=0,
                                       global_pool="token").cuda().half().eval()

    @torch.inference_mode()
    def __call__(self, frame, boxes):
        """One unit-length fingerprint per box, shape (len(boxes), dim)."""
        if not boxes:
            return np.zeros((0, self.model.num_features), dtype=np.float32)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crops = [cv2.resize(crop(rgb, box), (SIZE, SIZE)) for box in boxes]
        batch = (np.stack(crops).astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).cuda().half()
        feats = torch.nn.functional.normalize(self.model(x).float(), dim=1)
        return feats.cpu().numpy()


def crop(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px) + 1), min(h, int(y2 + py) + 1)
    return image[y1:y2, x1:x2]


def best_match(fingerprint, galleries, exclude=()):
    """Return (object_id, similarity) of the matching object, or (None, best) if
    nothing matches clearly. An object's score is its closest gallery view, since
    one view from the same side is enough. `exclude` holds objects visible right
    now under another track: one object cannot be in two places at once.

    The margin is measured only against runners-up that look different from the
    winner. A runner-up that looks like the winner is usually the same object
    stored twice, and counting it would turn every split into more splits (on
    sample3, 20 of 116 new objects were created that way)."""
    scores = sorted(((float((g @ fingerprint).max()), oid) for oid, g in galleries.items()
                     if oid not in exclude), reverse=True)
    if not scores:
        return None, 0.0
    best, oid = scores[0]
    rivals = [s for s, other in scores[1:]
              if float((galleries[oid] @ galleries[other].T).max()) < MATCH]
    second = rivals[0] if rivals else 0.0
    if best >= MATCH and best - second >= MARGIN:
        return oid, best
    return None, best


def add_view(gallery, fingerprint):
    """Add a fingerprint to an object's gallery if it shows a new viewpoint."""
    if len(gallery) and (gallery @ fingerprint).max() >= NEW_VIEW:
        return gallery
    gallery = np.vstack([gallery, fingerprint[None]]) if len(gallery) else fingerprint[None]
    return gallery[-GALLERY_SIZE:]
