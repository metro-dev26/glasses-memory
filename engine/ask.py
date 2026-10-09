"""Answer "where's my X?" from memory, by text or by photo.

Text is matched to object names with a sentence embedding, so "shades" finds
"sunglasses" (0.71) without any keyword list. It is not magic: "clicker" scores
0.42 against "remote control", below the threshold.
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from engine import identify

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ASK_MIN = 0.50           # on a hand-made word list, absent things scored <= 0.43 and
                         # synonyms 0.71-0.93; unmeasured on real questions
TIE = 0.05               # names this close to the best are ties: answer the most recent
STOP = {"where", "wheres", "where's", "is", "are", "my", "the", "a", "an", "did", "i",
        "leave", "left", "put", "last", "see", "saw", "seen", "find", "can't", "cant",
        "you", "what", "about", "please", "it"}

_text_model = None


def embed_text(texts):
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL, device="cpu")
    return _text_model.encode(texts, normalize_embeddings=True)


def by_text(memory, query, now):
    words = [w for w in re.findall(r"[a-z']+", query.lower()) if w not in STOP]
    objects = list(memory.objects.values())
    if not words or not objects:
        return not_found(query, "I don't remember anything yet." if not objects else
                         "What should I look for?", [])
    q = embed_text([" ".join(words)])[0]
    names = embed_text([o.name for o in objects])
    labels = embed_text([o.label for o in objects])
    scores = np.maximum(names @ q, labels @ q)
    best = float(scores.max())
    if best < ASK_MIN:
        closest = [objects[i].name for i in np.argsort(-scores)[:3]]
        return not_found(query, f"I don't remember seeing {' '.join(words)}.", closest)
    # The same thing may be stored twice (a missed re-identification): answer
    # with whichever of the near-equal matches was seen last.
    ties = [o for o, s in zip(objects, scores) if s >= best - TIE]
    return answer(query, max(ties, key=lambda o: o.last_seen), now)


def by_photo(memory, detector, embedder, image, now):
    """Match a photo of the object by its DINOv3 fingerprint."""
    boxes = detector.boxes(image)
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else \
        [0, 0, image.shape[1], image.shape[0]]
    f = embedder(image, [box])[0]
    galleries = {oid: o.gallery for oid, o in memory.objects.items()}
    scored = sorted(((float((g @ f).max()), oid) for oid, g in galleries.items()), reverse=True)
    if not scored or scored[0][0] < identify.MATCH:
        closest = [memory.objects[oid].name for _, oid in scored[:3]]
        return not_found("photo", "I don't remember seeing that.", closest)
    return answer("photo", memory.objects[scored[0][1]], now)


def answer(query, obj, now):
    """The protocol's answer message for a found object: the last sighting that
    has a snapshot is where it was left."""
    shown = [s for s in obj.sightings if s.snapshot_at >= 0]
    s = shown[-1] if shown else obj.sightings[-1]
    where = f" {s.where}" if s.where else ""
    return {"type": "answer", "query": query, "found": True, "object_id": obj.id,
            "text": f"Last seen {ago(now - s.end)}{where}.",
            "snapshot": f"/snapshot/{s.id}.jpg" if shown else None,
            "box": s.box, "nearby": s.nearby, "seen_at": s.snapshot_at,
            "sightings": [{"start": x.start, "end": x.end, "where": x.where}
                          for x in obj.sightings[-5:]],
            "candidates": []}


def not_found(query, text, candidates):
    return {"type": "answer", "query": query, "found": False, "object_id": None,
            "text": text, "snapshot": None, "box": None, "nearby": [], "sightings": [],
            "candidates": candidates}


def ago(seconds):
    seconds = max(0, round(seconds))
    if seconds < 10:
        return "just now"
    if seconds < 90:
        return f"{seconds} seconds ago"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes ago"
    return f"{round(seconds / 3600)} hours ago"
