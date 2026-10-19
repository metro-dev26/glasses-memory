"""Speak answers in a Kokoro voice, as a small service of its own.

Kokoro needs transformers < 4.48 with torch 2.5.1, while the engine's text
model (sentence-transformers) needs a newer one, so the voice runs in a
separate venv that has `kokoro` and `soundfile`:

    python -m server.voice                  # http://127.0.0.1:8010, POST /speak -> WAV

The live server forwards the dashboard's POST /speak here (GM_VOICE_URL).
When this is not running, the dashboard uses the browser's own speech.
"""
import io
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf
from kokoro import KPipeline

VOICE = "af_heart"       # Kokoro's warm American English voice
SPEED = 0.95
RATE = 24000             # Kokoro's output sample rate
PORT = 8010
MAX_TEXT = 500           # characters; answers are one sentence

pipeline = KPipeline(lang_code="a")   # "a" = American English


def speak(text):
    """`text` spoken, as WAV bytes."""
    chunks = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
              for _, _, a in pipeline(text, voice=VOICE, speed=SPEED)]
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks) if chunks else np.zeros(1, np.float32), RATE, format="WAV")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/speak":
            self.send_error(404)
            return
        size = min(int(self.headers.get("Content-Length") or 0), MAX_TEXT * 4)
        text = self.rfile.read(size).decode("utf-8", "replace")[:MAX_TEXT].strip()
        if not text:
            self.send_error(400, "nothing to say")
            return
        wav = speak(text)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


if __name__ == "__main__":
    speak("ready")       # load the voice now, so the first answer is not slow
    print(f"[voice] {VOICE} on http://127.0.0.1:{PORT}/speak", flush=True)
    # One request at a time: the model is not shared between threads.
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
