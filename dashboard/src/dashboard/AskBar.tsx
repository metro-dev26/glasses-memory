import { useEffect, useRef, useState } from "react";
import type { AnswerMsg } from "../protocol";
import { clock, elapsed } from "../time";

// SpeechRecognition is still prefixed in Chrome and missing in Firefox.
type Recognition = {
  lang: string;
  interimResults: boolean;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start(): void;
  stop(): void;
};
const SpeechRecognitionImpl: (new () => Recognition) | undefined =
  (window as unknown as Record<string, new () => Recognition>).SpeechRecognition ??
  (window as unknown as Record<string, new () => Recognition>).webkitSpeechRecognition;

const PHOTO_SIDE = 960; // same size the phone sends

/** Say `text` in the server's Kokoro voice (server/voice.py), or with the
 *  browser's own speech when that service is not running. Returns a stop function. */
function speak(text: string) {
  let stopped = false;
  let audio: HTMLAudioElement | null = null;
  fetch("/speak", { method: "POST", body: text })
    .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
    .then((wav) => {
      if (stopped) return;
      const url = URL.createObjectURL(wav);
      audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      void audio.play();
    })
    .catch(() => {
      if (stopped || !("speechSynthesis" in window)) return;
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    });
  return () => {
    stopped = true;
    audio?.pause();
    if ("speechSynthesis" in window) speechSynthesis.cancel();
  };
}

export function AskBar({ answer, pending, onAsk, onAskPhoto }: {
  answer: AnswerMsg | null;
  pending: boolean;
  onAsk: (query: string) => void;
  onAskPhoto: (jpegB64: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [listening, setListening] = useState(false);
  const [muted, setMuted] = useState(() => readMuted());
  const recognition = useRef<Recognition | null>(null);
  const file = useRef<HTMLInputElement>(null);

  // Speak each new answer, unless muted; a newer answer cuts off the old one.
  useEffect(() => {
    if (!answer || muted) return;
    return speak(answer.text);
  }, [answer, muted]);

  const submit = (q: string) => {
    if (q.trim()) onAsk(q.trim());
  };

  const listen = () => {
    if (!SpeechRecognitionImpl) return;
    if (listening) {
      recognition.current?.stop();
      return;
    }
    const r = new SpeechRecognitionImpl();
    r.lang = "en-US";
    r.interimResults = false;
    r.onresult = (e) => {
      const text = e.results[0][0].transcript;
      setQuery(text);
      submit(text);
    };
    r.onend = r.onerror = () => setListening(false);
    recognition.current = r;
    setListening(true);
    r.start();
  };

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    if (next && "speechSynthesis" in window) speechSynthesis.cancel();
    try {
      localStorage.setItem("gm-muted", next ? "1" : "0");
    } catch {
      // storage can be blocked; the toggle still works for this visit
    }
  };

  const pickPhoto = async (f: File | undefined) => {
    if (!f) return;
    onAskPhoto(await toJpegB64(f));
    if (file.current) file.current.value = "";
  };

  return (
    <section className="panel ask">
      <header>
        <h2>Ask</h2>
        <button className="ghost small" onClick={toggleMute} title="Speak answers aloud">
          {muted ? "voice off" : "voice on"}
        </button>
      </header>
      <form className="ask-row" onSubmit={(e) => { e.preventDefault(); submit(query); }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Where are my keys?" />
        {SpeechRecognitionImpl && (
          <button type="button" className={`icon ${listening ? "on" : ""}`} onClick={listen}
                  title={listening ? "Stop listening" : "Ask by voice"} aria-label="Ask by voice">
            <MicIcon />
          </button>
        )}
        <button type="button" className="icon" onClick={() => file.current?.click()}
                title="Ask by photo" aria-label="Ask by photo">
          <PhotoIcon />
        </button>
        <button type="submit" className="primary">Ask</button>
        <input ref={file} type="file" accept="image/*" hidden onChange={(e) => pickPhoto(e.target.files?.[0])} />
      </form>
      {pending && <div className="answer loading mono dim">Searching memory…</div>}
      {!pending && answer && <AnswerCard a={answer} onAsk={onAsk} />}
    </section>
  );
}

function AnswerCard({ a, onAsk }: { a: AnswerMsg; onAsk: (q: string) => void }) {
  if (!a.found) {
    return (
      <div className="answer">
        <p className="answer-text">{a.text}</p>
        {a.candidates.length > 0 && (
          <div className="chips">
            <span className="dim">Closest:</span>
            {a.candidates.map((c) => <button key={c} className="chip" onClick={() => onAsk(c)}>{c}</button>)}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="answer">
      <div className="answer-body">
        {a.snapshot && <Snapshot src={a.snapshot} />}
        <div className="answer-info">
          <p className="answer-text">{a.text}</p>
          {a.nearby.length > 0 && (
            <div className="chips">
              <span className="dim">Near</span>
              {a.nearby.map((n) => <span key={n} className="chip">{n}</span>)}
            </div>
          )}
          <Timeline a={a} />
        </div>
      </div>
    </div>
  );
}

/** The snapshot with the object's box drawn in the snapshot's own pixel space. */
/** The saved snapshot, which already has the object's box drawn in (yellow) at
 *  the moment it was taken. A box drawn over it here would be the object's
 *  latest box, wrong if it moved after the snapshot. Click: full screen. */
function Snapshot({ src }: { src: string }) {
  const [full, setFull] = useState(false);

  // Escape closes the full-screen view.
  useEffect(() => {
    if (!full) return;
    const close = (e: KeyboardEvent) => e.key === "Escape" && setFull(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [full]);

  const picture = <img src={src} alt="Where it was last seen" />;
  return (
    <>
      <button className="snapshot" title="Click to see it full screen" onClick={() => setFull(true)}>
        {picture}
      </button>
      {full && (
        <div className="lightbox" onClick={() => setFull(false)}>
          <div className="snapshot">{picture}</div>
        </div>
      )}
    </>
  );
}

/** Past sightings on one line from the first to now, newest at the right. */
function Timeline({ a }: { a: AnswerMsg }) {
  if (!a.sightings.length) return null;
  const now = Date.now() / 1000;
  const t0 = Math.min(...a.sightings.map((s) => s.start));
  const span = Math.max(1, now - t0);
  return (
    <div className="timeline">
      <div className="track">
        {a.sightings.map((s, i) => (
          <span key={i} className="seg" title={`${clock(s.start)}–${clock(s.end)} · ${s.where}`}
                style={{ left: `${((s.start - t0) / span) * 100}%`,
                         width: `max(3px, ${((s.end - s.start) / span) * 100}%)` }} />
        ))}
      </div>
      <div className="track-ends mono dim">
        <span>{clock(t0)}</span>
        <span>now</span>
      </div>
      <ol className="sightings mono">
        {[...a.sightings].reverse().map((s, i) => (
          <li key={i}><span>{clock(s.start)}</span><span>{elapsed(s.end - s.start)}</span><span>{s.where}</span></li>
        ))}
      </ol>
    </div>
  );
}

function readMuted() {
  try {
    return localStorage.getItem("gm-muted") === "1";
  } catch {
    return false;
  }
}

async function toJpegB64(f: File): Promise<string> {
  const bitmap = await createImageBitmap(f);
  const scale = Math.min(1, PHOTO_SIDE / Math.max(bitmap.width, bitmap.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bitmap.width * scale);
  c.height = Math.round(bitmap.height * scale);
  c.getContext("2d")!.drawImage(bitmap, 0, 0, c.width, c.height);
  bitmap.close();
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" strokeLinecap="round" />
    </svg>
  );
}

function PhotoIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="6" width="18" height="14" rx="2" />
      <circle cx="12" cy="13" r="3.5" />
      <path d="M8.5 6l1.5-2.5h4L15.5 6" strokeLinejoin="round" />
    </svg>
  );
}
