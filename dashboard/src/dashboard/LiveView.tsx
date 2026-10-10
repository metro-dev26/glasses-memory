import { useEffect, useRef, useState } from "react";
import type { Box, Detection, FrameMsg } from "../protocol";
import type { ConnState } from "../socket";
import { idColor } from "../colors";
import { agoSeconds } from "../time";

const GLIDE_MS = 120;      // boxes ease to the new position over about one frame interval
const REID_MS = 2500;      // how long "seen 3m ago" stays on a pill
const DIM_BELOW = 0.4;     // detections under this confidence are drawn dimmer and dashed
const STALE_MS = 2000;     // no frame for this long = the camera has stopped

// `/?clock` shows a millisecond clock for measuring glass-to-glass latency:
// point the phone at this screen, and the clock seen inside the video lags
// the live clock by exactly the end-to-end delay.
const SHOW_CLOCK = new URLSearchParams(location.search).has("clock");

/** Frames go straight from the socket to the canvas, outside React state:
 *  ten re-renders a second of the whole page would be wasted work. */
export class FrameBus {
  private listener: ((f: FrameMsg) => void) | null = null;
  push(f: FrameMsg) {
    this.listener?.(f);
  }
  listen(fn: (f: FrameMsg) => void) {
    this.listener = fn;
    return () => {
      this.listener = null;
    };
  }
}

interface Glide {
  from: Box;
  to: Box;
  start: number;
  det: Detection;
}

interface Hud {
  fps: number;
  latency: number;
  inView: number;
  inMemory: number;
  lastFrame: number;
}

export function LiveView({ bus, conn }: { bus: FrameBus; conn: ConnState }) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const clockEl = useRef<HTMLDivElement>(null);
  const [hud, setHud] = useState<Hud | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const el = canvas.current!;
    const ctx = el.getContext("2d")!;
    let image: ImageBitmap | null = null;
    let size = { w: 1, h: 1 };
    let glides = new Map<number, Glide>();
    const reid = new Map<number, { text: string; until: number }>();
    let decoding = false;
    let pending: FrameMsg | null = null;
    let lastHud = 0;
    let raf = 0;

    const resize = () => {
      const r = wrap.current!.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      el.width = Math.round(r.width * dpr);
      el.height = Math.round(r.height * dpr);
      el.style.width = `${r.width}px`;
      el.style.height = `${r.height}px`;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(wrap.current!);

    // Only the newest frame is decoded; frames that arrive mid-decode are skipped.
    const accept = async (f: FrameMsg) => {
      if (decoding) {
        pending = f;
        return;
      }
      decoding = true;
      try {
        const bytes = Uint8Array.from(atob(f.jpeg_b64), (c) => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
        image?.close();
        image = bitmap;
        size = { w: f.width, h: f.height };
        glides = nextGlides(glides, f.detections, performance.now());
        for (const d of f.detections) {
          if (d.reidentified && d.last_seen_ago_s !== undefined) {
            reid.set(d.track_id, { text: `seen ${agoSeconds(d.last_seen_ago_s)}`, until: performance.now() + REID_MS });
          }
        }
        if (performance.now() - lastHud > 250) {
          lastHud = performance.now();
          setHud({ fps: f.stats.fps, latency: f.stats.latency_ms, inView: f.stats.in_view,
                   inMemory: f.stats.in_memory, lastFrame: Date.now() });
        }
      } catch {
        // a corrupt frame is dropped; the next one replaces it
      }
      decoding = false;
      if (pending) {
        const next = pending;
        pending = null;
        accept(next);
      }
    };
    const unlisten = bus.listen(accept);

    const draw = (t: number) => {
      raf = requestAnimationFrame(draw);
      if (clockEl.current) clockEl.current.textContent = msClock();
      const W = el.width, H = el.height;
      ctx.clearRect(0, 0, W, H);
      if (!image) return;
      const dpr = window.devicePixelRatio || 1;
      const scale = Math.min(W / size.w, H / size.h);
      const ox = (W - size.w * scale) / 2, oy = (H - size.h * scale) / 2;
      ctx.drawImage(image, ox, oy, size.w * scale, size.h * scale);

      for (const g of glides.values()) {
        const k = easeOut(Math.min(1, (t - g.start) / GLIDE_MS));
        const b = g.from.map((v, i) => v + (g.to[i] - v) * k) as Box;
        const [x1, y1, x2, y2] = [ox + b[0] * scale, oy + b[1] * scale, ox + b[2] * scale, oy + b[3] * scale];
        const flash = reid.get(g.det.track_id);
        if (flash && flash.until < t) reid.delete(g.det.track_id);
        drawBox(ctx, x1, y1, x2, y2, g.det, dpr, flash && flash.until >= t ? flash.text : null);
      }
    };
    raf = requestAnimationFrame(draw);

    const tick = window.setInterval(() => setNow(Date.now()), 500);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      unlisten();
      window.clearInterval(tick);
      image?.close();
    };
  }, [bus]);

  const stale = !hud || now - hud.lastFrame > STALE_MS;
  return (
    <div className="live" ref={wrap}>
      <canvas ref={canvas} />
      {SHOW_CLOCK && <div className="ms-clock mono" ref={clockEl} />}
      <div className="hud mono">
        <div><span>conn</span><b className={`conn-${conn}`}>{conn}</b></div>
        <div><span>fps</span><b>{hud && !stale ? hud.fps.toFixed(1) : "–"}</b></div>
        <div><span>latency</span><b>{hud && !stale ? `${hud.latency} ms` : "–"}</b></div>
        <div><span>in view</span><b>{hud && !stale ? hud.inView : "–"}</b></div>
        <div><span>memory</span><b>{hud ? hud.inMemory : "–"}</b></div>
      </div>
      {conn !== "open" ? (
        <div className="live-msg">
          <b>Disconnected from the server</b>
          <span>Reconnecting automatically…</span>
        </div>
      ) : stale ? (
        <div className="live-msg">
          <b>Waiting for the camera</b>
          <span>Open /phone on the phone and press Start.</span>
        </div>
      ) : null}
    </div>
  );
}

/** Each track glides from wherever it is drawn now to its new box.
 *  A track seen for the first time starts at its box, with no glide. */
function nextGlides(old: Map<number, Glide>, dets: Detection[], t: number) {
  const next = new Map<number, Glide>();
  for (const d of dets) {
    const g = old.get(d.track_id);
    let from = d.box;
    if (g) {
      const k = easeOut(Math.min(1, (t - g.start) / GLIDE_MS));
      from = g.from.map((v, i) => v + (g.to[i] - v) * k) as Box;
    }
    next.set(d.track_id, { from, to: d.box, start: t, det: d });
  }
  return next;
}

function msClock() {
  const d = new Date();
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function easeOut(k: number) {
  return 1 - (1 - k) ** 3;
}

function drawBox(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number,
                 d: Detection, dpr: number, flash: string | null) {
  const color = idColor(d.object_id);
  const low = d.conf < DIM_BELOW;
  const w = x2 - x1, h = y2 - y1;
  ctx.save();
  ctx.globalAlpha = low ? 0.55 : 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 * dpr;

  // a faint full outline, with the corners drawn solid
  ctx.setLineDash(low ? [4 * dpr, 4 * dpr] : []);
  ctx.globalAlpha *= 0.45;
  ctx.strokeRect(x1, y1, w, h);
  ctx.globalAlpha = low ? 0.55 : 1;
  ctx.setLineDash([]);
  const c = Math.min(14 * dpr, w / 3, h / 3);
  ctx.beginPath();
  for (const [x, y, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
    ctx.moveTo(x, y + dy * c);
    ctx.lineTo(x, y);
    ctx.lineTo(x + dx * c, y);
  }
  ctx.stroke();

  // the label pill above the box
  const text = flash ?? `${d.name} · ${d.conf.toFixed(2)} · ${d.object_id === null ? "…" : `#${d.object_id}`}`;
  ctx.font = `${11 * dpr}px "JetBrains Mono", ui-monospace, monospace`;
  const pad = 5 * dpr, ph = 17 * dpr;
  const dot = flash ? 0 : 7 * dpr;
  const tw = ctx.measureText(text).width + pad * 2 + dot;
  const px = Math.max(0, Math.min(x1, ctx.canvas.width - tw));
  const py = y1 - ph - 3 * dpr < 0 ? y1 + 3 * dpr : y1 - ph - 3 * dpr;
  ctx.fillStyle = flash ? color : "rgba(12, 14, 17, 0.78)";
  ctx.beginPath();
  ctx.roundRect(px, py, tw, ph, ph / 2);
  ctx.fill();
  if (!flash) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px + pad + 2 * dpr, py + ph / 2, 2.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = flash ? "#0c0e11" : "#e9edf2";
  ctx.textBaseline = "middle";
  ctx.fillText(text, px + pad + dot, py + ph / 2 + 0.5 * dpr);
  ctx.restore();
}
