import { useEffect, useRef, useState } from "react";
import { useSocket } from "../socket";

const SEND_FPS = 10;
const LONG_SIDE = 960;
const QUALITY = 0.7;
const MAX_BUFFERED = 512 * 1024; // bytes; above this the network is behind, so skip frames

/** The "glasses": rear camera, about 10 JPEG frames a second to /ws/camera. */
export function Phone() {
  const video = useRef<HTMLVideoElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rate, setRate] = useState(0);
  const [serverFps, setServerFps] = useState<number | null>(null);
  const [awake, setAwake] = useState(false);

  const { state, socket } = useSocket("/ws/camera", (data) => {
    if (typeof data !== "string") return;
    const msg = JSON.parse(data);
    if (msg.type === "camera_ack") setServerFps(msg.fps);
  });

  useEffect(() => {
    if (!streaming) return;
    let stream: MediaStream | null = null;
    let lock: WakeLockSentinel | null = null;
    let timer: number | undefined;
    let busy = false;
    let sent: number[] = [];
    const canvas = document.createElement("canvas");
    let stopped = false;

    const keepAwake = async () => {
      try {
        lock = await navigator.wakeLock?.request("screen");
        setAwake(!!lock);
        lock?.addEventListener("release", () => setAwake(false));
      } catch {
        setAwake(false); // not supported, or refused (e.g. battery saver)
      }
    };
    // The browser drops the wake lock when the page is hidden; take it again on return.
    const onVisible = () => {
      if (document.visibilityState === "visible") keepAwake();
    };

    const send = () => {
      const v = video.current, ws = socket.current;
      if (busy || !v || v.readyState < 2 || ws?.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > MAX_BUFFERED) return;
      const scale = Math.min(1, LONG_SIDE / Math.max(v.videoWidth, v.videoHeight));
      canvas.width = Math.round(v.videoWidth * scale);
      canvas.height = Math.round(v.videoHeight * scale);
      canvas.getContext("2d")!.drawImage(v, 0, 0, canvas.width, canvas.height);
      busy = true;
      canvas.toBlob((blob) => {
        busy = false;
        if (!blob || ws.readyState !== WebSocket.OPEN) return;
        ws.send(blob);
        const now = performance.now();
        sent = sent.filter((t) => now - t < 2000);
        sent.push(now);
        setRate(sent.length / 2);
      }, "image/jpeg", QUALITY);
    };

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (stopped) return stream.getTracks().forEach((t) => t.stop());
        video.current!.srcObject = stream;
        await video.current!.play();
        setError(null);
        keepAwake();
        document.addEventListener("visibilitychange", onVisible);
        timer = window.setInterval(send, 1000 / SEND_FPS);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStreaming(false);
      }
    })();

    return () => {
      stopped = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      stream?.getTracks().forEach((t) => t.stop());
      lock?.release().catch(() => {});
      setRate(0);
      setAwake(false);
    };
  }, [streaming, socket]);

  const secure = window.isSecureContext;
  return (
    <div className="phone">
      <video ref={video} className={`preview ${streaming ? "" : "off"}`} playsInline muted />
      <div className="phone-stats mono">
        <span className={`status status-${state}`}><i />{state === "open" ? "connected" : state}</span>
        <span>sending {rate.toFixed(1)} fps</span>
        {serverFps !== null && <span>server {serverFps.toFixed(1)} fps</span>}
        {streaming && <span>{awake ? "screen stays on" : "screen may sleep"}</span>}
      </div>
      {!secure && (
        <p className="phone-msg">The camera needs HTTPS. Open this page through the Tailscale https:// address.</p>
      )}
      {error && <p className="phone-msg">Camera error: {error}</p>}
      <button className={`big ${streaming ? "stop" : "start"}`} disabled={!secure}
              onClick={() => setStreaming((s) => !s)}>
        {streaming ? "Stop" : "Start"}
      </button>
    </div>
  );
}
