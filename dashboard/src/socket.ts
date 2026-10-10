import { useCallback, useEffect, useRef, useState } from "react";

export type ConnState = "connecting" | "open" | "closed";

const RETRY_MS = [500, 1000, 2000, 4000]; // then every 4 s

/** ws(s)://<this host><path>, so the page works on localhost and over Tailscale HTTPS. */
export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

/**
 * A websocket that reconnects by itself. `onMessage` gets every message;
 * it is read through a ref so a new callback never tears the socket down.
 */
export function useSocket(path: string, onMessage: (data: string | ArrayBuffer) => void) {
  const [state, setState] = useState<ConnState>("connecting");
  const socket = useRef<WebSocket | null>(null);
  const handler = useRef(onMessage);
  handler.current = onMessage;

  useEffect(() => {
    let attempt = 0;
    let timer: number | undefined;
    let stopped = false;

    const connect = () => {
      setState("connecting");
      const ws = new WebSocket(wsUrl(path));
      ws.binaryType = "arraybuffer";
      socket.current = ws;
      ws.onopen = () => {
        attempt = 0;
        setState("open");
      };
      ws.onmessage = (e) => handler.current(e.data);
      ws.onclose = () => {
        socket.current = null;
        if (stopped) return;
        setState("closed");
        timer = window.setTimeout(connect, RETRY_MS[Math.min(attempt++, RETRY_MS.length - 1)]);
      };
    };
    connect();

    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket.current?.close();
    };
  }, [path]);

  const send = useCallback((data: string | Blob | ArrayBuffer) => {
    const ws = socket.current;
    if (ws?.readyState !== WebSocket.OPEN) return false;
    ws.send(data);
    return true;
  }, []);

  return { state, send, socket };
}
