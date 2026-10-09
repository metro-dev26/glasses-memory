import { useCallback, useEffect, useMemo, useState } from "react";
import type { AnswerMsg, ClientMsg, MemoryObject, ServerMsg } from "../protocol";
import { useSocket } from "../socket";
import { FrameBus, LiveView } from "./LiveView";
import { MemoryPanel } from "./MemoryPanel";
import { AskBar } from "./AskBar";
import { Rounds, type RoundState } from "./Rounds";
import { Forget } from "./Forget";

const ASK_TIMEOUT_MS = 10000;

export function App() {
  const bus = useMemo(() => new FrameBus(), []);
  const [objects, setObjects] = useState<MemoryObject[]>([]);
  const [answer, setAnswer] = useState<AnswerMsg | null>(null);
  const [asking, setAsking] = useState(false);
  const [round, setRound] = useState<RoundState>({ running: null, changes: [], report: null });

  const onMessage = useCallback((data: string | ArrayBuffer) => {
    if (typeof data !== "string") return;
    const msg = JSON.parse(data) as ServerMsg;
    switch (msg.type) {
      case "frame":
        bus.push(msg);
        break;
      case "memory":
        setObjects(msg.objects);
        break;
      case "answer":
        setAnswer(msg);
        setAsking(false);
        break;
      case "round":
        setRound((r) => msg.state === "running"
          ? { running: { id: msg.round_id, name: msg.name, started: msg.started },
              changes: r.running?.id === msg.round_id ? r.changes : [], report: null }
          : { running: null, changes: [], report: { id: msg.round_id, changes: msg.report } });
        break;
      case "change":
        setRound((r) => r.running?.id === msg.round_id ? { ...r, changes: [msg, ...r.changes] } : r);
        break;
    }
  }, [bus]);

  const { state, send } = useSocket("/ws/dashboard", onMessage);
  const open = state === "open";
  const post = useCallback((m: ClientMsg) => send(JSON.stringify(m)), [send]);

  // If no answer arrives (server restarted mid-question), stop showing "Searching".
  useEffect(() => {
    if (!asking) return;
    const t = window.setTimeout(() => setAsking(false), ASK_TIMEOUT_MS);
    return () => window.clearTimeout(t);
  }, [asking]);

  const ask = (query: string) => {
    if (post({ type: "ask", query })) setAsking(true);
  };
  const askPhoto = (jpeg_b64: string) => {
    if (post({ type: "ask_photo", jpeg_b64 })) setAsking(true);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" />
          <b>glasses-memory</b>
          <span className="dim">live</span>
        </div>
        <div className="topbar-right">
          <span className={`status status-${state}`}>
            <i />{state === "open" ? "Connected" : state === "connecting" ? "Connecting…" : "Disconnected, retrying"}
          </span>
          <Forget disabled={!open} onForget={() => { post({ type: "forget_all" }); setAnswer(null); }} />
        </div>
      </header>
      <main className="grid">
        <LiveView bus={bus} conn={state} />
        <aside className="side">
          <MemoryPanel objects={objects} onRename={(object_id, name) => post({ type: "rename", object_id, name })} />
          <AskBar answer={answer} pending={asking} onAsk={ask} onAskPhoto={askPhoto} />
          <Rounds round={round} conn={open}
                  onStart={(name) => post({ type: "round_start", name })}
                  onEnd={() => post({ type: "round_end" })} />
        </aside>
      </main>
    </div>
  );
}
