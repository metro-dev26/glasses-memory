import { useEffect, useState } from "react";
import type { ChangeKind, ChangeMsg } from "../protocol";
import { elapsed } from "../time";

export interface RoundState {
  running: { id: number; name: string; started: number } | null;
  changes: ChangeMsg[];                       // alerts of the running round, newest first
  report: { id: number; changes: ChangeMsg[] } | null;
}

const KINDS: ChangeKind[] = ["missing", "new", "moved"];

export function Rounds({ round, onStart, onEnd, conn }: {
  round: RoundState;
  onStart: (name: string) => void;
  onEnd: () => void;
  conn: boolean;
}) {
  const [name, setName] = useState("");
  const [now, setNow] = useState(Date.now() / 1000);

  useEffect(() => {
    if (!round.running) return;
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, [round.running]);

  const r = round.running;
  return (
    <section className="panel rounds">
      <header>
        <h2>Rounds</h2>
        {r && <span className="mono rec"><i />{r.name} · {elapsed(now - r.started)}</span>}
      </header>

      {r ? (
        <div className="round-controls">
          <span className="dim">Walk the space. Changes since the last round appear here.</span>
          <button className="danger-outline" onClick={onEnd} disabled={!conn}>End round</button>
        </div>
      ) : (
        <form className="round-controls" onSubmit={(e) => { e.preventDefault(); onStart(name.trim()); setName(""); }}>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Round name (optional)" />
          <button className="primary" type="submit" disabled={!conn}>Start round</button>
        </form>
      )}

      {r && (round.changes.length
        ? <ul className="alerts">{round.changes.map((c) => <Alert key={`${c.kind}-${c.object_id}`} c={c} />)}</ul>
        : <p className="empty">No changes yet.</p>)}

      {!r && round.report && <Report id={round.report.id} changes={round.report.changes} />}
    </section>
  );
}

function Alert({ c }: { c: ChangeMsg }) {
  return (
    <li className={`alert kind-${c.kind}`}>
      <div className="alert-head">
        <span className="kind mono">{c.kind}</span>
        <b>{c.name}</b>
      </div>
      <div className="pair">
        <figure><img src={c.before} alt="Before" /><figcaption className="mono">before</figcaption></figure>
        <figure><img src={c.after} alt="After" /><figcaption className="mono">now</figcaption></figure>
      </div>
      <p>{c.text}</p>
    </li>
  );
}

function Report({ id, changes }: { id: number; changes: ChangeMsg[] }) {
  return (
    <div className="report">
      <div className="report-head">
        <b>Round {id} report</b>
        <span className="counts mono">
          {KINDS.map((k) => (
            <span key={k} className={`kind-${k}`}>{changes.filter((c) => c.kind === k).length} {k}</span>
          ))}
        </span>
      </div>
      {changes.length ? (
        <ul className="report-list">
          {changes.map((c) => (
            <li key={`${c.kind}-${c.object_id}`} className={`kind-${c.kind}`}>
              <span className="kind mono">{c.kind}</span><b>{c.name}</b><span className="dim">{c.text}</span>
            </li>
          ))}
        </ul>
      ) : <p className="empty">Nothing changed since the last round.</p>}
    </div>
  );
}
