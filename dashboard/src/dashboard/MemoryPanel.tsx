import { useEffect, useRef, useState } from "react";
import type { MemoryObject } from "../protocol";
import { idColor } from "../colors";
import { ago } from "../time";

const NEW_MS = 4000; // how long a newly remembered object keeps its "new" highlight

export function MemoryPanel({ objects, onRename }: {
  objects: MemoryObject[];
  onRename: (id: number, name: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const [, tick] = useState(0);
  const firstSeen = useRef(new Map<number, number>());
  const loaded = useRef(false);

  // Remember when each id first appeared, so new rows can be highlighted.
  // Objects present in the very first memory message are not "new".
  useEffect(() => {
    const now = loaded.current ? Date.now() : 0;
    for (const o of objects) if (!firstSeen.current.has(o.object_id)) firstSeen.current.set(o.object_id, now);
    if (objects.length) loaded.current = true;
  }, [objects]);

  useEffect(() => {
    const t = window.setInterval(() => tick((n) => n + 1), 5000); // keep "2m ago" current
    return () => window.clearInterval(t);
  }, []);

  const q = filter.trim().toLowerCase();
  const rows = objects
    .filter((o) => !q || o.name.toLowerCase().includes(q) || o.label.toLowerCase().includes(q))
    .sort((a, b) => (firstSeen.current.get(b.object_id) ?? 0) - (firstSeen.current.get(a.object_id) ?? 0)
      || b.last_seen - a.last_seen);
  const live = objects.filter((o) => o.in_view).length;

  return (
    <section className="panel memory">
      <header>
        <h2>Memory</h2>
        <span className="mono dim">{objects.length} objects · {live} live</span>
      </header>
      <input className="search" placeholder="Filter by name" value={filter}
             onChange={(e) => setFilter(e.target.value)} />
      <ul className="rows">
        {rows.map((o) => (
          <Row key={o.object_id} o={o} onRename={onRename}
               fresh={Date.now() - (firstSeen.current.get(o.object_id) ?? 0) < NEW_MS} />
        ))}
        {!rows.length && <li className="empty">{objects.length ? "No match." : "Nothing remembered yet."}</li>}
      </ul>
    </section>
  );
}

function Row({ o, onRename, fresh }: { o: MemoryObject; onRename: (id: number, name: string) => void; fresh: boolean }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(o.name);

  const commit = () => {
    setEditing(false);
    const name = draft.trim();
    if (name && name !== o.name) onRename(o.object_id, name);
    else setDraft(o.name);
  };

  return (
    <li className={`row ${fresh ? "fresh" : ""}`}>
      <img className="thumb" src={o.thumb} alt="" style={{ borderColor: idColor(o.object_id) }} />
      <div className="row-main">
        <div className="row-top">
          {editing ? (
            <input className="rename" autoFocus value={draft}
                   onChange={(e) => setDraft(e.target.value)} onBlur={commit}
                   onKeyDown={(e) => {
                     if (e.key === "Enter") commit();
                     if (e.key === "Escape") { setDraft(o.name); setEditing(false); }
                   }} />
          ) : (
            <button className="name" title="Click to rename" onClick={() => { setDraft(o.name); setEditing(true); }}>
              {o.name}
            </button>
          )}
          {o.in_view && <span className="live-dot" title="In view now">live</span>}
        </div>
        <div className="row-sub mono">
          <span>{o.label}</span>
          <span>#{o.object_id}</span>
          <span>{o.in_view ? "in view" : `last seen ${ago(o.last_seen)}`}</span>
        </div>
        {o.where && <div className="row-where">{o.where}</div>}
      </div>
    </li>
  );
}
