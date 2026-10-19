import { useEffect, useRef, useState } from "react";
import type { MemoryObject } from "../protocol";
import { idColor } from "../colors";
import { ago } from "../time";

const NEW_MS = 4000; // how long a newly remembered object keeps its "new" highlight

/** Everything remembered, as photos grouped by kind, like aisles in a shop.
 *  Tapping a photo asks where that object is; tapping its name renames it. */
export function MemoryPanel({ objects, onFind, onRename }: {
  objects: MemoryObject[];
  onFind: (id: number) => void;
  onRename: (id: number, name: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const [kind, setKind] = useState<string | null>(null);
  const [, tick] = useState(0);
  const firstSeen = useRef(new Map<number, number>());
  const loaded = useRef(false);

  // Remember when each id first appeared, so new tiles can be highlighted.
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
  const matches = objects.filter((o) => !q || o.name.toLowerCase().includes(q) || o.label.toLowerCase().includes(q));
  const groups = byKind(matches);
  const shown = kind && groups.has(kind) ? new Map([[kind, groups.get(kind)!]]) : groups;
  const live = objects.filter((o) => o.in_view).length;

  return (
    <section className="panel memory">
      <header>
        <h2>Memory</h2>
        <span className="mono dim">{objects.length} objects · {live} live</span>
      </header>
      <input className="search" placeholder="Filter by name" value={filter}
             onChange={(e) => setFilter(e.target.value)} />
      {groups.size > 1 && (
        <div className="kinds">
          <button className={`chip ${!kind ? "on" : ""}`} onClick={() => setKind(null)}>All</button>
          {[...groups].map(([k, list]) => (
            <button key={k} className={`chip ${kind === k ? "on" : ""}`} onClick={() => setKind(kind === k ? null : k)}>
              {k} <span className="dim">{list.length}</span>
            </button>
          ))}
        </div>
      )}
      <div className="aisles">
        {[...shown].map(([k, list]) => (
          <section key={k} className="aisle">
            <h3>{k} <span className="dim">{list.length}</span></h3>
            <ul className="tiles">
              {list.map((o) => (
                <Tile key={o.object_id} o={o} onFind={onFind} onRename={onRename}
                      fresh={Date.now() - (firstSeen.current.get(o.object_id) ?? 0) < NEW_MS} />
              ))}
            </ul>
          </section>
        ))}
        {!matches.length && <p className="empty">{objects.length ? "No match." : "Nothing remembered yet."}</p>}
      </div>
    </section>
  );
}

/** Objects grouped by the detector's label: groups seen most recently first,
 *  and inside each, what's in view now, then the most recently seen. */
function byKind(objects: MemoryObject[]) {
  const recent = (a: MemoryObject, b: MemoryObject) => Number(b.in_view) - Number(a.in_view) || b.last_seen - a.last_seen;
  const groups = new Map<string, MemoryObject[]>();
  for (const o of [...objects].sort(recent)) {
    if (!groups.has(o.label)) groups.set(o.label, []);
    groups.get(o.label)!.push(o);
  }
  return groups;
}

function Tile({ o, onFind, onRename, fresh }: {
  o: MemoryObject;
  onFind: (id: number) => void;
  onRename: (id: number, name: string) => void;
  fresh: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(o.name);

  const commit = () => {
    setEditing(false);
    const name = draft.trim();
    if (name && name !== o.name) onRename(o.object_id, name);
    else setDraft(o.name);
  };

  return (
    <li className={`tile ${fresh ? "fresh" : ""}`}>
      <button className="tile-photo" title={`Where is this ${o.name}?`} onClick={() => onFind(o.object_id)}
              style={{ borderColor: idColor(o.object_id) }}>
        <img src={o.thumb} alt={o.name} />
        {o.in_view && <span className="live-dot" title="In view now">live</span>}
      </button>
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
      <span className="tile-sub mono">#{o.object_id} · {o.in_view ? "in view" : ago(o.last_seen)}</span>
      {o.where && <span className="tile-where" title={o.where}>{o.where}</span>}
    </li>
  );
}
