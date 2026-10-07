"""Build memory/index.html: ask "where's my X?" and see where it was left.

Runs fully offline (open the file in a browser). The memory is embedded in
the page because browsers block file:// pages from reading other files.

    python ask_page.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
MEMORY = ROOT / "memory"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glasses Memory</title>
<style>
  :root { --bg:#0e1013; --card:#171a1f; --line:#262b33; --text:#e8eaed; --dim:#8b929c; --accent:#f5d90a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:720px; margin:0 auto; padding:40px 16px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--dim); margin:0 0 24px; font-size:14px; }
  form { display:flex; gap:8px; }
  input { flex:1; background:var(--card); border:1px solid var(--line); color:var(--text);
          padding:14px 16px; border-radius:10px; font-size:17px; outline:none; }
  input:focus { border-color:var(--accent); }
  button { background:var(--accent); color:#111; border:0; border-radius:10px; padding:0 18px;
           font-weight:600; font-size:15px; cursor:pointer; }
  .chips { margin:12px 0 28px; display:flex; flex-wrap:wrap; gap:6px; }
  .chip { background:none; color:var(--dim); border:1px solid var(--line); padding:4px 10px;
          border-radius:99px; font-size:13px; font-weight:400; }
  .chip:hover { color:var(--text); border-color:var(--dim); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  .answer { padding:18px 20px; }
  .answer b { color:var(--accent); }
  .note { color:var(--dim); font-size:13px; margin-top:6px; }
  .media { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
  .media img, .media video { width:100%; aspect-ratio:9/16; object-fit:cover; background:#000; display:block; }
  .label { font-size:12px; color:var(--dim); padding:8px 20px 0; }
  .history { padding:12px 20px 18px; font-size:14px; color:var(--dim); }
  .history li { margin:2px 0; }
  .miss { color:var(--dim); padding:18px 20px; }
</style></head>
<body><main>
  <h1>Glasses Memory</h1>
  <p class="sub">Ask where you left something. Answers come from what the camera saw.</p>
  <form id="f"><input id="q" placeholder="where's my remote?" autofocus><button>Ask</button></form>
  <div class="chips" id="chips"></div>
  <div id="out"></div>
</main>
<script>
const MEMORY = __MEMORY__;
const VIDEO = "../" + MEMORY.video;
const STOP = new Set("where where's wheres is are my the a did i leave left put keep kept last see seen saw".split(" "));
const names = Object.keys(MEMORY.objects);

function find(query) {
  const words = query.toLowerCase().replace(/[^a-z ]/g, " ").split(/\\s+/).filter(w => w && !STOP.has(w));
  let best = null, bestScore = 0;
  for (const name of names) {
    const score = words.filter(w => name.includes(w) || name.split(" ").some(n => w.includes(n))).length;
    if (score > bestScore) { best = name; bestScore = score; }
  }
  return best;
}

function fmt(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0"); return `${m}:${s}`; }

function answer(query) {
  const out = document.getElementById("out");
  const name = find(query);
  if (!name) { out.innerHTML = `<div class="card miss">I don't remember anything like that. I know: ${names.join(", ")}.</div>`; return; }
  const eps = MEMORY.objects[name];
  if (!eps.length) { out.innerHTML = `<div class="card miss">I was shown the ${name}, but never saw it clearly after that.</div>`; return; }
  const last = eps[0];
  const from = Math.max(0, last.rest - 4);
  out.innerHTML = `
    <div class="card">
      <div class="answer"><b>${name}</b>: last seen <b>${fmt(last.rest)}</b> into the recording from ${MEMORY.recorded}.
        <div class="note">Last known, not current: it may have moved since.</div></div>
      <div class="media">
        <div><div class="label">where it was left</div><img src="${last.snapshot}" alt="${name}"></div>
        <div><div class="label">the moment (plays from ${fmt(from)})</div>
          <video id="v" src="${VIDEO}#t=${from},${last.rest + 1}" controls muted playsinline></video></div>
      </div>
      <ul class="history">${eps.map((e, i) => `<li>${fmt(e.start)} – ${fmt(e.end)}${i ? "" : " · last seen"}</li>`).join("")}</ul>
    </div>`;
  document.getElementById("v").play().catch(() => {});
}

document.getElementById("chips").innerHTML = names.map(n => `<button class="chip">${n}</button>`).join("");
document.querySelectorAll(".chip").forEach(c => c.onclick = () => { q.value = `where's my ${c.textContent}?`; answer(q.value); });
document.getElementById("f").onsubmit = e => { e.preventDefault(); answer(q.value); };
</script></body></html>
"""

if __name__ == "__main__":
    memory = json.loads((MEMORY / "memory.json").read_text())
    (MEMORY / "index.html").write_text(PAGE.replace("__MEMORY__", json.dumps(memory)))
    print(f"wrote {MEMORY / 'index.html'}")
