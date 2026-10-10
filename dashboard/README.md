# Dashboard and phone camera page

The UI for live glasses-memory, built from `docs/dashboard-brief.md`.
Vite + React + TypeScript, no UI kit, one stylesheet (`src/styles.css`).

| Page | Where | What it does |
|---|---|---|
| `/` | laptop, screen-shared | live view with boxes, memory, ask, rounds |
| `/phone` | phone browser | rear camera, ~10 JPEG frames/s to `/ws/camera` |

## Run against the mock server

```bash
cd dashboard
npm install
npm run build          # type-check + build into dist/
npm run mock           # serves dist/ and the protocol on http://localhost:8000
```

Open http://localhost:8000 for the dashboard and http://localhost:8000/phone
for the camera page (localhost counts as secure, so the camera works there;
on a real phone it needs the Tailscale HTTPS address).

While editing, `npm run dev` serves with hot reload on :5173 and forwards
`/ws`, `/thumb`, `/snapshot` and `/keyframe` to the mock on :8000.

The mock (`mock/mock_server.py`) needs `fastapi`, `uvicorn` and `pillow`: activate
the repo venv first (`source venv/bin/activate`, or `venv\Scripts\activate` on
Windows), since `npm run mock` runs whichever `python` is on the PATH.

## Files

- `src/protocol.ts`: the message types from the brief, section 4
- `src/socket.ts`: websocket that reconnects by itself
- `src/dashboard/LiveView.tsx`: canvas overlay (boxes glide, colour by object id, HUD)
- `src/dashboard/MemoryPanel.tsx`, `AskBar.tsx`, `Rounds.tsx`, `Forget.tsx`: the right column
- `src/phone/Phone.tsx`: camera capture, JPEG encoding, wake lock
- `mock/mock_server.py`: a drawn scene with 8 fake objects; the keys leave and come back re-identified
