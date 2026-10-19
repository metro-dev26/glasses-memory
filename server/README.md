# Live server

FastAPI app that joins the phone, the engine and the dashboard
(protocol: `docs/dashboard-brief.md` section 4).

```bash
cd dashboard && npm install && npm run build && cd ..   # the pages it serves
python -m server.app --store store/live                 # http://localhost:8000
```

**Recorded video (the current demo mode):** play a video file through the same
engine and dashboard, at real speed and 10 fps, as if the phone were sending it:

```bash
python -m server.app --video videos/x.mp4 --loop       # --loop replays it; objects come back re-identified
```

- Only the newest camera frame is processed; older ones are dropped, so the
  delay stays at one frame instead of growing into a queue.
- Memory is saved every 10 s and on shutdown (`Ctrl+C`).
- **Safety rounds:** Start round, walk the whole space, End round. Each round
  is compared with the one before: **missing** (seen last round, not this one),
  **new**, and **moved** (on a different surface, or a missing and a new object
  with the same unique label: re-ID often turns a moved object into a new one).
  Rounds are stored in the store's `memory.db` (`rounds`, `round_seen`), so a
  round from yesterday is still there today. Logic: `engine/rounds.py`.
- **Recording:** the Record button on the dashboard or the phone page saves
  every processed frame to `videos/rec-<date>-<time>/` and, when stopped, an
  H.264 `.mp4` beside it (needs `ffmpeg`). The folder replays exactly with
  `--video FOLDER`. `--record FOLDER` starts recording from launch.
- A page that stops reading (a stalled phone or tab) is dropped after 2 s and
  reconnects by itself, so it cannot freeze the video for everyone else.
- The log prints frames per second and server latency every 30 s.
- It answers only to localhost and this machine's own Tailscale name (add
  others with `GM_ALLOW_HOSTS=name1,name2`), and websockets only from its own
  pages: otherwise a web page open on the laptop could reach it through localhost.

## Phone camera over Tailscale (HTTPS), on hold

Live from the phone is paused while we demo on recorded videos; this is kept
for when it comes back.

Phone browsers only allow the camera on HTTPS. Tailscale gives the laptop an
HTTPS address inside the tailnet. One-time setup:

1. Tailscale admin console -> DNS: enable **MagicDNS** and **HTTPS Certificates**.
2. On the laptop, once: `sudo tailscale set --operator=$USER` (lets `tailscale serve` run without sudo).
3. Install Tailscale on the phone and log in to the same tailnet. Anyone else
   who should open the dashboard joins the tailnet or gets the machine shared.

Each session:

```bash
python -m server.app --store store/live
tailscale serve --bg 8000          # https://<machine>.<tailnet>.ts.net -> localhost:8000
tailscale serve status             # shows the address
```

Phone: open `https://<machine>.<tailnet>.ts.net/phone`. Dashboard: the same address.
Stop serving with `tailscale serve --https=443 off`.

## Measuring end-to-end latency (glass to glass)

Open the dashboard with `?clock`: a millisecond clock appears in the corner.
Point the phone at the laptop screen. The dashboard then shows the clock twice,
live and inside the video; the difference is the full phone -> server ->
dashboard delay. Take several screenshots and read both numbers in each.
