# Live server

FastAPI app that joins the phone, the engine and the dashboard
(protocol: `docs/dashboard-brief.md` section 4).

```bash
cd dashboard && npm install && npm run build && cd ..   # the pages it serves
python -m server.app --store store/live                 # http://localhost:8000
```

- Only the newest camera frame is processed; older ones are dropped, so the
  delay stays at one frame instead of growing into a queue.
- Memory is saved every 10 s and on shutdown (`Ctrl+C`).
- Rounds have a name and a clock but no change detection yet (phase 3):
  `round_end` sends an empty report and `/keyframe/` answers 404.
- The log prints frames per second and server latency every 30 s.
- It answers only to localhost and `*.ts.net` host names (add others with
  `--allow-host`), and websockets only from its own pages: otherwise a web
  page open on the laptop could reach it through localhost.

## Phone camera over Tailscale (HTTPS)

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
