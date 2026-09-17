# Web

The public "Robot Scientists" landing page — a static, two-screen teaser
site, separate from `Hooke/webui/` (the internal task-picker interface
used to run experiments).

- **Screen 1**: `Hooke/media/teaser/video/Robot_Scientists_Teaser_web.mp4`
  full-bleed, autoplay/muted/loop (referenced in place, not copied here).
- **Screen 2**: the "Robot Scientists" headline and two module cards
  (Simulation, Physical World) over a field of slowly-drifting lab/space
  line icons. Physical World links to the GitHub repo; Simulation links
  to a live, running instance of `Hooke/webui/` (see below).

No build step — plain HTML/CSS/JS.

## Deploying

Serve via GitHub Pages with the source set to the repo root (Settings →
Pages → Deploy from a branch → `/ (root)`). The whole repository is then
published as static files, so `index.html`'s relative reference to
`../Hooke/media/teaser/video/...` resolves correctly — the site actually
lives at `https://<org>.github.io/Hooke/Web/`.

A tiny redirect stub at the **repo root** (`/index.html`, not this one)
forwards `https://<org>.github.io/Hooke/` straight to `Web/index.html`,
so that's the link worth sharing. The address bar updates to the `/Web/`
URL right after the instant redirect.

## Preview locally

From the **repo root** (not this folder, so the relative video path
still resolves the same way it will on Pages):

```bash
python3 -m http.server 8099
# then open http://localhost:8099/Web/
```

## Module links

The Physical World card points at a real doc
(`docs/isaac_completion_plan.md`) via its GitHub blob URL rather than a
relative path, so it renders correctly regardless of where `Web/` itself
ends up hosted.

The Simulation card points at a **live, running instance** of
`Hooke/webui/`, not a doc — see below.

## Exposing the live simulator (Simulation card)

The Simulation card links to whatever `http://localhost:8080` (or
wherever `Hooke/webui/server.py` is running) is currently exposed as
publicly. `scripts/start-public-tunnel.sh` automates opening a
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
to it:

```bash
# 1. Start the webui itself (from the Hooke/ simulator directory):
cd Hooke && export MUJOCO_GL=egl && python -m webui.server --host 0.0.0.0 --port 8080

# 2. In another terminal, from the repo root:
Web/scripts/start-public-tunnel.sh 8080
```

The script prints the resulting `https://<random-words>.trycloudflare.com`
URL. **This is not a permanent address** — every restart of the tunnel
gets a brand-new random hostname, since a quick tunnel needs no
Cloudflare account and therefore can't reserve a stable name for you.
When it changes, update the Simulation card's `href` in `index.html` to
match and push.

A genuinely permanent hostname (survives restarts, human-chosen name)
needs binding one of your own accounts — a domain in Cloudflare for a
named tunnel, ngrok's free static domain, or Tailscale Funnel are the
usual options — none of which this script sets up, since that requires
your own account credentials/OAuth, not something to automate on your
behalf without you doing the signup step yourself.
