# Web

The public "Robot Scientists" landing page — a static, two-screen teaser
site, separate from `Hooke/webui/` (the internal task-picker interface
used to run experiments).

- **Screen 1**: `Hooke/media/teaser/video/Robot_Scientists_Teaser_web.mp4`
  full-bleed, autoplay/muted/loop (referenced in place, not copied here).
- **Screen 2**: the "Robot Scientists" headline and four module cards
  (Simulation, Space & Surface, Robot Learning, Physical World), each
  linking out to the relevant part of the GitHub repo, over a field of
  slowly-drifting lab/space line icons.

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

Card links point at real files (`docs/space_experiments.md`,
`openpi/README.md`, etc.) via their GitHub blob URLs rather than
relative paths, so they render correctly regardless of where `Web/`
itself ends up hosted. Update them if any of those docs move.
