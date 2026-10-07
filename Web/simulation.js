// Simulation showcase page — static gallery, no framework needed.
// Video paths are relative to Web/, resolving into ../docs/assets/ where
// the actual preview clips live (checked into the repo, so they load the
// same whether served from GitHub Pages or a local http.server).

const REAL_LAB = [
  { school: 'Queensland University of Technology', qs: 213, file: 'queensland-university-of-technology-qut',
    detail: 'Rebuilt object-for-object from the fit-out contractor’s own two interior photos of this exact room.' },
  { school: 'Western University', qs: 120, file: 'western-university',
    detail: 'Rebuilt from the lab group’s own equipment photos — Experion, NanoDrop, Droplet Digital PCR.' },
  { school: 'University of Wisconsin–Madison', qs: 116, file: 'university-of-wisconsin-madison',
    detail: 'Wood cabinet benches and overhead steel shelving, from the architecture firm’s own interior photo.' },
  { school: 'Queen Mary University of London', qs: 120, file: 'queen-mary-university-of-london',
    detail: 'Blue fume-hood cabinets and a triple stainless wash station, from the Blizard Institute’s own photo.' },
  { school: 'University of Southern California', qs: 125, file: 'university-of-southern-california',
    detail: 'A night-city-view window and wood cabinet benches, from the architecture firm’s own interior photo.' },
  { school: 'University College Dublin', qs: 126, file: 'university-college-dublin',
    detail: 'A wall of wash-basins and a window onto trees, from the Conway Institute’s own interior photo.' },
  { school: 'University of St Andrews', qs: 104, file: 'university-of-st-andrews',
    detail: 'Pale-blue walls and blue carpet, from the UK NMR network’s own interior photos of this suite.' },
  { school: 'Technical University of Denmark', qs: 109, file: 'technical-university-of-denmark',
    detail: 'White cleanroom shelving and a stainless workstation, from DTU Nanolab’s own site photos.' },
  { school: 'Rice University', qs: 141, file: 'rice-university',
    detail: 'Deep amber photolithography safe-light, from Rice Magazine’s own cleanroom feature.' },
  { school: 'University of Groningen', qs: 159, file: 'university-of-groningen',
    detail: 'The building’s signature orange vinyl floor, from the Linnaeusborg facility’s own photos.' },
  { school: 'Eindhoven University of Technology', qs: 136, file: 'eindhoven-university-of-technology',
    detail: 'A yellow-lit cleanroom window, from the TU/e Microfab lab’s own interior photo.' },
  { school: 'Universidad de Chile', qs: 139, file: 'universidad-de-chile',
    detail: 'Blue cabinets and distilled-water carboys, from the department’s own interior photo.' },
];

const SPACE = [
  { label: 'Lunar', slug: 'lunar' },
  { label: 'Martian', slug: 'martian' },
  { label: 'Orbital', slug: 'orbital' },
];

const MICRO = [
  { file: 'microscopy-cell-manipulation', title: 'Grasp and transfer',
    detail: 'Independently driven fine forceps lift and relocate a 36 µm cell with <3 µm placement error.' },
  { file: 'microscopy-cell-injection', title: 'Volume-controlled injection',
    detail: 'Membrane puncture and 0.5 pL delivery into a 36 × 28 × 8 µm adherent cell.' },
  { file: 'microscopy-cell-pushing', title: 'Cell pushing',
    detail: 'A 4 µm blunt probe holds ≥50 ms contact within 3 µm of the target, ≤4.5 µm compression.' },
];

function el(tag, className, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children || []) node.append(child);
  return node;
}

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function hoverVideo(src) {
  const video = document.createElement('video');
  video.src = src;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  // Every clip here is well under 100KB, so playing all of them at once
  // (rather than gating on hover) costs nothing and reads as a living
  // gallery instead of a grid of black boxes. Honor reduced-motion by
  // holding on the first frame instead, with click-to-play as an escape
  // hatch either way.
  video.preload = reduceMotion ? 'auto' : 'metadata';
  video.autoplay = !reduceMotion;
  video.addEventListener('click', () => (video.paused ? video.play() : video.pause()));
  return video;
}

function buildRealLabGrid() {
  const grid = document.getElementById('reallab-grid');
  if (!grid) return;
  for (const lab of REAL_LAB) {
    const card = el('a', 'video-card', []);
    card.href = `https://github.com/RobotEurekaLab/Hooke/blob/main/real_lab/labs/${lab.file}.md`;
    card.target = '_blank';
    card.rel = 'noopener';
    const frame = el('div', 'video-frame', [hoverVideo(`../docs/assets/real-lab-${lab.file}-mujoco.mp4`)]);
    const qs = el('span', 'qs-badge', []);
    qs.textContent = `QS ${lab.qs}`;
    frame.append(qs);
    const body = el('div', 'video-card-body', []);
    const h3 = el('h3', null, []); h3.textContent = lab.school;
    const p = el('p', null, []); p.textContent = lab.detail;
    body.append(h3, p);
    card.append(frame, body);
    grid.append(card);
  }
}

function buildSpaceGrid() {
  const grid = document.getElementById('space-grid');
  if (!grid) return;
  for (const world of SPACE) {
    const row = el('div', 'pair-row', []);
    const label = el('div', 'pair-label', []);
    label.textContent = world.label;
    row.append(label);
    for (const backend of ['mujoco', 'isaac']) {
      const col = el('div', 'pair-col', []);
      const video = hoverVideo(`../docs/assets/space-experiment-${world.slug}-sample_transfer-${backend}.mp4`);
      const tag = el('span', 'backend-tag', []);
      tag.textContent = backend === 'mujoco' ? 'MuJoCo' : 'Isaac Sim';
      col.append(video, tag);
      row.append(col);
    }
    grid.append(row);
  }
}

function buildMicroGrid() {
  const grid = document.getElementById('micro-grid');
  if (!grid) return;
  for (const shot of MICRO) {
    const card = el('div', 'shot-card', []);
    const img = document.createElement('img');
    img.src = `../docs/assets/${shot.file}.png`;
    img.loading = 'lazy';
    img.alt = shot.title;
    const body = el('div', 'shot-card-body', []);
    const h3 = el('h3', null, []); h3.textContent = shot.title;
    const p = el('p', null, []); p.textContent = shot.detail;
    body.append(h3, p);
    card.append(img, body);
    grid.append(card);
  }
}

buildRealLabGrid();
buildSpaceGrid();
buildMicroGrid();

// Fade each section in as it enters view, same restrained reveal used on
// the landing page's screen 2.
const revealTargets = document.querySelectorAll('.sim-section, .sim-hero, .sim-closing');
revealTargets.forEach((sectionEl) => {
  sectionEl.style.opacity = '0';
  sectionEl.style.transform = 'translateY(22px)';
  sectionEl.style.transition = 'opacity .7s ease, transform .7s ease';
});
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      io.unobserve(entry.target);
    });
  },
  { threshold: 0.12 }
);
revealTargets.forEach((sectionEl) => io.observe(sectionEl));
