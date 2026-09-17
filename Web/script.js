// Robot Scientists landing page — minimal behavior, no framework needed.

const heroSection = document.getElementById('hero');
const modulesSection = document.getElementById('modules');
const scrollCue = document.querySelector('.scroll-cue');
const video = document.querySelector('.hero-video');
const soundToggle = document.querySelector('.sound-toggle');
const iconMuted = soundToggle?.querySelector('.icon-muted');
const iconUnmuted = soundToggle?.querySelector('.icon-unmuted');

// Scroll-cue click scrolls straight to screen 2.
scrollCue?.addEventListener('click', () => {
  modulesSection.scrollIntoView({ behavior: 'smooth' });
});

// The video must start muted or every major browser refuses to autoplay
// it at all. Give visitors an explicit, one-click way to turn sound on
// instead (autoplay-with-sound has no reliable cross-browser trigger).
function setMuted(muted) {
  if (!video) return;
  video.muted = muted;
  soundToggle?.setAttribute('aria-pressed', String(!muted));
  soundToggle?.setAttribute('aria-label', muted ? 'Turn sound on' : 'Turn sound off');
  if (iconMuted) iconMuted.hidden = !muted;
  if (iconUnmuted) iconUnmuted.hidden = muted;
}

soundToggle?.addEventListener('click', () => setMuted(!video.muted));

// Autoplay can be blocked until a user gesture on some mobile browsers;
// retry once on first interaction so the hero never gets stuck on a
// frozen frame.
function tryPlay() {
  video?.play().catch(() => {});
}
tryPlay();
window.addEventListener('pointerdown', tryPlay, { once: true });
window.addEventListener('scroll', tryPlay, { once: true, passive: true });

// Fade the headline/cards in once screen 2 actually enters view, rather
// than on page load (most visitors won't have scrolled there yet).
const revealTargets = document.querySelectorAll('.headline, .subhead, .cards');
revealTargets.forEach((el) => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(18px)';
  el.style.transition = 'opacity .7s ease, transform .7s ease';
});

const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      revealTargets.forEach((el, i) => {
        setTimeout(() => {
          el.style.opacity = '1';
          el.style.transform = 'translateY(0)';
        }, i * 90);
      });
      io.disconnect();
    });
  },
  { threshold: 0.35 }
);
io.observe(modulesSection);

// data-scroll attribute on <html> for any future section-aware styling
// (e.g. hiding the top nav's background over the video vs. screen 2).
// Also auto-mutes the still-looping background video the moment screen 2
// comes into view, so sound a visitor turned on doesn't keep playing
// once the video itself is out of sight.
const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      document.documentElement.dataset.scroll = entry.target.id;
      if (entry.target.id === 'modules' && video && !video.muted) {
        setMuted(true);
      }
    });
  },
  { threshold: 0.5 }
);
sectionObserver.observe(heroSection);
sectionObserver.observe(modulesSection);
