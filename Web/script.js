// Robot Scientists landing page — minimal behavior, no framework needed.

const heroSection = document.getElementById('hero');
const modulesSection = document.getElementById('modules');
const scrollCue = document.querySelector('.scroll-cue');
const video = document.querySelector('.hero-video');

// Scroll-cue click scrolls straight to screen 2.
scrollCue?.addEventListener('click', () => {
  modulesSection.scrollIntoView({ behavior: 'smooth' });
});

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
const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        document.documentElement.dataset.scroll = entry.target.id;
      }
    });
  },
  { threshold: 0.5 }
);
sectionObserver.observe(heroSection);
sectionObserver.observe(modulesSection);
