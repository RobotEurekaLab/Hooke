# Universidad de los Andes — Room Photo Research Notes

Facility: Centro de Microscopía — MicroCore, Laboratorio B101-B102, Bloque B

## Search queries tried
- `Universidad de los Andes Centro de Microscopia MicroCore B101 foto laboratorio interior`

## URLs checked
- https://corefacilities.uniandes.edu.co/microcore/ — fetched directly. Found: a logo/branding set, individual equipment product photos for all four named instruments (Olympus FV1000 confocal, Tescan Lyra 3 FIB-SEM, Tescan Vega 4 SEM, Asylum Research MFP-3D-BIO AFM), staff headshot thumbnails, a workflow diagram, and a "banner" image plus several "resultados" (results) images. The banner image (`microcore-corefacilities-vic-uniandes-b.jpg`) was downloaded and visually inspected — it is a purely graphic/abstract green-and-black branding pattern with the μ·core logo, not a photo of any kind. The "resultados" images (0–5) were not individually downloaded, as their filenames and page context (a "results" section) strongly suggest they are sample micrographs produced by the instruments (e.g. example SEM/confocal output images) rather than room photos — consistent with the equipment-only pattern seen on the rest of the page.
- https://wwwpre.uniandes.edu.co/es/noticias/ciencias-biologicas/la-cabeza-de-una-mosca-vista-desde-el-laboratorio-de-microscopia-de-los-andes (a news article, "13 imágenes sorprendentes captadas por los microscopios de Los Andes") — WebFetch failed with a DNS error (`getaddrinfo ENOTFOUND wwwpre.uniandes.edu.co`); the `wwwpre` subdomain appears to not resolve. Not retried on the main `uniandes.edu.co` domain this pass.
- https://www.uniandes.edu.co/es/noticias/ingenieria/colombia-cuenta-por-primera-vez-con-un-equipo-dual-beam-microscopio-de-barrido-de-electrones-de-alta-resolucion (dual-beam SEM news article, already noted in existing card) — not fetched this pass.

## What was found
No genuine room-level photo. MicroCore's own page is thorough on equipment and staff but contains no interior/room photography.

## Newly confirmed facts beyond existing card
None confirmed. (The "13 imágenes sorprendentes" article title strongly suggests its images are microscope-output photos — e.g. "head of a fly" — rather than room photos, so it was deprioritized once the subdomain failed to resolve; a retry on the corrected domain, or fetching the dual-beam SEM announcement article, remain the most promising untried leads.)

## Confidence tier
**Nothing new found.** Equipment and staff are thoroughly documented on the facility's own page, but no room photo was found. Two unexplored news-article leads remain (the "13 imágenes" piece, likely on the main uniandes.edu.co domain rather than wwwpre, and the dual-beam SEM announcement) — either could plausibly contain an incidental room photo alongside the microscope-output images, and would be the natural next step.
