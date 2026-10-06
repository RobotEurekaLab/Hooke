# Technische Universität Wien (TU Wien) — Room Photo Research Notes

Facility: USTEM (University Service Centre for Transmission Electron Microscopy), Room 057-02, Freihaus building, Stadionallee 2

## Search queries tried
- `TU Wien USTEM Freihaus electron microscopy lab photo Stöger-Pollach room 057`

## URLs checked
- https://www.ustem.tuwien.ac.at/searchtem_gallery/EN/ — 301-redirects to https://www.tuwien.at/ustem (the old subdomain's specific gallery sub-path has been consolidated/lost in a site migration).
- https://www.ustem.tuwien.ac.at/courses_and_conferences/scientific_conferences/lab_opening/EN/ — also 301-redirects to the same generic https://www.tuwien.at/ustem landing page; the specific "lab opening" sub-page content is no longer separately reachable.
- https://www.tuwien.at/ustem — fetched via WebFetch (AI-processed) and separately downloaded raw via curl for direct inspection.
  - **WebFetch's processed summary** described several promising-sounding images present on the page: "USTEM Team im Stiegenhaus des Freihauses" (team photo in the Freihaus stairwell), "Gebäudefront des USTEM Neubaus" (building front of the new USTEM building), "Forscher am Elektronenmikroskop" (researcher at the electron microscope), "Mitarbeiter von USTEM im Gespräch" (staff in discussion), plus historical items (an old brass Reichert microscope, a 1942 Siemens & Halske ÜM100, a 2000-era TECNAI F20, a chalkboard with equations).
  - **Direct raw-HTML inspection (curl + grep)** of the same URL found the page is a JavaScript single-page application: the static HTML contains no real `<img>` src URLs for any of the above (only base64 placeholder GIFs and one unrelated accessibility-badge logo). A second explicit WebFetch asking for raw `<img>` tag src attributes confirmed this — only lazy-loading placeholders are present in the fetchable markup.
  - This means the descriptive image list above is very likely genuine page content (an "about us / history" section of USTEM's site, which plausibly would include exactly these team/building/historical photos) but the actual image file URLs are not extractable with the current toolset (WebFetch/curl), since they only load after client-side JavaScript execution.
- Attempted to check archive.org (Wayback Machine) for a cached, pre-migration snapshot of the old ustem.tuwien.ac.at gallery/lab_opening pages, which might retain working image URLs — both the WebFetch tool (explicitly refuses web.archive.org) and a direct `archive.org/wayback/available` API query (returned HTTP 429 Too Many Requests) failed to produce results in this pass.

## What was found
No directly downloadable/verifiable room photo, despite strong circumstantial evidence that real interior/team/building photos exist on USTEM's own current website.

## Newly confirmed facts beyond existing card
- Strong indirect evidence (via page-content description) that USTEM's site includes a researcher-at-the-microscope photo and a team photo in the Freihaus stairwell, plus historical equipment/building photos spanning from 1942 to the present — but none independently verified as actual image files in this pass.

## Confidence tier
**Location confirmed, no room photo** (not independently verified — the task's bar requires actually viewing/downloading a confirmed image, which was not achieved here due to the page's JS-only image loading). **Recommended follow-up:** a future pass with a JS-capable browser/headless-rendering tool (rather than WebFetch/curl) could very likely recover real interior photos from https://www.tuwien.at/ustem directly, and/or a retry of the Wayback Machine lookup once rate-limiting clears.
