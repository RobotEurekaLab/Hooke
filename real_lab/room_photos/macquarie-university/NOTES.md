# Macquarie University -- no facility identified by prior research

Confidence tier: **location confirmed, no room photo** (access blocked)

## Search queries tried
- "Macquarie University core research facility lab interior photo microscopy"
- "Macquarie Analytical and Fabrication Facility microscopy unit lab photo people microscopes"
- "Macquarie University microscopy unit 14 Eastern Road lab photo" (not run -- WebSearch budget exhausted session-wide before this query could execute)

## URLs checked
- https://www.mq.edu.au/faculty-of-science-and-engineering/our-research/macquarie-analytical-and-fabrication-facility/microscopy-unit -- WebFetch returned HTTP 403 Forbidden. Direct curl with a browser User-Agent also returned HTTP 403 (site has bot/anti-scraping protection on mq.edu.au).
- https://www.mq.edu.au/research/research-centres-groups-and-facilities/facilities/macquarie-analytical-and-fabrication-facility/major-facilities -- same HTTP 403 Forbidden result.

## What was found
Search-result snippets (not page fetches, since the site blocks both WebFetch and curl) describe:
- **Macquarie Analytical and Fabrication Facility (MAFF)**, Microscopy Unit, located in the **Basement of 14 Eastern Road**, Macquarie University, NSW 2109.
- A search-engine-indexed snippet mentions the facility's own page includes a photo of "three people in lab coats looking into microscopes" on the MAFF major-facilities page -- this could not be independently verified since the page is inaccessible to this agent (403 on every fetch attempt).
- Equipment: UltraMicroscope BLAZE (3D imaging), JEOL 1400 TEM, WiTec AFM-Confocal Raman-TERS system, plus standard light/fluorescence/electron microscopy.

## Newly confirmed facts beyond "(none identified)"
- Candidate facility: Macquarie Analytical and Fabrication Facility (MAFF) -- Microscopy Unit, Basement, 14 Eastern Road, Macquarie University.

## Judgment call
mq.edu.au appears to have Cloudflare-or-similar bot protection blocking both the WebFetch tool and direct curl (even with a realistic browser User-Agent string) -- this is a hard access barrier, not a lack of material. A future pass with a real browser / different IP range might succeed where this one could not.
