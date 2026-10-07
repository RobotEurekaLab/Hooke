# Chalmers University of Technology -- Nanofabrication Laboratory (Myfab Chalmers, MC2 department)

Confidence tier: **room photo confirmed**

## Search queries tried
- "Chalmers University of Technology core facility lab interior photo cleanroom nanofabrication"
- "Chalmers Nanofabrication Laboratory MC2 cleanroom photo site:chalmers.se"

## URLs checked
- https://www.chalmers.se/en/infrastructure/myfab-chalmers/facility/ -- WebFetch confirmed a photo captioned "Researchers inside the cleanroom" exists, but couldn't resolve the actual asset URL (wrapped in Next.js image optimization). Re-fetched the raw HTML via curl with a browser User-Agent and extracted the underlying CMS image URL directly from the `srcset`/`_next/image` query parameters: `https://cms.www.chalmers.se/Media/lkyphtfs/9242_st301-209b_1920x1080px.jpg`. Downloaded via the `_next/image` proxy URL (direct `cms.www.chalmers.se` host timed out; `/api/media/` endpoint 404'd; the `_next/image/?url=...` proxy succeeded).
- https://www.chalmers.se/en/departments/mc2/research/nanofabrication/ (referenced, not independently fetched -- facility page already gave what was needed).

## What the downloaded photo shows
`cleanroom.jpg`: A bright, white-lit cleanroom bay (not the yellow photolithography-safe lighting seen in some other university cleanrooms -- this area evidently handles non-photoresist-sensitive steps). Two researchers in full white cleanroom bunny suits (hood, suit, gloves) seated at a perforated stainless-steel vibration-isolation table, handling silicon wafers with tweezers next to small dishes/trays. To the left: a large "FHR" branded sputtering/thin-film deposition system (model "MS 150x4-L") with a control monitor showing a process-status GUI, plus a separate vacuum-chamber tool with safety interlocks and warning labels. White gridded ceiling with embedded fluorescent panel lighting and a traffic-light-style status indicator. A glass partition wall is visible in the background separating this bay from another room with more benches and monitors. Epoxy/vinyl speckled floor.

## Downloaded files
- `cleanroom.jpg` (1920x1080, full cleanroom bay interior)

## Newly confirmed facts beyond the existing card
- The cleanroom is 1240 m² of classified cleanroom area, in continuous operation since 2001, run by the Dept. of Microtechnology and Nanoscience (MC2) as part of the national Myfab university-cleanroom network, and open to external academic/commercial users (quantum computing and wireless-communication component fabrication mentioned as current research areas).
- Specific equipment visible/named in the photo: FHR MS 150x4-L sputter deposition system.
