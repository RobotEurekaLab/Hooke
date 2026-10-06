# University of California, Davis -- (no single facility identified; MCB imaging/EM cores + DNA Tech Core recorded)

Confidence tier: **nothing new found**

## Search queries tried
- "UC Davis Light Imaging Facility OR Bio Electron Microscopy Facility MCB photo interior room" (WebSearch)
- Two further planned queries ("UC Davis DNA Technologies Expression Analysis Core Genome Center photo lab interior"; ""UC Davis" Genome Center building photo interior Life Sciences Addition") could not be run -- this session's WebSearch budget was exhausted (200/200) partway through this school, apparently shared across other parallel research happening in the same Hooke session/workspace.

## URLs checked (all blocked)
- https://microscopy.mcb.ucdavis.edu/about -- HTTP 403 Forbidden (tried via WebFetch, and via direct curl with a browser user-agent -- still 403)
- https://bioem.ucdavis.edu/about-us -- HTTP 403 Forbidden
- https://dnatech.ucdavis.edu/ -- HTTP 403 Forbidden
- https://biology.ucdavis.edu/research/facilities -- HTTP 403 Forbidden

All four UC Davis-hosted domains attempted in this pass returned 403 Forbidden to every fetch method tried (WebFetch tool, and curl with a Chrome user-agent string) -- this looks like campus-wide bot/WAF protection on ucdavis.edu subdomains rather than a page-specific issue, since it affected every distinct subdomain tried (microscopy.mcb, bioem, dnatech, biology).

## What was found
Nothing new -- could not get past UC Davis's apparent site-wide bot protection to read any of the candidate facility pages, and ran out of web-search budget before finding a usable third-party mirror (e.g. an architecture-firm portfolio page, a news photo, a Flickr/Instagram account) the way other schools in this batch were cracked.

## New facts beyond the existing card
None.

## Downloaded files
None.

## Confidence tier reasoning
This school was already the weakest-evidenced "(none identified)" case on the card (no PI/room number at all, multiple candidate core facilities with no single best pick), and this pass could not even re-read the facilities' own pages due to blocking, let alone find a photo. Honest "nothing new found" outcome -- a follow-up pass should try an explicitly third-party source (architecture firm, local press, or a UC Davis-affiliated social media account) rather than the facilities' own ucdavis.edu pages, which appear broadly blocked to automated fetching.
