# University of Otago -- Otago Micro and Nanoscale Imaging (OMNI)

Confidence tier: **location confirmed, no room photo**

## Search queries tried
- "Otago Micro and Nanoscale Imaging OMNI facility interior photo"
- (WebSearch budget exhausted mid-session for remaining follow-ups; relied on WebFetch against known
  otago.ac.nz/omni URLs from the existing card)

## URLs checked
- https://www.otago.ac.nz/omni/contacts/location-of-confocal-and-electron-microscopy-facilities --
  fetched successfully (text only, no photos): confirms OMNI's confocal (Room B01g) and electron
  microscopy (Room B10) units are in the basement of the Lindo Ferguson Building, Great King Street,
  Dunedin, opposite Dunedin Public Hospital -- same facts as existing card, reconfirmed. Page gives a
  walking-directions-style description of basement corridor signage ("Electron Microscopy, Confocal
  Microscopy, EM Tech Unit") but no photos.
- https://www.otago.ac.nz/omni/about -- 403 Forbidden (site blocked this fetch).
- https://www.otago.ac.nz/omni/people/image-gallery -- 403 Forbidden.
- https://www.otago.ac.nz/omni/electron-microscopy -- 403 Forbidden.
- https://www.otago.ac.nz/omni/confocal-microscopy -- 403 Forbidden (also noted as a cert error in the
  existing card from the prior pass -- otago.ac.nz appears to intermittently/selectively block
  automated fetches on several of its OMNI subpages).

## What was found
No interior/room photo retrievable this pass -- most OMNI subpages beyond the one "location" page
returned 403 Forbidden to automated fetch, which appears to be a site-side block rather than a
content issue (the one page that did load was plain text with no images at all).

## Newly confirmed facts beyond existing card
- None new -- the one accessible page reconfirmed facts already in the existing card (Room B01g
  confocal, Room B10 EM, Lindo Ferguson Building basement).

## Downloaded files
(none)

## Judgment call
otago.ac.nz appears to actively block automated WebFetch on most of its OMNI pages (repeated 403s
across /about, /people/image-gallery, /electron-microscopy, /confocal-microscopy); a future pass
might have better luck with a different fetch approach/user-agent, or by searching third-party
sources (news articles, OMNI's own social media, conference photos) instead of the otago.ac.nz domain
directly.
