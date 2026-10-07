# Newcastle University -- Genomics Core Facility (GCF), Institute of Genetic Medicine

Confidence tier: **location confirmed, no room photo** (photo exists but could not be downloaded)

## Search queries tried
- "Newcastle University Genomics Core Facility Bioscience Building International Centre for Life photo"
- Direct fetch of https://research.ncl.ac.uk/genomicscorefacility/
- Direct fetch of https://diagnosticsnortheast.org.uk/under-the-microscope-episode-8-unlocking-innovation-at-the-genomics-core-facility/

## URLs checked
- https://research.ncl.ac.uk/singlecell/gcf.php (card's prior source) -- not re-fetched this pass (prior pass could not directly read it)
- https://research.ncl.ac.uk/genomicscorefacility/ -- only two small navigation icons (mail/social), no facility photos
- https://diagnosticsnortheast.org.uk/under-the-microscope-episode-8-unlocking-innovation-at-the-genomics-core-facility/ -- **found a real GCF team photo** ("GCF team (left to right): Katherine Johnson, Robert Jackson, Jonathan Coxhead, Rafiqul Hussain, Rachel Smith") plus several more blog images, but the hosting WordPress site (diagnosticsnortheast.org.uk) returned HTTP 403 Forbidden on every direct download attempt (tried with a browser user-agent and a referer header; also tried via WebFetch directly) -- likely bot/hotlink protection. Could not retrieve the actual image bytes in this pass.
- https://www.10xgenomics.com/blog/strengthening-the-core-collaboration-when-tissue-meets-genomics-at-newcastle-university -- a 10x Genomics case-study blog post about this facility, surfaced in search, not fetched this pass -- worth trying in a follow-up (10x's own blog CDN may not be as aggressively protected as diagnosticsnortheast.org.uk)

## What was found
Confirmed named staff for the GCF: Jonathan Coxhead, Rafiqul Hussain, Rachel Queen/Smith (name appears slightly differently across sources -- "Rachel Queen" per the card vs. "Rachel Smith" per the DxNE team-photo caption; flagging as a possible name discrepancy/change, not resolved), plus two new names: Katherine Johnson and Robert Jackson. A genuine team photo exists showing these people, presumably inside or near the facility, but it was not retrievable as image bytes in this pass due to server-side blocking.

## New facts beyond the existing card
- Two additional GCF staff names beyond the card: Katherine Johnson, Robert Jackson.
- Possible name discrepancy: "Rachel Queen" (card) vs. "Rachel Smith" (DxNE blog) -- could be a name change (e.g. marriage) or a different person; unresolved.

## Downloaded files
None -- the one promising photo (GCF team photo) is blocked by the hosting site's anti-hotlinking/bot protection.

## Confidence tier reasoning
A specific, real photo of facility staff was located and described, but not successfully downloaded despite multiple attempts -- so this stays at "location confirmed, no room photo" rather than "room photo confirmed," since no image bytes were actually retrieved and verified by this agent. A follow-up pass should try the 10x Genomics blog post as an alternate, less-protected host for similar imagery.
