# National Tsing Hua University -- Confocal Microscope System (CMS), Instrumentation Center

Confidence tier: **location confirmed, no room photo** (building-level interior found;
actual instrument room not captured)

## Search queries tried
- "Confocal Microscope System facility Instrumentation Center National Tsing Hua University 室內 照片"
- "清華大學 貴重儀器中心 共軛焦顯微鏡 儀器室 環景 導覽"

## URLs checked
- https://nscric.site.nthu.edu.tw/p/404-1186-124139.php (English version, main EN page -- no photos, just a
  table of instruments/contacts/operators; references a "儀器室環景導覽" (instrument-room panoramic tour) nav item)
- https://nscric.site.nthu.edu.tw/p/404-1186-249667.php ("各儀器室環景導覽" -- panoramic tours index page).
  This page embeds several Matterport 3D-tour iframes, one per instrument/room group. The iframe directly
  under the label "共軛焦顯微鏡系統" (Confocal Microscope System), grouped with "生物型核磁共振儀" (biomolecular
  NMR) and "X光生物巨分子單晶繞射儀" (X-ray macromolecular crystallography), is:
  https://my.matterport.com/show/?m=zheu2jSKbjz&lang=en
- https://my.matterport.com/show/?m=zheu2jSKbjz&lang=en -- Matterport 3D Showcase titled "生科二館"
  (Life Science Building 2). Confirms the CMS/NMR/X-ray group lives in that building, consistent with
  earlier research placing the Confocal system in Life Sciences Building 2, rooms 540/640.

## What was found
Pulled the Matterport showcase's og:image (the tour's default/starting view thumbnail, 1920x1080,
from https://my.matterport.com/api/v2/player/models/zheu2jSKbjz/thumb/) -- saved as
nthu_cms_matterport_thumb.jpg. This shows the BUILDING ENTRANCE/LOBBY of 生命科學二館 (Life Science
Building 2): a tiled facade with a glass double-door entrance, a terrazzo-look lobby floor visible
through the doors, suspended ceiling tiles, an umbrella stand, a small round table, and decorative
window displays (one with a brain-themed art panel, one with paper-crystal/floral decorations) on
either side of the entrance. A wall plaque reads "生命科學二館" with a founding date.

This confirms the building is real and the tour exists, but the default thumbnail is the lobby, not
the actual CMS instrument room (room 540/640). Matterport's interactive viewer has many additional
"sweep" positions inside the actual lab rooms, but those are only reachable by navigating the live
3D viewer (requires JS execution) -- not extractable via a static thumbnail fetch in the time budget
for this pass.

## Newly confirmed facts beyond existing card
- The CMS group (confocal + biomolecular NMR + X-ray macromolecular crystallography) is documented in
  a dedicated Matterport 3D virtual tour, confirming it physically sits in 生命科學二館 (Life Science
  Building 2) -- consistent with prior room 540/640 citation.
- A real, navigable 3D tour of the actual CMS room exists at
  https://my.matterport.com/show/?m=zheu2jSKbjz -- worth revisiting with a headless browser / Matterport
  API if deeper room-level captures are wanted later.

## Downloaded files
- nthu_cms_matterport_thumb.jpg (building lobby/entrance, not the instrument room itself)

## Judgment call
Not tiered as "room photo confirmed" because the downloaded image is the building lobby, not the
instrument room interior -- flagging the Matterport tour URL as the most promising lead for a future
pass with a JS-capable fetch.
