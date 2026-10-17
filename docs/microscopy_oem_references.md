# Manufacturer references for the microscopy workstation

Reference audit: 2026-09-19.

Original Sutter dimensional and installation drawings are available for a
reconstruction based on the MP-285 and XenoWorks families. Nikon publishes
front and side dimensions for the TE2000-S. Complete manufacturer 3D CAD for
either assembly was not obtained in this search.

These sources now support the independently reconstructed
[`sutter-mp285-reference` profile](microscopy_sutter_reference.md).
The optional `parallel-v4` profile retains the imported MicroManipulatorStepper
mechanism. Neither that mechanism nor the new reconstruction is Sutter OEM CAD.
See [the rendering guide](microscopy_rendering.md) for shared geometry export.

## Primary drawings

Page numbers below are one-based PDF pages; printed page numbers are supplied
where they differ. Dimensions apply to the depicted configuration.

| Component | Original manufacturer source | Evidence and limits |
| --- | --- | --- |
| MP-285 mechanism | [Sutter catalog, PDF 89 / printed 90](https://www.sutter.com/hubfs/WEB%20-%20PDF%20Files/SutterCatalog.pdf#page=89) | Dimensioned top, front and side views, plus an isometric view. The shown assembly is 4.83 in wide, 6.83 in deep and 5.78 in high. This is not a full-travel envelope or a drawing of every XenoWorks variant. |
| MP-285 base | [Basic operation manual, PDF 14 / printed 6, Figure 2-5](https://www.sutter.com/hubfs/WEB%20-%20Manuals/MP-285_OpMan_ROE_BasicOp.pdf#page=14) | Four mounting holes on a 2.5 in / 63.5 mm square. The main base outline is 2.75 in / 69.85 mm square. Four precision washers separate it from the mounting surface; washer thickness is unspecified. |
| MP-285 mounting hardware | [Reference manual, PDF 64–66 / printed 52–54](https://www.sutter.com/hubfs/WEB%20-%20Manuals/MP-285_RefMan.pdf#page=64) | M3 or #4 mounting screws, M3 × 0.5 × 14 mm screws for adapter 285210, and 0.5 in / 12.7 mm vertical mount increments. |
| XenoWorks installation | [XWM-100 manual, PDF 25–26 / printed 23–24](https://www.sutter.com/hubfs/WEB%20-%20Manuals/XW_XWM-100_OpMan_r103h.pdf#page=25) | Figures 2-1 and 2-2 show the upright mechanism, broad adjustable base plate and lower dovetail engaging a microscope adapter. Left and right assemblies mirror. |
| XenoWorks pipette holder | [BRE manual, PDF 15 / printed 9, Figure 2-1](https://www.sutter.com/hubfs/WEB%20-%20Manuals/XW_BRE_Digital_OpMan.pdf#page=15) | Exploded holder, chuck, O-ring and tubing arrangement. This is an assembly illustration without a complete dimension set. |
| Nikon TE2000-S | [Nikon brochure, PDF 24](https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf#page=24) | Dimensioned front and side outlines: 195 mm base width, 260 mm overall front-view width, 476.4 mm base depth, 611 mm total height and 449.4 mm eyepoint. Pages 22–23 show system components. Stage height and mounting-hole coordinates are not specified. |

The MP-285 catalog dimensions convert to approximately 122.68 × 173.48 ×
146.81 mm (width × depth × height). Conversion does not imply this level of
manufacturing accuracy. Keep the catalog pose separate from the rounded
product-page dimensions and from XenoWorks variant specifications.

Sutter also publishes [right-angle](https://www.sutter.com/hubfs/WEB%20-%20Product%20Images/mp285_rtangle.pdf),
[vertical-mount](https://www.sutter.com/hubfs/WEB%20-%20Product%20Images/mp285_vertical.pdf)
and [exploded adapter](https://www.sutter.com/hubfs/WEB%20-%20Product%20Images/mp285_exploded_view.pdf)
illustrations. These help identify parts and fasteners but do not provide
complete manufacturing geometry. MP-225 drawings describe a different model
and must not supply missing MP-285 dimensions.

## Installation and tool geometry

The [XenoWorks product page](https://www.sutter.com/microinjection/xenoworksmm)
identifies **BR-N27** as the Nikon TE2000 adapter. Combined with Figures 2-1
and 2-2, the documented assembly order is:

1. Microscope frame and microscope-specific adapter.
2. Dovetail and adjustable base plate.
3. Left or right electric XYZ mechanism.
4. Adjustable pipette clamp, holder and glass capillary.

The mechanism is based on MP-285 electric linear axes with 25 mm travel per
axis. Its rotary pipette mount sets the approach angle. The current exposed
parallel linkage and custom bench supports do not represent that mechanism
or the illustrated XenoWorks microscope installation.

The product page specifies BR-AW and BR-AW-L rod clamps with nominal lengths
of 50 and 75 mm and rod diameters of 2–4 mm. The
[BRE product page](https://www.sutter.com/microinjection/xenoworksdm)
specifies a 4 mm holder rod and 1 mm capillary glass, with 1.2 and 1.5 mm
alternatives. These are stock glass/rod dimensions, not needle-tip diameters.
The existing micrometre-scale cell and tip dimensions must remain distinct
from the millimetre-scale holder and capillary stock.

The published BRM-ADPTR-PLT dimensions are nominal: 5.75 × 6 × 0.3 in and
145 × 155 × 8 mm do not convert exactly. Its hole pattern and dovetail
cross-section are still unknown. The XWM manual refers to a separate
microscope-adapter instruction sheet for alignment settings; the BR-N27
sheet was not obtained. Product compatibility alone does not establish its
mounting coordinates or certify an independently reconstructed interface.

### Cell-contact direction

[XWM-100 manual, PDF 31–32 / printed 29–30](https://www.sutter.com/hubfs/WEB%20-%20Manuals/XW_XWM-100_OpMan_r103h.pdf#page=31)
distinguishes three arrangements:

| Application | Illustrated arrangement |
| --- | --- |
| Suspended cells in a dish, Figure 3-8 | Holding shaft at 35° and injection shaft at 45°, with bent terminal tips approaching the cell approximately horizontally. |
| Suspended cells on a depression slide, Figure 3-9 | Straight pipettes at shallow angles of 0–15°. |
| Adherent cells, Figure 3-10 | A straight injection pipette at 45°. |

A shaft angle does not determine the final tip direction when the glass is
bent. The Sutter reference profile reconstructs the glass centreline,
cell-contact tangent and optical focus together. Its bend dimensions remain
estimates. Other profiles retain their earlier straight inclined pipettes.

## Other microscope references checked

| Microscope | Original source and useful material |
| --- | --- |
| Nikon Ti2 | [Brochure, PDF 12 / printed 22–23](https://downloads.microscope.healthcare.nikon.com/phase7/literature/Brochures/Ti2_2CE-MPOH-3_12602T_24P.pdf#page=12): dimensioned front/side outlines and stage heights for specific configurations. |
| Olympus / Evident IX73 | [Instruction manual](https://adobeassets.evidentscientific.com/content/dam/mis/ix73/manuals/AX8157_13_IX73_INST_E.pdf): assembly diagram on PDF 56; manipulator accessory mounting on PDF 66. |
| Olympus / Evident IX83 | [Instruction manual](https://adobeassets.evidentscientific.com/content/dam/mis/ix83/manuals/AX8149_18_IX83_P1ZF_Inst_E.pdf): assembly diagram on PDF 63; accessory mounting on PDF 72. |
| ZEISS Axio Observer | [Instruction manual](https://asset-downloads.zeiss.com/catalogs/download/mic/85933b01-9d7e-4616-a07d-c502f5224d50/EN_IM_Axio_Observer_3-5-7_V02.pdf): component and installation illustrations; approximate overall dimensions on PDF 105. |

The IX73/IX83 manuals specify 12 accessory holes, M6 screws with a maximum
10 mm insertion, and spacers below 11.5 mm diameter and above 5 mm height.
These are useful installation references for those stands; they are not
TE2000 interface specifications. Continue with TE2000-S to keep one coherent
microscope configuration.

## CAD availability and source handling

No complete MP-285/XenoWorks or target inverted-microscope STEP, IGES, DWG,
DXF or STL was obtained from the official resources searched.
This does not establish that manufacturer CAD is unavailable by request.

[Nikon's OEM CAD catalog](https://www.microscope.healthcare.nikon.com/products/oem/cad)
offers a request route for selected components, chiefly objectives. The
[inspected component agreement](https://www.microscope.healthcare.nikon.com/en_AOM/products/oem/cad/request?id=69155)
limits use to preliminary fit/layout work and restricts modification and
distribution without permission. It does not establish access to a complete
TE2000 stand. [Evident describes selected-component STEP appearance models](https://evidentscientific.com/en/insights/3d-cad-data-microscope-components)
provided through a form/email process; complete IX73/IX83 stand availability
and redistribution rights were not established. No request form was submitted.

A subsequent check of [Evident's public OEM catalog](https://evidentscientific.com/en/downloads/oem)
obtained genuine STEP files for the IX-ULWCD condenser and BX53F2 upright
frame. These do not supply a Nikon stand or complete inverted microscope.
The downloaded packages contain no license; the
[site terms](https://evidentscientific.com/en/legal/terms-of-use) restrict
alteration and distribution without permission. They remain local references
and are not imported into the distributed scene.

The downloaded documents are manufacturer references, not open-source asset
packages. No academic redistribution grant was found. Keep the original PDFs,
photographs and extracted page images under ignored `temp/`; retain source
links and factual measurements in repository documentation. Do not describe
independent reconstruction as OEM CAD or validated one-to-one hardware.

Local evidence is stored in `temp/microscopy_oem_references/20260919/`:

- `sutter/manifest.json`: download URLs, file sizes and SHA-256 hashes.
- `sutter/verified_dimensions.json`: dimensions, source pages and unresolved
  interfaces, including conflicting manufacturer specifications.
- `microscope/downloads.json`: microscope source URLs and hashes.
- `microscope/evidence-index.json`: downloaded documents and inspected pages.

## Reconstruction acceptance criteria

- Match the selected OEM orthographic outlines before material/render polish.
- Keep measured, nominal and estimated dimensions distinguishable.
- Articulate the actual serial XYZ stages, with separate mount and tool frames.
- Reconstruct the left/right installation and bent-tip geometry explicitly;
  keep needle approach direction separate from the holder angle.
- Verify reach, clearances, focus and the existing cell tasks on MuJoCo and
  native Isaac using the same geometry and SI units.
- Compare front, side and experiment views; presentation renders alone cannot
  establish correct assembly or successful cell contact.
