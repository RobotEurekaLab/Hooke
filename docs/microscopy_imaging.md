# Simulated cell imaging

The cell workstation provides two views of the same experiment:

| Channel | Appearance | Use |
| --- | --- | --- |
| Phase contrast | Grayscale cells, internal structure and phase halos | Unstained-cell display and autofocus |
| Fluorescence | Blue nuclear label and green injected tracer | Image-only target localization and dose visualization |

Choose a channel in **Microscope View** on `/microscopy`. The fluorescence
channel is available for adherent-cell and suspended-cell experiments. The
calibration-bead experiments retain their existing imaging model.

## Image formation

`cell_appearance.py` generates a specimen-fixed optical-path projection from
the nominal ellipsoid thickness, a nucleus, nucleoli, fine granules and larger
inclusions. Background cells have independent dimensions, orientation and
contour variations. These are original statistical shapes; they are not
measured organelles or a particular validated cell line.

Membrane variations remain inside the nominal mechanical envelope. Multiple
spatial scales of cytoplasmic texture move with the specimen; gentler nuclear
contrast and bounded phase exposure reduce saturated concentric rings.
The annular pupil transitions smoothly rather than introducing a sharp
numerical edge. These appearance refinements do not change cell size,
mechanical contact or the separate fluorescence control signal.

`phase_contrast.py` computes scalar thin-object transmission and averages the
intensity from twelve annular illumination directions. Each direction uses a
shifted objective pupil and an attenuating phase ring. The current estimates
are 550 nm wavelength, objective NA 0.6 and illumination NA 0.35. A neutral
camera exposure, vignette and fixed detector-noise realization complete the
display. Patch boundaries taper into the background rather than appearing
as square seams.

The phase display combines a defocused whole-cell projection with an estimated
2 µm slab through the actual ellipsoid at the focal height. Moving away from
the cell equator therefore changes the visible section; a thick cell is not
collapsed into one plane. This empirical blend is not confocal sectioning or
a calibrated thick-specimen phase-contrast model. Nuclear fluorescence retains
its separate projected mask and axial blur.

The pipette silhouette uses the same SI meridian as the hollow 3D capillary.
Its opacity is partitioned between depth planes along the actual inclined
shaft, then blurred and composited with floating-point premultiplied colour.
Focusing the distal tip does not sharpen the entire shaft. The empirical
defocus scale remains 0.45 µm per pixel of blur; it is not a measured axial
point-spread function. Glass refraction and volumetric scattering remain
unresolved. Public calibration labels these estimates explicitly.

Glass-wall contrast is an estimated phase-edge display, not measured optical
transmission. Status annotations stay outside the specimen area so a puncture
marker cannot cover the micron-scale tip. Digital crops enlarge recorded
pixels for inspection without changing the needle diameter or optical
resolution.

Fluorescence uses estimated nuclear and cytoplasmic masks. The green tracer
intensity is illustrative; it is not a measurement of concentration or
quantum yield. Injection does not recolour the unstained phase-contrast view.
The controller receives the separate nuclear-label image for localization;
the visible fluorescence channel makes that observation accessible to users.

## Scale and mechanics

The image retains a 160 µm object-space field, 768 × 768 pixels and a 20 µm
scale bar. The adherent target is nominally 36 × 28 × 8 µm. The suspended
target has a 36 µm diameter. The injection mouth is 1.2 µm outside diameter
and 0.5 µm inside diameter. Injection and holding pipettes retain their
35° and 30° elevations above horizontal.

Optical halos are not membrane coordinates: phase-contrast minima can occur
inside or outside an object's geometric boundary. The scale regression
therefore measures the filled tracer footprint and compares it with the
compiled 3D sample. Mechanical contact continues to use its ellipsoid
envelope; statistical visual contours do not replace the contact solver.

The unchanged reduced mechanical model includes contact, membrane indentation,
threshold puncture, pressure response, capillary resistance and conserved
liquid volume. It does not predict viability, membrane repair or resolved
intracellular mechanics. Public calibration explicitly reports
`physical_optics_calibrated: false` and
`biological_texture_calibrated: false`.

After cell localization, injection demos physically move the focus joint to
the planned insertion height and return to the cell-centre plane after
withdrawal. The controller does not enlarge the needle or alter the mechanical
contact boundary to make the image clearer. Observation records preserve the
actual focus and needle pose.

Completed commands archive a bounded, per-step `interaction.npz` beside the
episode result. It contains needle and cell positions, the cell pose used for
contact, normalized ellipsoid distance, penetration, indentation, reaction
force, focal height, pressure and conserved delivered volume. `result.json`
records the schema, row count, dropped-row count and SHA-256 checksum. The
contact pose is distinct from the subsequently updated suspended-cell pose.
These traces verify the reduced membrane model; cells are not rigid PhysX
contact bodies, and puncture is not a resolved mesh-tearing simulation.

Both engines use this object-space imaging model. Their world views come
from MuJoCo EGL or Isaac RTX respectively. Live images and paired channel
records are written to the configured, ignored microscopy media directory.
The neutral laboratory materials and backdrop affect world rendering only.
The microscope stand remains an independent TE2000-S dimensional/photo
reconstruction, rather than complete manufacturer CAD.
The same world geometry can also be exported to an
[editable Blender scene](microscopy_rendering.md) for offline presentation.

## References and asset provenance

- [Nikon: introduction to phase-contrast microscopy](https://www.microscopyu.com/techniques/phase-contrast/introduction-to-phase-contrast-microscopy)
  explains annular illumination, phase conversion and characteristic halos.
- [Nikon: phase-contrast microscope configuration](https://www.microscopyu.com/techniques/phase-contrast/phase-contrast-microscope-configuration)
  describes the contrast and out-of-focus limitations of thick specimens.
- [Catfaster: HeLa cell phase-contrast photograph](https://commons.wikimedia.org/wiki/File:HeLa_Cells_Culture_Phase_Contrast_1_v1.jpg),
  CC BY-SA 4.0, was viewed as an unmodified local morphology reference. It is
  not incorporated into the cell textures or distributed as an application
  asset. Its download and license record remain under ignored `temp/`.

The formulas above are implementation estimates informed by these references;
they have not been fitted to either a real instrument or the reference image.

## Validation — 2026-09-18

- 40 regression cases passed, including channel separation, image-only
  localization, axial focus response, SI sample/pipette dimensions, conservative
  injection, failure controls, bounded blur caching and API isolation. The
  optional CAD case used the complete fixture directory through
  `HOOKE_TEST_CAD_ASSET_ROOT`.
- The live browser completed adherent-cell and suspended-cell injection on
  both engines: four demos, 62 passed experiment criteria, zero script errors.
  Native traces recorded 9,393 and 14,385 actual PhysX steps respectively.
- Both channels were retained from the same injection observation. All six
  images on the temporary LAN review page loaded successfully.
- The original homepage HTML, CSS and Examples behavior still match `main`.

Detailed results and generated images are retained under ignored
`temp/microscopy_realism/20260918T074312114538Z/`. These checks validate the
implemented simulation and display; they do not establish biological or
instrument calibration.
