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

`phase_contrast.py` computes scalar thin-object transmission and averages the
intensity from eight annular illumination directions. Each direction uses a
shifted objective pupil and an attenuating phase ring. The current estimates
are 550 nm wavelength, objective NA 0.6 and illumination NA 0.35. A neutral
camera exposure, vignette and fixed detector-noise realization complete the
display. Patch boundaries taper into the background rather than appearing
as square seams.

Object height and focus apply independent Gaussian blur to each cell and
pipette plane. The empirical defocus scale is 0.45 µm per pixel of blur;
this is not a measured axial point-spread function. The pipette silhouette
uses the same SI meridian as the hollow 3D capillary, with translucent glass
walls. Its shaft is approximated as a single plane at tip height. Glass
refraction, optical propagation through the full shaft and volumetric cell
scattering remain unresolved.

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

Both engines use this object-space imaging model. Their world views come
from MuJoCo EGL or Isaac RTX respectively. Live images and paired channel
records are written to the configured, ignored microscopy media directory.
The neutral laboratory materials and backdrop affect world rendering only.
The microscope stand remains an independent TE2000-S dimensional/photo
reconstruction, rather than complete manufacturer CAD.

## References and asset provenance

- [Nikon: introduction to phase-contrast microscopy](https://www.microscopyu.com/techniques/phase-contrast/introduction-to-phase-contrast-microscopy)
  explains annular illumination, phase conversion and characteristic halos.
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
