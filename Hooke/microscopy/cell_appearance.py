"""Original statistical cell appearance in SI units, independent of mechanics.

Contours and intracellular structure are visual estimates, not measured cell
ultrastructure. They remain within the nominal mechanical envelopes. Texture
is attached to specimen coordinates so translation cannot make it swim.
"""

import numpy as np


def background_geometry(index):
    """Estimated, non-colliding context cells with distinct size and orientation."""
    rng = np.random.default_rng(3301+index*53)
    radii = np.array([rng.uniform(16e-6, 25e-6), rng.uniform(8e-6, 13e-6),
                      rng.uniform(3e-6, 5e-6)])
    return radii, float(rng.uniform(-np.pi, np.pi))


def _texture(dx, dy, rng, length_m, waves=12):
    """Band-limited, aperiodic texture evaluated in physical coordinates."""
    field = np.zeros_like(dx)
    for _ in range(waves):
        angle, phase = rng.uniform(0, 2*np.pi, 2)
        period = length_m*rng.uniform(.7, 1.5)
        field += np.cos(2*np.pi*(dx*np.cos(angle)+dy*np.sin(angle))/period+phase)
    return field/np.sqrt(waves)


def cell_fields(dx, dy, radii, nuclear_radii, *, index, indentation_m=0., axial_slab=None):
    """Return optical path, absorption, cytoplasm and nuclear label masks.

    Nuclear optical contrast is deliberately modest: a nucleus need not have
    a larger refractive index than its cytoplasm. The nuclear label stays
    centrosymmetric for the existing centroid-based image controller.
    """
    radii, nuclear_radii = np.asarray(radii), np.asarray(nuclear_radii)
    rng = np.random.default_rng(7189+index*101)
    angle = np.arctan2(dy/radii[1], dx/radii[0])
    irregularity = .008 if index == 0 else .024
    contour = np.zeros_like(angle)
    for harmonic in (3, 5, 7):
        contour += 1-np.cos(harmonic*angle+rng.uniform(0, 2*np.pi))
    # Keep the calibrated major/minor diameters and the contact envelope.
    outline = 1-irregularity*contour*np.sin(2*angle)**2
    contact_angle = np.arctan2(np.sin(angle-np.pi/4), np.cos(angle-np.pi/4))
    outline -= min(.15, indentation_m/radii[0])*np.exp(-contact_angle**2/.12)
    distance = np.hypot(dx/radii[0], dy/radii[1])/outline
    cytoplasm = np.clip((1-distance)*35, 0., 1.)
    thickness = 2*radii[2]*np.sqrt(np.clip(1-distance**2, 0., 1.))
    if axial_slab is not None:
        offset, depth = axial_slab
        thickness = np.maximum(0., np.minimum(thickness/2, offset+depth/2)
                               - np.maximum(-thickness/2, offset-depth/2))
        cytoplasm *= np.clip(thickness/depth, 0., 1.)
    coarse = _texture(dx, dy, rng, 5.5e-6)
    fine = _texture(dx, dy, rng, 1.3e-6)
    opd = .010*thickness*(1+.018*coarse)
    opd += (4e-9*coarse+2e-9*fine)*cytoplasm
    absorption = (.014+.004*np.clip(coarse, -1., 1.))*cytoplasm

    nuclear_angle = np.arctan2(dy/nuclear_radii[1], dx/nuclear_radii[0])
    nuclear_outline = .955+.025*np.cos(2*nuclear_angle+.7)+.012*np.cos(4*nuclear_angle-.4)
    nuclear_distance = np.hypot(dx/nuclear_radii[0], dy/nuclear_radii[1])/nuclear_outline
    nucleus = np.clip((1-nuclear_distance)*22, 0., 1.)
    # Smooth nuclear path gradients avoid drawing a second, identical membrane
    # around the label; fine chromatin and nucleoli carry the internal contrast.
    nuclear_thickness = 2*nuclear_radii[2]*np.clip(1-nuclear_distance**2, 0., 1.)**1.1
    if axial_slab is not None:
        nuclear_thickness = np.maximum(0., np.minimum(nuclear_thickness/2, offset+depth/2)
                                       - np.maximum(-nuclear_thickness/2, offset-depth/2))
        nucleus *= np.clip(nuclear_thickness/depth, 0., 1.)
    chromatin = _texture(dx, dy, rng, 1.8e-6)
    opd += .009*nuclear_thickness+3e-9*chromatin*nucleus
    absorption += .007*nucleus

    # Granular cytoplasm is denser around the nucleus but not a regular ring.
    for count, width_range, strength_range in (
        (130, (.14e-6, .34e-6), (3e-9, 11e-9)),
        (28, (.4e-6, .85e-6), (-16e-9, 22e-9)),
    ):
        for _ in range(count):
            theta = rng.uniform(0, 2*np.pi)
            radius = np.sqrt(rng.uniform(.04, .9))
            x, y = radius*radii[:2]*[np.cos(theta), np.sin(theta)]
            width = rng.uniform(*width_range)
            stretch = rng.uniform(.65, 1.5)
            spot = np.exp(-.5*((dx-x)**2*stretch+(dy-y)**2/stretch)/width**2)
            spot *= cytoplasm*(1-.85*nucleus)
            opd += rng.uniform(*strength_range)*spot
            absorption += .008*spot
    for x, y, width in ((-.27*nuclear_radii[0], .18*nuclear_radii[1], .62e-6),
                        (.32*nuclear_radii[0], -.22*nuclear_radii[1], .43e-6)):
        spot = np.exp(-.5*((dx-x)**2+(dy-y)**2)/width**2)*nucleus
        opd += 23e-9*spot
        absorption += .05*spot
    return opd, absorption, cytoplasm, nucleus
