"""Original statistical cell appearance in SI units, independent of mechanics.

The contour perturbations and organelles are visual estimates, not measured
cell ultrastructure or a replacement for the mechanical contact envelope.
"""

import numpy as np


def background_geometry(index):
    """Estimated, non-colliding context cells with distinct size and orientation."""
    rng = np.random.default_rng(3301+index*53)
    radii = np.array([rng.uniform(16e-6, 25e-6), rng.uniform(8e-6, 13e-6),
                      rng.uniform(3e-6, 5e-6)])
    return radii, float(rng.uniform(-np.pi, np.pi))


def cell_fields(dx, dy, radii, nuclear_radii, *, index, indentation_m=0.):
    """Return projected optical path, absorption, cytoplasm and nuclear masks."""
    radii, nuclear_radii = np.asarray(radii), np.asarray(nuclear_radii)
    rng = np.random.default_rng(7189 + index * 101)
    angle = np.arctan2(dy / radii[1], dx / radii[0])
    outline = np.ones_like(angle)
    for harmonic in (3, 5, 7):
        outline += (.009 if index == 0 else .035) * np.cos(harmonic * angle + rng.uniform(0, 2*np.pi))
    contact_angle = np.arctan2(np.sin(angle-np.pi/4), np.cos(angle-np.pi/4))
    outline -= min(.15, indentation_m/radii[0]) * np.exp(-contact_angle**2/.12)
    distance = np.hypot(dx/radii[0], dy/radii[1]) / outline
    cytoplasm = np.clip((1-distance)*35, 0., 1.)
    thickness = 2*radii[2]*np.sqrt(np.clip(1-distance**2, 0., 1.))
    opd = .013*thickness
    absorption = .018*cytoplasm
    nuclear_distance = np.hypot(dx/nuclear_radii[0], dy/nuclear_radii[1])
    nucleus = np.clip((1-nuclear_distance)*22, 0., 1.)
    opd += .019*2*nuclear_radii[2]*np.sqrt(np.clip(1-nuclear_distance**2, 0., 1.))
    absorption += .012*nucleus
    # A few perinuclear vacuoles and many fine, aperiodic granules. Positions
    # are fixed in the specimen, so stage motion cannot make the texture swim.
    for count, width_range, strength_range in (
        (65, (.16e-6, .38e-6), (1.5e-9, 5e-9)),
        (18, (.45e-6, 1.1e-6), (-12e-9, 18e-9)),
    ):
        for _ in range(count):
            theta = rng.uniform(0, 2*np.pi)
            radius = np.sqrt(rng.uniform(.08, .9))
            x, y = radius*radii[:2]*[np.cos(theta), np.sin(theta)]
            width = rng.uniform(*width_range)
            spot = np.exp(-.5*((dx-x)**2+(dy-y)**2)/width**2)*cytoplasm
            opd += rng.uniform(*strength_range)*spot
    for x, y, width in ((-.27*nuclear_radii[0], .18*nuclear_radii[1], .65e-6),
                        (.32*nuclear_radii[0], -.22*nuclear_radii[1], .48e-6)):
        spot = np.exp(-.5*((dx-x)**2+(dy-y)**2)/width**2)*nucleus
        opd += 30e-9*spot
        absorption += .045*spot
    return opd, absorption, cytoplasm, nucleus
