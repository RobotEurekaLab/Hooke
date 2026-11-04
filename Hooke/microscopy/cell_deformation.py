"""Contact geometry for the estimated optical membrane deformation.

Offsets are in the cell's unrotated local frame, in metres. These helpers do
not change the reduced mechanical model or imply volume conservation.
"""

import numpy as np


def needle_contact_offset(tip_offset_m, radii, needle_back):
    """Intersect the intracellular tip's backward ray with its entry surface."""
    tip, radii, back = map(np.asarray, (tip_offset_m, radii, needle_back))
    scaled = tip/radii
    if float(scaled@scaled) >= 1.:
        return None
    direction = back/radii
    a, b = float(direction@direction), float(scaled@direction)
    distance = (-b+np.sqrt(max(0., b*b-a*(float(scaled@scaled)-1))))/a
    return tip+distance*back


def needle_contact_descriptor(offset_m, radii, needle_back, indentation_m):
    """Convert the scalar axial indentation into a local normal displacement."""
    offset, radii, back = map(np.asarray, (offset_m, radii, needle_back))
    normal = offset/radii**2
    normal /= np.linalg.norm(normal)
    return dict(offset_m=offset.tolist(),
                indentation_m=float(max(0., indentation_m)*max(0., normal@back)))


def handling_contact_descriptor(inward_normal, radii, compression_m):
    """Return the membrane support point opposing a probe or jaw reaction."""
    normal, radii = map(np.asarray, (inward_normal, radii))
    offset = -radii**2*normal/np.linalg.norm(radii*normal)
    return dict(offset_m=offset.tolist(), indentation_m=float(max(0., compression_m)))
