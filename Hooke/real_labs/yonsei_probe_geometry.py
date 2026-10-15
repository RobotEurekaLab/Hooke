"""Photo-informed three-sided RF fixture with real open mechanical interfaces.

The visible 40A-GSG-100-DP label and manufacturer naming support a 100 micrometre
GSG pitch and rear-facing 45-degree connector. Station dimensions, hidden
positioners, needle diameters and inert pad geometry remain authored estimates.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

PITCH = 0.0001
TIP_RADIUS = 0.000025
TIP_POSITION = (0.003, 0, 0.0824)
PAD_POSITION = (0.001, 0, 0.08164)
MANUFACTURER = "https://ggb.com/wp-content/uploads/2025/04/mod40a.pdf"


class Parts:
    def __init__(self, builder):
        self.b = builder

    def name(self, key):
        return f"{self.b.name}__mics_{key}"

    def box(self, parent, key, pos, half, material="metal", **kwargs):
        return self.b.box(
            parent, pos, half, material, collision=True, name=self.name(key), **kwargs
        )

    def cylinder(self, parent, key, pos, radius, half, material="metal", **kwargs):
        return self.b.cylinder(
            parent,
            pos,
            radius,
            half,
            material,
            collision=True,
            name=self.name(key),
            **kwargs,
        )

    def rod(self, parent, key, a, c, radius, material="metal"):
        return self.b.rod(
            parent, a, c, radius, material, collision=True, name=self.name(key)
        )

    def wedge(self, parent, key, side, sections, material):
        """A closed convex tapered housing, with the same visible/contact solid."""
        vertices = []
        for x, width, low, high in sections:
            vertices.extend(
                (side * x, y, z)
                for y, z in ((-width, low), (width, low), (width, high), (-width, high))
            )
        quads = (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        faces = [
            triangle for a, c, d, e in quads for triangle in ((a, c, d), (a, d, e))
        ]
        volume = sum(
            sum(
                vertices[a][i]
                * (
                    vertices[c][(i + 1) % 3] * vertices[d][(i + 2) % 3]
                    - vertices[c][(i + 2) % 3] * vertices[d][(i + 1) % 3]
                )
                for i in range(3)
            )
            for a, c, d in faces
        )
        if volume < 0:
            faces = [tuple(reversed(f)) for f in faces]
        mesh = self.name(key + "_mesh")
        ET.SubElement(
            self.b.root.find("asset"),
            "mesh",
            name=mesh,
            vertex=numbers([v for p in vertices for v in p]),
            face=" ".join(str(i) for f in faces for i in f),
        )
        self.b.geom(
            parent,
            "mesh",
            (),
            material=material,
            collision=True,
            name=self.name(key),
            mesh=mesh,
        )

    def sleeve(self, parent, key, center, axis, length, bore, wall=0.004):
        axes = [i for i in range(3) if i != axis]
        for ii, a in enumerate(axes):
            other = axes[1 - ii]
            for sign in (-1, 1):
                pos = list(center)
                pos[a] += sign * (bore[ii] + wall / 2)
                half = [0.0, 0.0, 0.0]
                half[axis] = length
                half[a] = wall / 2
                half[other] = bore[1 - ii] + (wall if ii == 0 else 0)
                self.box(parent, f"{key}_{a}_{sign}", pos, half, "dark")


def _head(b, p, parent, side):
    label = "right" if side > 0 else "left"
    p.wedge(
        parent,
        label + "_housing",
        side,
        [(0.033, 0.007, 0.087, 0.103), (0.115, 0.023, 0.094, 0.120)],
        "mics_coating",
    )
    p.wedge(
        parent,
        label + "_nose",
        side,
        [(0.006, 0.00050, 0.0828, 0.0837), (0.040, 0.006, 0.088, 0.099)],
        "bright",
    )
    for i, y in enumerate((-PITCH, 0, PITCH)):
        p.rod(
            parent,
            f"{label}_tip_{i}",
            (side * 0.006, y, 0.0832),
            (side * 0.003, y, 0.0824),
            TIP_RADIUS,
            "bright",
        )
        b.site(
            parent, f"{label}_tip_{i}", (side * 0.003, y, 0.0824 - TIP_RADIUS), 0.00002
        )
    # Named smooth plated body, nut and relief follow the source's diagonal coax.
    p.rod(
        parent,
        label + "_connector_base",
        (side * 0.078, 0, 0.111),
        (side * 0.098, 0, 0.131),
        0.008,
        "bright",
    )
    p.rod(
        parent,
        label + "_connector_nut",
        (side * 0.092, 0, 0.125),
        (side * 0.106, 0, 0.139),
        0.010,
        "metal",
    )
    # A six-flat coupling exterior replaces the capsule silhouette while the
    # original enclosing contact proxy remains unchanged.
    nut = parent.find(f"geom[@name='{p.name(label + '_connector_nut')}']")
    nut.set('rgba', '0 0 0 0')
    nut.set('group', '3')
    _hex_coupling(b, p, parent, label, side)
    p.rod(
        parent,
        label + "_connector_barrel",
        (side * 0.103, 0, 0.136),
        (side * 0.124, 0, 0.157),
        0.0075,
        "bright",
    )
    p.cylinder(
        parent,
        label + "_mount_screw",
        (side * 0.102, -0.009, 0.123),
        0.006,
        0.005,
        "bright",
    )
    p.cylinder(
        parent,
        label + "_screw_recess",
        (side * 0.102, -0.009, 0.128),
        0.0025,
        0.0005,
        "dark",
    )
    for x, y in ((0.049, -0.006), (0.073, 0.009), (0.108, 0.014)):
        top = 0.103 + (x - 0.033) * (0.120 - 0.103) / (0.115 - 0.033)
        b.cylinder(parent, (side * x, y, top + 0.001), 0.0028, 0.001,
                   'bright', density=0, name=p.name(f'{label}_head_fastener_{x}'))
        b.cylinder(parent, (side * x, y, top + 0.0021), 0.0012, 0.0001,
                   'dark', density=0)
    # Source cables leave the photograph. These supported short service stubs
    # deliberately terminate openly; no external instrument/electrical feed is implied.
    controls = [
        (side * 0.123, 0, 0.156),
        (side * 0.145, 0.018, 0.188),
        (side * 0.180, 0.058, 0.221),
        (side * 0.196, 0.083, 0.216),
    ]
    points = []
    for i in range(19):
        t = i / 18
        weights = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3)
        points.append(
            tuple(sum(w * v[j] for w, v in zip(weights, controls)) for j in range(3))
        )
    for i, (a, c) in enumerate(zip(points, points[1:])):
        p.rod(parent, f"{label}_coax_stub_{i}", a, c, 0.009, "rubber")
    p.cylinder(
        parent,
        label + "_truncated_service_end",
        points[-1],
        0.008,
        0.001,
        "dark",
        euler=(math.pi / 2, 0, 0),
    )


def _hex_coupling(b, p, parent, label, side):
    start, end = (side * 0.092, 0, 0.125), (side * 0.106, 0, 0.139)
    # Orthogonal unit directions around the 45-degree connector axis.
    u, v = (0, 1, 0), (-1 / math.sqrt(2), 0, side / math.sqrt(2))
    vertices = [
        tuple(center[k] + 0.0098 * (math.cos(angle) * u[k] + math.sin(angle) * v[k]) for k in range(3))
        for center in (start, end)
        for angle in (i * math.tau / 6 for i in range(6))
    ]
    faces = [(0, i + 1, i) for i in range(1, 5)]
    faces += [(6, 6 + i, 7 + i) for i in range(1, 5)]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend(((i, j, 6 + j), (i, 6 + j, 6 + i)))
    mesh = p.name(label + '_coupling_surface_mesh')
    ET.SubElement(b.root.find('asset'), 'mesh', name=mesh,
                  vertex=numbers([v for point in vertices for v in point]),
                  face=' '.join(str(v) for face in faces for v in face))
    b.geom(parent, 'mesh', (), mesh=mesh, material='bright', density=0,
           name=p.name(label + '_coupling_surface'))


def _chuck(b, p, base):
    p.box(base, "slide_foundation", (0, 0, 0.027), (0.102, 0.091, 0.003), "metal")
    for i, y in enumerate((-0.060, 0.060)):
        for j, x in enumerate((-0.092, 0.092)):
            p.box(
                base,
                f"x_pedestal_{i}_{j}",
                (x, y, 0.0365),
                (0.006, 0.008, 0.0065),
                "dark",
            )
        p.box(base, f"x_rail_{i}", (0, y, 0.0455), (0.098, 0.0025, 0.0025), "bright")
    chuck = b.moving(
        base, "chuck_x", (0, 0, 0.058), (1, 0, 0), (-0.003, 0.003), mass=0.07, kp=250
    )
    p.box(chuck, "chuck_carriage", (0, 0, 0), (0.086, 0.076, 0.006), "dark")
    for i, x in enumerate((-0.060, 0.060)):
        for j, y in enumerate((-0.060, 0.060)):
            p.sleeve(
                chuck, f"x_shoe_{i}_{j}", (x, y, -0.0125), 0, 0.018, (0.0031, 0.0031)
            )
    p.cylinder(chuck, "chuck_hub", (0, 0, 0.011), 0.068, 0.010)
    p.cylinder(chuck, "chuck_disc", (0, 0, 0.015), 0.082, 0.006, "bright")
    p.cylinder(chuck, "chuck_face", (0, 0, 0.0213), 0.070, 0.0003, "metal")
    p.box(chuck, "substrate", (0, 0, 0.0224), (0.010, 0.008, 0.0008), "lens")
    p.box(chuck, "inert_die", (0, 0, 0.0234), (0.0024, 0.002, 0.0002), "copper")
    b.site(chuck, "rf_coupon", (0, 0, 0.0234), 0.00015)
    for side in (-1, 1):
        for i, y in enumerate((-PITCH, 0, PITCH)):
            p.box(
                chuck,
                f"pad_{side}_{i}",
                (side * 0.001, y, 0.02362),
                (0.000025, 0.000035, 0.00002),
                "bright",
            )
            b.site(chuck, f"pad_{side}_{i}", (side * 0.001, y, 0.02364), 0.000015)
    for i in range(9):
        p.box(
            chuck,
            f"rear_pad_{i}",
            ((i - 4) * 0.00025, 0.0014, 0.02362),
            (0.000055, 0.000055, 0.00002),
            "bright",
        )
    # Explicit thin clips connect the large substrate to the chuck face.
    for i, x in enumerate((-0.011, 0.011)):
        for j, y in enumerate((-0.006, 0.006)):
            p.box(
                chuck,
                f"clip_post_{i}_{j}",
                (x, y, 0.02235),
                (0.0006, 0.0012, 0.00075),
                "dark",
            )
            p.box(
                chuck,
                f"clip_arm_{i}_{j}",
                (math.copysign(0.0102, x), y, 0.02315),
                (0.0013, 0.0012, 0.00005),
                "metal",
            )
    for i, x in enumerate((-0.062, 0.062)):
        p.cylinder(
            chuck, f"chuck_screw_{i}", (x, -0.044, 0.0225), 0.004, 0.0015, "dark"
        )
    return chuck


def _positioners(b, p, base):
    for side in (-1, 1):
        label = "right" if side > 0 else "left"
        p.box(
            base,
            label + "_positioner_base",
            (side * 0.155, 0.014, 0.042),
            (0.034, 0.052, 0.018),
            "dark",
        )
        p.box(
            base,
            label + "_fixed_spine",
            (side * 0.18, 0.014, 0.085),
            (0.009, 0.043, 0.025),
            "dark",
        )
        p.rod(
            base,
            label + "_adjuster_shaft",
            (side * 0.180, 0.014, 0.082),
            (side * 0.204, 0.014, 0.082),
            0.0035,
            "bright",
        )
        p.cylinder(
            base,
            label + "_adjuster_dial",
            (side * 0.206, 0.014, 0.082),
            0.010,
            0.008,
            "metal",
            euler=(0, math.pi / 2, 0),
        )
        for j, y in enumerate((-0.023, 0.048)):
            p.box(
                base,
                f"{label}_z_rail_{j}",
                (side * 0.145, y, 0.090),
                (0.003, 0.003, 0.030),
                "bright",
            )
            p.box(
                base,
                f"{label}_z_cap_{j}",
                (side * 0.145, y, 0.123),
                (0.006, 0.006, 0.003),
                "dark",
            )
        parent = (
            base
            if side < 0
            else b.moving(
                base,
                "right_probe_lift",
                (0, 0, 0),
                (0, 0, 1),
                (0, 0.010),
                mass=0.045,
                kp=180,
            )
        )
        p.box(
            parent,
            label + "_saddle",
            (side * 0.112, 0.012, 0.089),
            (0.027, 0.024, 0.009),
            "metal",
        )
        p.box(
            parent,
            label + "_saddle_bridge",
            (side * 0.131, 0.012, 0.083),
            (0.0085, 0.044, 0.008),
            "metal",
        )
        if side > 0:
            for j, y in enumerate((-0.023, 0.048)):
                p.sleeve(
                    parent,
                    f"right_z_shoe_{j}",
                    (0.145, y, 0.083),
                    2,
                    0.010,
                    (0.0036, 0.0036),
                )
        else:
            p.box(
                base,
                "left_fixed_saddle_post",
                (-0.137, 0.012, 0.070),
                (0.014, 0.030, 0.010),
                "dark",
            )
        _head(b, p, parent, side)


def _rear_wedge(b, p, base):
    p.box(base, "rear_foot", (0, 0.139, 0.049), (0.052, 0.028, 0.025), "dark")
    p.box(base, "rear_support", (0, 0.135, 0.083), (0.044, 0.022, 0.012), "metal")
    p.box(base, "rear_connector", (0, 0.110, 0.101), (0.041, 0.044, 0.013), "dark")
    p.box(base, "rear_front_bezel", (0, 0.068, 0.096), (0.037, 0.005, 0.013), "metal")
    for i in range(9):
        x = (i - 4) * 0.005
        p.rod(
            base,
            f"rear_fanout_{i}",
            (x, 0.064, 0.091),
            ((i - 4) * 0.00025, 0.0015, 0.0825),
            0.00005,
            "bright",
        )
    # An actual flat band carries parallel fine ribs, not nine detached sticks.
    p.box(base, "ribbon_base", (0, 0.126, 0.139), (0.031, 0.0016, 0.031), "cream")
    p.box(
        base,
        "ribbon_end",
        (0, 0.121, 0.188),
        (0.031, 0.0016, 0.023),
        "cream",
        euler=(0.22, 0, 0),
    )
    for i in range(17):
        x = (i - 8) * 0.0035
        p.rod(
            base,
            f"ribbon_rib_lower_{i}",
            (x, 0.1242, 0.109),
            (x, 0.1242, 0.169),
            0.00025,
            "blue",
        )
        p.rod(
            base,
            f"ribbon_rib_upper_{i}",
            (x, 0.1242, 0.169),
            (x, 0.1144, 0.210),
            0.00025,
            "blue",
        )


def build_probe_station(b, base, params):
    ET.SubElement(b.root.find('asset'), 'material', name=b.name + '__mat_mics_coating',
                  rgba='.045 .055 .075 1', specular='.30', shininess='.35')
    p = Parts(b)
    p.box(base, "foundation", (0, 0, 0.012), (0.22, 0.18, 0.012), "dark")
    for i, x in enumerate((-0.195, 0.195)):
        for j, y in enumerate((-0.155, 0.155)):
            p.cylinder(base, f"anchor_{i}_{j}", (x, y, 0.024), 0.008, 0.002, "metal")
    _chuck(b, p, base)
    _positioners(b, p, base)
    _rear_wedge(b, p, base)
    b.metadata["capabilities"] = [
        "retained_chuck_translation",
        "guided_right_probe_lift",
        "dry_true_scale_gsg_alignment",
    ]
    b.metadata["inspection_interface"] = {
        "tip_sites": [f"{b.name}__site_right_tip_{i}" for i in range(3)],
        "pad_sites": [f"{b.name}__site_pad_1_{i}" for i in range(3)],
        "gsg_pitch_m": PITCH,
        "needle_radius_m": TIP_RADIUS,
        "needle_radius_basis": "Authored geometric estimate; no OEM tip-radius measurement",
        "tip_spacing_basis": "Official installed photo label 40A-GSG-100-DP-N; manufacturer naming specifies pitch in micrometres",
        "manufacturer_reference": MANUFACTURER,
        "sample_attachment": "Rigid inert die on retained substrate with visible clips; attachment physics unmodeled",
        "axis_target_x_m": 0.002,
        "dry_inspection_lift_m": 0.0003,
        "no_contact_measurement": True,
    }
    b.metadata["limitations"].extend(
        [
            "Official close-up supports opposing tapered heads, diagonal coax connectors, silver chuck and rear ribbon wedge. Hidden guides, dimensions and mounting hardware are original estimates, not OEM CAD.",
            "Two original slide frames/ranges/masses/gains/force limits are preserved; ideal joints and gravity compensation are not calibrated mechanics.",
            "The true 100 micrometre GSG spacing is source-informed; 50 micrometre needle diameter and 50 by 70 micrometre inert pads are authored. No electrical topology, RF signal, contact force, scraping or calibration model.",
            "Dry inspection remains above the inert pads. The two coax and ribbon service ends are intentionally truncated; no RF source or flexible electrical feed is simulated.",
        ]
    )
