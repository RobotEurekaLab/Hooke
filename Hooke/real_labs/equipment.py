"""Procedural, articulated equipment for research laboratory reconstructions.

Geometry is original, in metres, and uses manufacturer references only where
listed.  These are representative mechanisms, not certified device twins.
Actuators expose physical loading, positioning and access operations; chemistry,
optics, fluid flow and scientific measurements require separate process models.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any
from xml.etree import ElementTree as ET

from real_labs.equipment_details import liquid_handler_details, microtome_details


_PALETTE = {
    "shell": (0.88, 0.9, 0.91, 1),
    "cream": (0.92, 0.9, 0.83, 1),
    "metal": (0.52, 0.59, 0.64, 1),
    "bright": (0.76, 0.81, 0.85, 1),
    "dark": (0.055, 0.075, 0.095, 1),
    "rubber": (0.032, 0.04, 0.048, 1),
    "blue": (0.08, 0.34, 0.57, 1),
    "cyan": (0.13, 0.62, 0.69, 1),
    "orange": (0.96, 0.4, 0.095, 1),
    "copper": (0.61, 0.28, 0.12, 1),
    "glass": (0.39, 0.69, 0.76, 0.14),
    "lens": (0.055, 0.24, 0.32, 1),
    "screen": (0.025, 0.12, 0.18, 1),
    "led": (0.24, 0.88, 0.7, 1),
    "warning": (0.96, 0.73, 0.14, 1),
    "plant": (0.18, 0.42, 0.11, 1),
    "sample": (0.74, 0.3, 0.37, 1),
    "guard_red": (0.72, 0.035, 0.055, 1),
}


SOURCES = {
    "liquid_handler": {
        "reference": "Opentrons Flex external envelope; original gantry model",
        "url": "https://docs.opentrons.com/flex/installation/requirements/",
        "dimensions_m": [0.87, 0.69, 0.84],
        "dimension_basis": "Published W/D/H; internal mechanism approximated",
    },
    "sequencer": {
        "reference": "Illumina NovaSeq X external envelope",
        "url": "https://support-docs.illumina.com/IN/NovaSeqX/Content/LabRequirements.htm",
        "dimensions_m": [0.864, 0.933, 1.588],
        "dimension_basis": "Published W/D/H including monitor; compartments approximated",
    },
    "microfluidic_station": {
        "reference": "Nikon Ti2 inverted optical arrangement and motorized stage",
        "url": "https://www.microscope.healthcare.nikon.com/products/inverted-microscopes/eclipse-ti2-i/specifications",
        "dimensions_m": [0.77, 0.63, 0.74],
        "dimension_basis": "Assembly estimate; XY stroke ±57/±36.5 mm and focus stroke 7 mm from Ti2-I specification",
    },
    "isotope_analyzer": {
        "reference": "Picarro L21x0-i water-isotope analysis workflow",
        "url": "https://www.picarro.com/sites/default/files/manuals/L21x0-i-Installation-Operation-Manual-40035-Rev_G.pdf",
        "dimensions_m": [0.92, 0.57, 0.57],
        "dimension_basis": "Estimated analyzer/autosampler assembly; carousel is a generic mechanism",
    },
    "deposition_chamber": {
        "reference": "Representative research vacuum deposition system",
        "url": "https://engineering.virginia.edu/department/materials-science-and-engineering/academics/graduate-programs/mse-lab-tours",
        "dimensions_m": [0.78, 0.73, 0.85],
        "dimension_basis": "Estimated; no claim of exact UVA equipment or geometry",
    },
    "phenotyping_booth": {
        "reference": "Purdue AAPF conveyor-based plant imaging workflow",
        "url": "https://ag.purdue.edu/aapf/virtual-tour.html",
        "dimensions_m": [1.44, 1.8, 2.35],
        "dimension_basis": "Estimated portal, conveyor and camera arrangement",
    },
    "magnetic_microrobot": {
        "reference": "Representative optical magnetic microrobot workstation",
        "url": "https://www.epfl.ch/research/domains/robotics/virtual-lab-tour/",
        "dimensions_m": [0.7, 0.63, 0.74],
        "dimension_basis": "Estimated; generic microscope and orthogonal coil assembly",
    },
    "sample_processor": {
        "reference": "Leica HistoCore BIOCUT envelope and specimen travel; generic motorized carriage",
        "url": "https://www.leicabiosystems.com/us/histology-equipment/microtomes/histocore-biocut/",
        "dimensions_m": [0.477, 0.62, 0.295],
        "dimension_basis": "Published W/D/H and 24 mm feed/70 mm stroke; motorized surrogate, not a BIOCUT control replica",
    },
    "aerosol_sampler": {
        "reference": "TSI DustTrak DRX 8533 monitor with generic sampling manifold",
        "url": "https://tsi.com/products/aerosol-and-dust-monitors/aerosol-and-dust-monitors/dusttrak-drx-aerosol-monitor-8533",
        "dimensions_m": [0.59, 0.45, 0.64],
        "dimension_basis": "Assembly estimate; monitor 0.216×0.224×0.135 m and 37 mm filter from manufacturer datasheet",
    },
    "rover": {
        "reference": "NASA SLOPE and JPL mobility/sample-acquisition concepts",
        "url": "https://www.nasa.gov/glenn/facilities/simulated-lunar-operations/",
        "dimensions_m": [1.12, 0.88, 1.18],
        "dimension_basis": "Original small six-wheel rover; not a NASA flight-hardware replica",
    },
    "robot_arm": {
        "reference": "Representative collaborative robot manipulation system",
        "url": "https://uwaterloo.ca/robohub/",
        "dimensions_m": [0.82, 0.26, 0.91],
        "dimension_basis": "Original generic six-axis manipulator; no manufacturer accuracy claim",
    },
    "mobile_robot": {
        "reference": "Representative laboratory mobile robot",
        "url": "https://uwaterloo.ca/robohub/",
        "dimensions_m": [0.66, 0.51, 0.59],
        "dimension_basis": "Original differential-drive platform; footprint estimated",
    },
}


def _numbers(values: Sequence[float]) -> str:
    return " ".join(f"{v:.8g}" for v in values)


class _Builder:
    """Namespaced primitives with small, explicit articulation contracts."""

    def __init__(self, root: ET.Element, name: str) -> None:
        self.root = root
        self.name = name
        self.count = 0
        self.metadata: dict[str, Any] = {
            "joints": {},
            "actuators": {},
            "sites": {},
            "joint_ranges": {},
            "actuator_modes": {},
            "capabilities": [],
            "fidelity": "reference-informed procedural geometry; articulated mechanical surrogate",
            "limitations": [
                "No validated scientific transfer function, optics, fluid chemistry, or biological process",
                "Only documented exterior dimensions are manufacturer-derived; internal geometry is estimated",
            ],
        }
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        for key, rgba in _PALETTE.items():
            ET.SubElement(
                asset,
                "material",
                name=f"{name}__mat_{key}",
                rgba=_numbers(rgba),
                specular="0.4" if key in {"metal", "bright", "lens"} else "0.18",
                shininess="0.55" if key in {"metal", "bright", "lens"} else "0.18",
                emission="0.22" if key == "led" else "0",
            )
        actuator = root.find("actuator")
        self.actuator = (
            actuator if actuator is not None else ET.SubElement(root, "actuator")
        )

    def unique(self, key: str) -> str:
        self.count += 1
        return f"{self.name}__{key}_{self.count}"

    def geom(
        self,
        parent: ET.Element,
        shape: str,
        size: Sequence[float],
        pos: Sequence[float] = (0, 0, 0),
        material: str = "shell",
        *,
        collision: bool = False,
        **kwargs: Any,
    ) -> ET.Element:
        attrs = {
            "name": self.unique(shape),
            "type": shape,
            "size": _numbers(size),
            "material": f"{self.name}__mat_{material}",
            "contype": "1" if collision else "0",
            "conaffinity": "1" if collision else "0",
            "density": "650",
            "group": "0",
            "friction": "0.8 0.01 0.001",
        }
        if "fromto" not in kwargs:
            attrs["pos"] = _numbers(pos)
        for key, value in kwargs.items():
            attrs[key] = (
                _numbers(value) if isinstance(value, (tuple, list)) else str(value)
            )
        return ET.SubElement(parent, "geom", attrs)

    def box(
        self,
        parent: ET.Element,
        pos: Sequence[float],
        size: Sequence[float],
        material: str = "shell",
        **kwargs: Any,
    ) -> ET.Element:
        return self.geom(parent, "box", size, pos, material, **kwargs)

    def cylinder(
        self,
        parent: ET.Element,
        pos: Sequence[float],
        radius: float,
        half_height: float,
        material: str = "metal",
        **kwargs: Any,
    ) -> ET.Element:
        return self.geom(
            parent, "cylinder", (radius, half_height), pos, material, **kwargs
        )

    def rod(
        self,
        parent: ET.Element,
        start: Sequence[float],
        end: Sequence[float],
        radius: float = 0.008,
        material: str = "metal",
        **kwargs: Any,
    ) -> ET.Element:
        return self.geom(
            parent,
            "capsule",
            (radius,),
            material=material,
            fromto=tuple(start) + tuple(end),
            **kwargs,
        )

    def ring(
        self,
        parent: ET.Element,
        center: Sequence[float],
        radius: float,
        tube: float,
        material: str = "metal",
        plane: str = "xy",
        segments: int = 24,
    ) -> None:
        a, b = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        for i in range(segments):
            points = []
            for angle in (2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments):
                point = list(center)
                point[a] += radius * math.cos(angle)
                point[b] += radius * math.sin(angle)
                points.append(point)
            self.rod(parent, points[0], points[1], tube, material)

    def housing(
        self,
        parent: ET.Element,
        levels: Sequence[tuple[float, float, float, float]],
        radius: float = 0.012,
        material: str = "shell",
    ) -> None:
        """A tapered rounded shell, specified as (z, half-width, half-depth, y)."""
        vertices: list[tuple[float, float, float]] = []
        sides = 24
        for z, width, depth, center_y in levels:
            corner_radius = min(radius, width * 0.4, depth * 0.4)
            for quadrant in range(4):
                sx = 1 if quadrant in (0, 3) else -1
                sy = 1 if quadrant in (0, 1) else -1
                cx, cy = sx * (width - corner_radius), center_y + sy * (
                    depth - corner_radius
                )
                for step in range(6):
                    angle = quadrant * math.pi / 2 + step * math.pi / 10
                    vertices.append(
                        (
                            cx + corner_radius * math.cos(angle),
                            cy + corner_radius * math.sin(angle),
                            z,
                        )
                    )
        faces: list[tuple[int, int, int]] = []
        for level in range(len(levels) - 1):
            offset = level * sides
            for i in range(sides):
                nxt = (i + 1) % sides
                faces.extend(
                    (
                        (offset + i, offset + nxt, offset + sides + nxt),
                        (offset + i, offset + sides + nxt, offset + sides + i),
                    )
                )
        for i in range(1, sides - 1):
            faces.append((0, i + 1, i))
            top = (len(levels) - 1) * sides
            faces.append((top, top + i, top + i + 1))
        mesh = self.unique("housing_mesh")
        ET.SubElement(
            self.root.find("asset"),
            "mesh",
            name=mesh,
            vertex=_numbers([v for point in vertices for v in point]),
            face=" ".join(str(v) for face in faces for v in face),
        )
        self.geom(parent, "mesh", (), material=material, mesh=mesh)

    def site(
        self, parent: ET.Element, key: str, pos: Sequence[float], size: float = 0.004
    ) -> str:
        name = f"{self.name}__site_{key}"
        ET.SubElement(
            parent,
            "site",
            name=name,
            pos=_numbers(pos),
            size=str(size),
            rgba="0.15 0.8 0.6 0",
            group="4",
        )
        self.metadata["sites"][key] = name
        return name

    def moving(
        self,
        parent: ET.Element,
        key: str,
        pos: Sequence[float],
        axis: Sequence[float],
        limits: tuple[float, float] | None,
        *,
        kind: str = "slide",
        mass: float = 0.5,
        mode: str = "position",
        kp: float = 180,
        force: float = 120,
    ) -> ET.Element:
        body = ET.SubElement(
            parent,
            "body",
            name=f"{self.name}__body_{key}",
            pos=_numbers(pos),
            gravcomp="1",
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass=str(mass),
            diaginertia=_numbers([max(1e-5, mass * 0.006)] * 3),
        )
        joint = f"{self.name}__joint_{key}"
        attrs = {
            "name": joint,
            "type": kind,
            "axis": _numbers(axis),
            "damping": "2.5" if kind == "slide" else "0.8",
            "armature": "0.005",
        }
        if limits is not None:
            attrs.update(limited="true", range=_numbers(limits))
        else:
            attrs["limited"] = "false"
        ET.SubElement(body, "joint", attrs)
        actuator = f"{self.name}__actuator_{key}"
        aa = {
            "name": actuator,
            "joint": joint,
            "forcelimited": "true",
            "forcerange": _numbers((-force, force)),
        }
        if mode == "velocity":
            aa.update(kv="10", ctrllimited="true", ctrlrange="-8 8")
        else:
            aa.update(kp=str(kp), kv=str(2 * math.sqrt(kp * mass)))
            if limits is not None:
                aa.update(ctrllimited="true", ctrlrange=_numbers(limits))
        ET.SubElement(self.actuator, mode, aa)
        self.metadata["joints"][key] = joint
        self.metadata["actuators"][key] = actuator
        self.metadata["joint_ranges"][key] = list(limits) if limits else None
        self.metadata["actuator_modes"][key] = mode
        return body

    def feet(
        self, parent: ET.Element, width: float, depth: float, height: float = 0.024
    ) -> None:
        for x in (-width / 2 + 0.045, width / 2 - 0.045):
            for y in (-depth / 2 + 0.045, depth / 2 - 0.045):
                self.cylinder(
                    parent,
                    (x, y, height / 2),
                    0.018,
                    height / 2,
                    "rubber",
                    collision=True,
                )

    def vents(
        self, parent: ET.Element, pos: Sequence[float], width: float, rows: int = 6
    ) -> None:
        x, y, z = pos
        for row in range(rows):
            self.box(parent, (x, y, z + row * 0.012), (width / 2, 0.002, 0.002), "dark")

    def screen(
        self,
        parent: ET.Element,
        pos: Sequence[float],
        width: float = 0.15,
        height: float = 0.11,
    ) -> None:
        x, y, z = pos
        self.box(parent, pos, (width / 2 + 0.007, 0.009, height / 2 + 0.007), "dark")
        self.box(parent, (x, y - 0.01, z), (width / 2, 0.001, height / 2), "screen")
        self.box(
            parent,
            (x, y - 0.012, z + height * 0.33),
            (width * 0.43, 0.0007, height * 0.035),
            "cyan",
        )
        for i, frac in enumerate((0.66, 0.47, 0.79)):
            self.box(
                parent,
                (
                    x - width * (1 - frac) * 0.42,
                    y - 0.012,
                    z + height * (0.08 - i * 0.19),
                ),
                (width * 0.42 * frac, 0.0007, height * 0.024),
                "bright",
            )


def _liquid_handler(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    b.feet(base, 0.87, 0.69)
    b.box(base, (0, 0, 0.085), (0.423, 0.333, 0.053), "shell", collision=True)
    b.housing(
        base,
        [
            (0.025, 0.421, 0.331, 0),
            (0.034, 0.435, 0.345, 0),
            (0.132, 0.435, 0.345, 0),
            (0.145, 0.426, 0.336, 0),
        ],
        radius=0.02,
    )
    b.box(base, (0, 0.02, 0.152), (0.393, 0.295, 0.009), "metal", collision=True)
    b.box(base, (0, 0.329, 0.47), (0.435, 0.016, 0.31), "dark", collision=True)
    for x in (-0.414, 0.414):
        for y in (-0.319, 0.31):
            b.box(
                base,
                (x, y, 0.48),
                (0.021, 0.022, 0.3),
                "dark" if y < 0 else "shell",
                collision=True,
            )
        b.box(base, (x, 0, 0.46), (0.002, 0.292, 0.266), "glass")
        b.rod(base, (x * 0.88, -0.24, 0.64), (x * 0.88, 0.24, 0.64), 0.009)
    b.box(base, (0, 0, 0.808), (0.435, 0.345, 0.032), "shell", collision=True)
    for ix in range(3):
        for iy in range(4):
            x, y = (ix - 1) * 0.146, (iy - 1.5) * 0.109
            b.box(base, (x, y, 0.164), (0.067, 0.044, 0.003), "dark")
            b.site(base, f"deck_{iy * 3 + ix + 1}", (x, y, 0.174))
    for ix in range(2):
        px, py = (ix - 1) * 0.146, -0.1635
        b.box(base, (px, py, 0.176), (0.0639, 0.0428, 0.011), "cream", collision=True)
        for row in range(8):
            for col in range(12):
                b.cylinder(
                    base,
                    (px + (col - 5.5) * 0.009, py + (row - 3.5) * 0.009, 0.188),
                    0.003,
                    0.001,
                    "dark",
                )
    gantry_y = b.moving(
        base, "gantry_y", (0, 0, 0.653), (0, 1, 0), (-0.22, 0.22), mass=3
    )
    b.box(gantry_y, (0, 0, 0), (0.36, 0.035, 0.039), "bright")
    b.rod(gantry_y, (-0.34, -0.024, -0.013), (0.34, -0.024, -0.013), 0.007, "dark")
    carriage = b.moving(
        gantry_y, "gantry_x", (0, 0, -0.035), (1, 0, 0), (-0.28, 0.28), mass=1.2
    )
    b.box(carriage, (0, 0, 0), (0.048, 0.046, 0.05), "dark")
    pipette = b.moving(
        carriage, "pipette_z", (0, -0.025, -0.055), (0, 0, -1), (0, 0.24), mass=0.35
    )
    b.box(pipette, (0, 0, -0.044), (0.034, 0.034, 0.059), "shell")
    for i in range(8):
        x = (i - 3.5) * 0.009
        b.site(pipette, f"tip_{i + 1}", (x, -0.006, -0.149), 0.001)
    door = b.moving(base, "door", (0, -0.341, 0.2), (0, 0, 1), (0, 0.52), mass=1)
    b.box(door, (0, 0, 0.24), (0.38, 0.003, 0.236), "glass")
    for x in (-0.384, 0.384):
        b.box(door, (x, 0, 0.24), (0.01, 0.008, 0.24), "metal")
    b.box(door, (0, -0.005, 0.005), (0.384, 0.01, 0.012), "dark")
    b.rod(door, (-0.07, -0.025, 0.055), (0.07, -0.025, 0.055), 0.009, "dark")
    b.site(door, "door_handle", (0, -0.025, 0.055))
    # Set the display ahead of the sliding-door sweep, on a frame-mounted arm.
    b.screen(base, (0.305, -0.386, 0.493), 0.148, 0.091)
    b.vents(base, (0, -0.346, 0.055), 0.21, 4)
    liquid_handler_details(b, base, gantry_y, carriage, pipette, door)
    b.metadata["capabilities"] = [
        "XYZ liquid-tool positioning",
        "eight tip target sites",
        "12 deck locations",
        "access-door actuation",
    ]
    b.metadata["limitations"].append(
        "The retained sliding access door is a control surrogate; the Flex reference uses a hinged door"
    )


def _sequencer(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    b.feet(base, 0.864, 0.933)
    # Separate shell panels preserve clearance for real sample loading.
    b.box(base, (0, 0.018, 0.045), (0.432, 0.4485, 0.018), "shell", collision=True)
    b.box(base, (0, 0.447, 0.68), (0.432, 0.02, 0.65), "shell", collision=True)
    for x in (-0.417, 0.417):
        b.box(base, (x, 0.018, 0.68), (0.015, 0.429, 0.65), "shell", collision=True)
    b.box(base, (0, -0.438, 0.34), (0.413, 0.007, 0.3), "dark")
    b.box(base, (0, -0.448, 0.95), (0.411, 0.01, 0.345), "dark")
    b.box(base, (0, 0.01, 1.325), (0.429, 0.443, 0.012), "blue")
    for x in (-0.407, 0.407):
        b.box(base, (x, -0.457, 0.72), (0.015, 0.008, 0.602), "shell")
    for ix, x in enumerate((-0.205, 0.205)):
        drawer = b.moving(
            base,
            f"reagent_drawer_{ix + 1}",
            (x, -0.43, 0.365),
            (0, -1, 0),
            (0, 0.38),
            mass=4,
        )
        b.box(drawer, (0, -0.02, 0), (0.183, 0.015, 0.255), "shell")
        b.box(drawer, (0, 0.155, -0.235), (0.177, 0.18, 0.009), "metal", collision=True)
        b.box(drawer, (0, 0.11, -0.05), (0.145, 0.13, 0.18), "cream", collision=True)
        b.box(drawer, (0, -0.038, 0.17), (0.067, 0.005, 0.012), "dark")
        b.site(drawer, f"reagent_cartridge_{ix + 1}", (0, 0.11, 0.14))
    tray = b.moving(
        base, "flowcell_tray", (0, -0.43, 0.93), (0, -1, 0), (0, 0.25), mass=2
    )
    b.box(tray, (0, -0.025, 0), (0.385, 0.015, 0.155), "shell")
    b.box(tray, (0, 0.095, -0.045), (0.355, 0.135, 0.008), "metal", collision=True)
    for i, x in enumerate((-0.17, 0.17)):
        b.box(tray, (x, 0.07, -0.03), (0.067, 0.102, 0.005), "lens", collision=True)
        for col in range(4):
            b.box(
                tray,
                (x + (col - 1.5) * 0.025, 0.07, -0.024),
                (0.002, 0.08, 0.0008),
                "cyan",
            )
        b.site(tray, f"flowcell_{i + 1}", (x, 0.07, -0.018))
    b.box(base, (0.27, -0.02, 1.42), (0.021, 0.022, 0.081), "dark")
    b.screen(base, (0.2, -0.07, 1.487), 0.324, 0.188)
    b.vents(base, (0, -0.449, 0.08), 0.46, 6)
    b.box(base, (-0.35, -0.46, 1.255), (0.011, 0.001, 0.007), "led")
    b.metadata["capabilities"] = [
        "two reagent drawers",
        "dual flow-cell loading tray",
        "cartridge and flow-cell target sites",
    ]


def _microscope(b: _Builder, base: ET.Element) -> ET.Element:
    """Inverted optical layout: objective below stage, condenser above specimen."""
    b.feet(base, 0.38, 0.5)
    b.housing(
        base,
        [
            (0.026, 0.181, 0.231, 0.015),
            (0.038, 0.185, 0.24, 0.015),
            (0.075, 0.18, 0.226, 0.025),
            (0.145, 0.144, 0.164, 0.065),
        ],
        radius=0.02,
    )
    b.box(base, (0, 0.025, 0.075), (0.15, 0.19, 0.047), "shell", collision=True)
    b.housing(
        base,
        [(0.032, 0.184, 0.235, 0.015), (0.039, 0.184, 0.235, 0.015)],
        radius=0.02,
        material="rubber",
    )
    b.box(base, (0, 0.183, 0.33), (0.095, 0.055, 0.204), "shell", collision=True)
    b.box(base, (0, 0.218, 0.51), (0.065, 0.022, 0.159), "metal")
    b.housing(
        base,
        [
            (0.63, 0.067, 0.117, 0.115),
            (0.645, 0.071, 0.118, 0.115),
            (0.718, 0.06, 0.09, 0.139),
        ],
        radius=0.013,
    )
    b.vents(base, (0, 0.23, 0.648), 0.084, 5)
    b.cylinder(base, (0, 0, 0.61), 0.033, 0.03, "dark")
    b.cylinder(base, (0, 0, 0.572), 0.022, 0.019, "bright")
    b.cylinder(base, (0, 0, 0.55), 0.019, 0.004, "lens")
    b.box(base, (0.089, 0.163, 0.345), (0.003, 0.079, 0.072), "dark")
    for x in (-0.16, 0.16):
        b.cylinder(
            base,
            (x, -0.004, 0.13),
            0.034,
            0.02,
            "dark",
            quat=(0.70710678, 0, 0.70710678, 0),
        )
        b.cylinder(
            base,
            (x * 1.16, -0.004, 0.13),
            0.014,
            0.009,
            "bright",
            quat=(0.70710678, 0, 0.70710678, 0),
        )
    b.box(
        base,
        (0, -0.158, 0.213),
        (0.065, 0.079, 0.04),
        "shell",
        quat=(0.976296, 0.21644, 0, 0),
    )
    for x in (-0.037, 0.037):
        b.rod(base, (x, -0.195, 0.226), (x, -0.273, 0.307), 0.021, "dark")
        b.rod(base, (x, -0.265, 0.297), (x, -0.294, 0.328), 0.025, "rubber")
    b.box(base, (0.191, 0.083, 0.174), (0.052, 0.045, 0.043), "dark")
    b.cylinder(
        base,
        (0.137, 0.083, 0.174),
        0.023,
        0.02,
        "metal",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    focus = b.moving(
        base, "focus", (0, 0, 0.291), (0, 0, 1), (-0.003, 0.004), mass=0.2, kp=600
    )
    b.cylinder(focus, (0, 0.042, 0), 0.063, 0.015, "dark")
    for i in range(5):
        a = i * 2 * math.pi / 5
        x, y = 0.042 * math.sin(a), 0.042 - 0.042 * math.cos(a)
        b.cylinder(focus, (x, y, 0.035), 0.012, 0.02, "bright")
        b.cylinder(focus, (x, y, 0.046), 0.0125, 0.003, "blue" if i % 2 else "warning")
        b.cylinder(focus, (x, y, 0.059), 0.007, 0.004, "lens")
    stage_x = b.moving(
        base, "stage_x", (0, 0, 0.324), (1, 0, 0), (-0.057, 0.057), mass=0.5, kp=500
    )
    for y in (-0.085, 0.085):
        b.box(stage_x, (0, y, 0), (0.17, 0.014, 0.012), "dark")
    stage_y = b.moving(
        stage_x,
        "stage_y",
        (0, 0, 0.021),
        (0, 1, 0),
        (-0.0365, 0.0365),
        mass=0.2,
        kp=500,
    )
    for x in (-0.104, 0.104):
        b.box(stage_y, (x, 0, 0), (0.031, 0.079, 0.007), "dark", collision=True)
    for y in (-0.066, 0.066):
        b.box(stage_y, (0, y, 0), (0.074, 0.013, 0.007), "dark", collision=True)
    b.ring(stage_y, (0, 0, 0.007), 0.027, 0.003, "bright")
    b.site(stage_y, "specimen", (0, 0, 0.016), 0.001)
    ET.SubElement(
        base,
        "camera",
        name=f"{b.name}__microscope_camera",
        pos="0 0 0.53",
        quat="1 0 0 0",
        fovy="11",
    )
    b.metadata["cameras"] = {"microscope": f"{b.name}__microscope_camera"}
    return stage_y


def _microfluidic_station(
    b: _Builder, base: ET.Element, params: dict[str, Any]
) -> None:
    stage = _microscope(b, base)
    b.box(stage, (0, 0, 0.012), (0.0375, 0.0125, 0.0007), "glass", collision=True)
    for y in (-0.003, 0.003):
        b.rod(stage, (-0.031, y, 0.013), (0.028, y, 0.013), 0.00035, "cyan")
    for x, key in ((-0.031, "chip_inlet"), (0.028, "chip_outlet")):
        b.cylinder(stage, (x, 0, 0.016), 0.0025, 0.003, "cream")
        b.site(stage, key, (x, 0, 0.019), 0.001)
    b.box(base, (0.35, 0.068, 0.09), (0.13, 0.122, 0.075), "shell", collision=True)
    b.screen(base, (0.35, -0.058, 0.111), 0.155, 0.075)
    for i in range(4):
        x = 0.275 + i * 0.05
        b.cylinder(base, (x, 0.075, 0.185), 0.011, 0.021, "metal")
        b.cylinder(base, (x, 0.075, 0.21), 0.012, 0.003, "blue")
        b.site(base, f"pressure_port_{i + 1}", (x, 0.075, 0.216))
    b.rod(base, (0.275, 0.075, 0.218), (0.24, 0.05, 0.36), 0.002, "cyan")
    b.rod(base, (0.24, 0.05, 0.36), (0.05, 0.01, 0.375), 0.002, "cyan")
    valve = b.moving(
        base,
        "flow_valve",
        (0.435, -0.063, 0.037),
        (0, 1, 0),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.02,
    )
    b.cylinder(
        valve, (0, 0, 0), 0.016, 0.01, "dark", quat=(0.70710678, 0.70710678, 0, 0)
    )
    b.box(valve, (0, -0.012, 0), (0.012, 0.002, 0.002), "bright")
    b.metadata["capabilities"] = [
        "XY specimen positioning",
        "7 mm objective focus",
        "microfluidic chip inlet/outlet targets",
        "flow-valve command",
        "top-view camera",
    ]


def _isotope_analyzer(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    for xs, ys in (((-0.415, -0.025), (-0.24, 0.24)), ((0.14, 0.42), (-0.19, 0.2))):
        for x in xs:
            for y in ys:
                b.cylinder(
                    base, (x, y, 0.0125), 0.018, 0.0125, "rubber", collision=True
                )
    b.box(base, (-0.22, 0, 0.17), (0.24, 0.285, 0.145), "shell", collision=True)
    b.box(base, (-0.22, -0.288, 0.177), (0.233, 0.005, 0.136), "dark")
    b.screen(base, (-0.27, -0.298, 0.225), 0.22, 0.105)
    b.vents(base, (-0.22, -0.297, 0.06), 0.36, 6)
    b.box(base, (0.015, 0.12, 0.35), (0.007, 0.09, 0.037), "shell")
    b.cylinder(base, (-0.03, 0.14, 0.338), 0.025, 0.026, "dark")
    b.site(base, "vaporizer_inlet", (-0.03, 0.14, 0.37))
    b.box(base, (0.28, 0.005, 0.07), (0.18, 0.24, 0.045), "shell", collision=True)
    carousel = b.moving(
        base,
        "carousel",
        (0.27, -0.04, 0.138),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.7,
    )
    b.cylinder(carousel, (0, 0, 0), 0.154, 0.018, "dark")
    b.cylinder(carousel, (0, 0, 0.025), 0.037, 0.014, "bright")
    for i in range(20):
        a = i * math.tau / 20
        x, y = 0.125 * math.cos(a), 0.125 * math.sin(a)
        if i != 0:
            b.cylinder(carousel, (x, y, 0.044), 0.006, 0.024, "glass")
            b.cylinder(carousel, (x, y, 0.07), 0.0064, 0.003, "blue")
        else:
            b.cylinder(carousel, (x, y, 0.02), 0.008, 0.002, "metal", collision=True)
            b.site(carousel, "sample_seat", (x, y, 0.047), 0.002)
        b.site(carousel, f"vial_{i + 1}", (x, y, 0.074), 0.002)
    b.box(base, (0.28, 0.174, 0.303), (0.031, 0.035, 0.231), "metal")
    b.box(base, (0.28, 0.006, 0.516), (0.036, 0.196, 0.025), "shell")
    needle = b.moving(
        base, "injection_z", (0.27, -0.165, 0.434), (0, 0, -1), (0, 0.23), mass=0.1
    )
    b.box(needle, (0, 0, 0), (0.02, 0.021, 0.044), "blue")
    b.rod(needle, (0, 0, -0.045), (0, 0, -0.117), 0.0015, "bright")
    b.site(needle, "needle_tip", (0, 0, -0.118), 0.001)
    b.rod(base, (0.26, 0.17, 0.54), (0.04, 0.18, 0.57), 0.003, "copper")
    b.rod(base, (0.04, 0.18, 0.57), (-0.03, 0.14, 0.37), 0.003, "copper")
    b.metadata["capabilities"] = [
        "20 indexed vial target sites",
        "autosampler carousel",
        "injection needle translation",
        "vaporizer inlet target",
    ]


def _deposition_chamber(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    b.feet(base, 0.78, 0.73, height=0.025)
    b.box(base, (0, 0, 0.065), (0.39, 0.365, 0.04), "dark", collision=True)
    for x in (-0.19, 0.19):
        b.box(base, (x, 0, 0.185), (0.022, 0.18, 0.085), "metal")
    # Open cylindrical chamber built from tangent panels, leaving a real loading cavity.
    for i in range(32):
        a = i * math.tau / 32
        b.box(
            base,
            (0.24 * math.cos(a), 0.24 * math.sin(a), 0.43),
            (0.014, 0.025, 0.155),
            "bright",
            quat=(math.cos(a / 2), 0, 0, math.sin(a / 2)),
            collision=True,
        )
    b.cylinder(base, (0, 0, 0.278), 0.25, 0.012, "metal", collision=True)
    for z in (0.285, 0.583):
        b.ring(base, (0, 0, z), 0.256, 0.014, "metal")
        for i in range(12):
            a = i * math.tau / 12
            b.cylinder(
                base,
                (0.256 * math.cos(a), 0.256 * math.sin(a), z + 0.014),
                0.006,
                0.006,
                "dark",
            )
    b.cylinder(
        base,
        (0, -0.255, 0.445),
        0.064,
        0.036,
        "metal",
        quat=(0.70710678, 0.70710678, 0, 0),
    )
    b.cylinder(
        base,
        (0, -0.294, 0.445),
        0.048,
        0.002,
        "lens",
        quat=(0.70710678, 0.70710678, 0, 0),
    )
    lid = b.moving(
        base,
        "chamber_lid",
        (0, 0.285, 0.617),
        (1, 0, 0),
        (-1.7, 0),
        kind="hinge",
        mass=3,
    )
    b.cylinder(lid, (0, -0.285, 0), 0.26, 0.014, "bright", collision=True)
    b.rod(lid, (-0.07, -0.46, 0.045), (0.07, -0.46, 0.045), 0.012, "dark")
    b.site(lid, "lid_handle", (0, -0.46, 0.045))
    platen = b.moving(
        base,
        "platen",
        (0, 0, 0.341),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.4,
    )
    b.cylinder(platen, (0, 0, 0), 0.11, 0.009, "dark")
    b.cylinder(platen, (0, 0, 0.013), 0.05, 0.002, "lens")
    # A supported platen core gives thin coupons robust box contact at SI scale.
    b.box(platen, (0, 0, -0.0025), (0.042, 0.042, 0.0175), "dark", collision=True)
    b.site(platen, "substrate", (0, 0, 0.019))
    b.cylinder(base, (0.34, 0.14, 0.286), 0.059, 0.16, "metal")
    b.rod(base, (0.2, 0.14, 0.4), (0.34, 0.14, 0.4), 0.03, "metal")
    for z in (0.18, 0.205, 0.23, 0.255):
        b.ring(base, (0.34, 0.14, z), 0.058, 0.005, "dark", segments=16)
    b.box(base, (-0.298, 0.18, 0.48), (0.075, 0.105, 0.12), "shell")
    b.screen(base, (-0.298, 0.07, 0.5), 0.115, 0.09)
    valve = b.moving(
        base,
        "vacuum_valve",
        (0.29, -0.08, 0.485),
        (0, 0, 1),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.1,
    )
    b.box(valve, (0, 0, 0), (0.042, 0.009, 0.007), "blue")
    b.site(base, "gas_inlet", (-0.24, 0.05, 0.5))
    b.metadata["capabilities"] = [
        "hinged vacuum-chamber lid",
        "rotating substrate platen",
        "vacuum valve angle",
        "substrate loading target",
    ]


def _phenotyping_booth(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    for x in (-0.66, 0.66):
        for y in (-0.6, 0.6):
            b.box(base, (x, y, 0.005), (0.049, 0.049, 0.005), "rubber", collision=True)
            b.box(base, (x, y, 1.18), (0.045, 0.045, 1.17), "shell", collision=True)
        b.box(base, (x, 0, 2.29), (0.05, 0.65, 0.06), "shell")
        b.box(base, (x, 0, 1.29), (0.012, 0.55, 0.84), "dark")
        for y in (-0.45, 0.45):
            b.box(base, (x * 0.95, y, 1.42), (0.012, 0.025, 0.68), "cream")
    b.box(base, (0, 0.62, 1.48), (0.61, 0.025, 0.76), "dark")
    b.box(base, (0, 0, 2.305), (0.64, 0.64, 0.045), "shell")
    for x in (-0.28, 0.28):
        b.box(base, (x, 0, 2.246), (0.055, 0.45, 0.012), "cream")
    for x in (-0.4, 0.4):
        b.box(base, (x, 0, 0.58), (0.024, 0.9, 0.07), "metal", collision=True)
        for y in (-0.72, 0.72):
            b.cylinder(base, (x, y, 0.005), 0.033, 0.005, "rubber", collision=True)
            b.box(base, (x, y, 0.27), (0.025, 0.025, 0.26), "metal")
    for i in range(19):
        y = -0.84 + i * 0.09
        b.cylinder(
            base,
            (0, y, 0.614),
            0.027,
            0.365,
            "bright",
            quat=(0.70710678, 0, 0.70710678, 0),
        )
    # Keep the specimen inside the rear backdrop at full travel. The original
    # 1.08 m surrogate stroke let the pot/canopy pass through the enclosure.
    carrier = b.moving(
        base, "conveyor", (0, -0.54, 0.66), (0, 1, 0), (0, 0.80), mass=3, kp=300
    )
    b.box(carrier, (0, 0, 0), (0.255, 0.255, 0.025), "dark", collision=True)
    turn = b.moving(
        carrier,
        "plant_turntable",
        (0, 0, 0.047),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=1,
    )
    b.cylinder(turn, (0, 0, 0), 0.2, 0.018, "metal", collision=True)
    b.site(turn, "plant_pot", (0, 0, 0.025))
    # Pot and leaves stay visible as a replaceable default specimen.
    if params.get("include_sample", True):
        b.cylinder(turn, (0, 0, 0.115), 0.116, 0.085, "dark")
        b.cylinder(turn, (0, 0, 0.202), 0.11, 0.003, "copper")
        b.rod(turn, (0, 0, 0.2), (0.015, 0, 0.76), 0.005, "plant")
        for i in range(7):
            a = i * 2.4
            z = 0.27 + i * 0.062
            end = (0.2 * math.cos(a), 0.2 * math.sin(a), z + 0.09)
            b.rod(turn, (0, 0, z), end, 0.004, "plant")
            b.geom(
                turn,
                "ellipsoid",
                (0.085, 0.025, 0.005),
                end,
                "plant",
                quat=(math.cos(a / 2), 0, 0, math.sin(a / 2)),
            )
    camera = b.moving(
        base, "camera_height", (0.46, 0.05, 1.13), (0, 0, 1), (-0.18, 0.52), mass=0.6
    )
    b.box(camera, (0, 0, 0), (0.063, 0.066, 0.064), "dark")
    b.cylinder(
        camera,
        (-0.08, 0, 0),
        0.033,
        0.028,
        "metal",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    b.cylinder(
        camera,
        (-0.112, 0, 0),
        0.029,
        0.003,
        "lens",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    b.site(camera, "camera_optical_center", (-0.116, 0, 0))
    ET.SubElement(
        camera,
        "camera",
        name=f"{b.name}__plant_camera",
        pos="-0.118 0 0",
        quat="0.70710678 0 0.70710678 0",
        fovy="65",
    )
    b.screen(base, (0.665, -0.63, 1.55), 0.19, 0.14)
    b.metadata["cameras"] = {"plant_side": f"{b.name}__plant_camera"}
    b.metadata["capabilities"] = [
        "plant-carrier conveyor translation",
        "plant rotation",
        "camera-height positioning",
        "side-view camera",
    ]


def _magnetic_microrobot(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    stage = _microscope(b, base)
    b.cylinder(stage, (0, 0, 0.014), 0.02, 0.001, "glass", collision=True)
    b.ring(stage, (0, 0, 0.02), 0.02, 0.001, "cream", segments=24)
    for x in (-0.069, 0.069):
        for dx in (-0.006, 0, 0.006):
            b.ring(
                base,
                (x + dx, 0, 0.379),
                0.043,
                0.0025,
                "copper",
                plane="yz",
                segments=20,
            )
        b.box(base, (x, 0.063, 0.364), (0.02, 0.016, 0.018), "dark")
    for y in (-0.07, 0.07):
        for dy in (-0.006, 0, 0.006):
            b.ring(
                base,
                (0, y + dy, 0.379),
                0.043,
                0.0025,
                "copper",
                plane="xz",
                segments=20,
            )
    for z in (0.354, 0.415):
        b.ring(base, (0, 0, z), 0.055, 0.0035, "copper", segments=24)
    b.box(base, (0.34, 0.095, 0.083), (0.14, 0.13, 0.06), "shell")
    b.screen(base, (0.34, -0.037, 0.095), 0.16, 0.065)
    for i in range(3):
        b.cylinder(
            base,
            (0.255 + i * 0.08, -0.042, 0.038),
            0.008,
            0.005,
            "copper",
            quat=(0.70710678, 0.70710678, 0, 0),
        )
    # Microscale marker is real-size; detailed optical view requires a magnified camera.
    b.geom(
        stage,
        "capsule",
        (0.00006,),
        material="dark",
        fromto=(-0.0002, 0, 0.017, 0.0002, 0, 0.017),
    )
    b.site(stage, "micro_agent", (0, 0, 0.017), 0.0001)
    b.metadata["capabilities"] = [
        "XY specimen positioning",
        "objective focus",
        "optical sample camera",
        "three-axis coil geometry and specimen targets",
    ]
    b.metadata["limitations"].append(
        "Coils are geometric references only; no electromagnetic force or microrobot locomotion model"
    )


def _sample_processor(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    # The handwheel extends beyond the cast chassis. Support the chassis,
    # rather than placing feet at the full instrument-envelope corners.
    feet = ET.SubElement(base, "body", name=b.unique("support_feet"), pos="-0.035 0 0")
    b.feet(feet, 0.36, 0.57, height=0.03)
    # A smaller collision core stays inside the tapered cast housing. The
    # instrument front must remain open around the travelling specimen clamp.
    b.box(base, (-0.035, 0.094, 0.132), (0.147, 0.181, 0.105), "shell", collision=True)
    b.box(base, (-0.035, -0.223, 0.035), (0.155, 0.071, 0.012), "shell", collision=True)
    wheel = b.moving(
        base,
        "handwheel",
        (0.196, 0.09, 0.182),
        (1, 0, 0),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.8,
    )
    b.cylinder(
        wheel, (0, 0, 0), 0.101, 0.009, "shell", quat=(0.70710678, 0, 0.70710678, 0)
    )
    b.cylinder(
        wheel, (0.009, 0, 0), 0.084, 0.002, "shell", quat=(0.70710678, 0, 0.70710678, 0)
    )
    b.rod(wheel, (0.018, -0.078, 0), (0.043, -0.078, 0), 0.013, "dark")
    feed = b.moving(
        base,
        "specimen_feed",
        (-0.035, -0.145, 0.196),
        (0, -1, 0),
        (0, 0.024),
        mass=0.6,
        kp=1000,
    )
    stroke = b.moving(
        feed, "section_stroke", (0, 0, 0), (0, 0, -1), (0, 0.07), mass=0.3, kp=1000
    )
    b.box(stroke, (0, -0.015, 0), (0.038, 0.018, 0.026), "metal")
    b.box(stroke, (0, -0.036, 0), (0.0275, 0.004, 0.021), "cream")
    b.box(stroke, (0, -0.041, 0), (0.018, 0.001, 0.012), "sample")
    b.site(stroke, "specimen_block", (0, -0.043, 0))
    b.box(base, (-0.035, -0.211, 0.087), (0.076, 0.024, 0.021), "metal")
    b.box(base, (-0.035, -0.19, 0.112), (0.06, 0.002, 0.012), "bright")
    b.site(base, "blade_edge", (-0.035, -0.19, 0.124), 0.002)
    guard = b.moving(
        base,
        "blade_guard",
        (-0.095, -0.2, 0.127),
        (1, 0, 0),
        (0, 1.65),
        kind="hinge",
        mass=0.03,
    )
    b.rod(guard, (0, 0, 0), (0.12, 0, 0), 0.003, "guard_red")
    b.site(guard, "guard_handle", (0.115, 0, 0))
    b.vents(base, (-0.034, 0.294, 0.08), 0.2, 6)
    microtome_details(b, base, stroke, guard, wheel)
    b.metadata["capabilities"] = [
        "24 mm block feed",
        "70 mm sectioning carriage",
        "blade guard",
        "handwheel articulation",
        "specimen and blade alignment sites",
    ]
    b.metadata["limitations"].append(
        "No deformable tissue sectioning or coupled handwheel-drive transmission"
    )


def _aerosol_sampler(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    for x in (-0.21, -0.08):
        for y in (-0.06, 0.09):
            b.cylinder(base, (x, y, 0.0085), 0.011, 0.0085, "rubber", collision=True)
    for x in (0.10, 0.25):
        for y in (0.02, 0.15):
            b.cylinder(base, (x, y, 0.0025), 0.012, 0.0025, "rubber", collision=True)
    b.box(base, (-0.14, 0.016, 0.0845), (0.108, 0.112, 0.0675), "blue")
    b.box(base, (-0.14, 0.016, 0.08), (0.107, 0.111, 0.062), "blue", collision=True)
    b.screen(base, (-0.14, -0.1, 0.094), 0.132, 0.077)
    b.box(base, (-0.14, 0.018, 0.157), (0.077, 0.023, 0.009), "dark")
    b.cylinder(base, (-0.2, 0.08, 0.182), 0.008, 0.025, "metal")
    b.rod(base, (-0.2, 0.08, 0.203), (-0.22, 0.08, 0.3), 0.004, "dark")
    b.rod(base, (-0.22, 0.08, 0.3), (0.17, 0.08, 0.34), 0.004, "dark")
    b.box(base, (0.176, 0.083, 0.025), (0.115, 0.13, 0.02), "metal")
    b.rod(base, (0.245, 0.125, 0.04), (0.245, 0.125, 0.6), 0.008, "metal")
    b.rod(base, (0.245, 0.125, 0.55), (0.17, 0.08, 0.55), 0.008, "metal")
    b.cylinder(base, (0.17, 0.08, 0.34), 0.024, 0.09, "metal")
    b.cylinder(base, (0.17, 0.08, 0.468), 0.038, 0.038, "bright")
    b.cylinder(base, (0.17, 0.08, 0.519), 0.018, 0.017, "metal")
    b.cylinder(base, (0.17, 0.08, 0.56), 0.051, 0.015, "metal")
    b.cylinder(base, (0.17, 0.08, 0.582), 0.056, 0.005, "dark")
    b.site(base, "sampling_inlet", (0.17, 0.08, 0.54))
    drawer = b.moving(
        base, "filter_drawer", (-0.14, -0.025, 0.156), (0, -1, 0), (0, 0.14), mass=0.15
    )
    b.box(drawer, (0, 0, 0), (0.054, 0.059, 0.009), "dark", collision=True)
    b.cylinder(drawer, (0, 0, 0.011), 0.0185, 0.002, "cream")
    b.ring(drawer, (0, 0, 0.014), 0.021, 0.002, "metal", segments=16)
    b.site(drawer, "filter", (0, 0, 0.017))
    b.box(base, (0.005, -0.083, 0.035), (0.035, 0.035, 0.025), "shell")
    valve = b.moving(
        base,
        "flow_valve",
        (0.005, -0.083, 0.075),
        (0, 0, 1),
        (0, math.pi),
        kind="hinge",
        mass=0.02,
    )
    b.cylinder(valve, (0, 0, 0), 0.018, 0.013, "dark")
    b.box(valve, (0, 0, 0.014), (0.014, 0.002, 0.001), "bright")
    b.metadata["capabilities"] = [
        "37 mm gravimetric filter drawer",
        "flow-valve angle",
        "sampling-inlet and filter target sites",
    ]


def _arm(b: _Builder, base: ET.Element, prefix: str = "", scale: float = 1.0) -> None:
    def point(x: float, y: float, z: float) -> tuple[float, float, float]:
        return x * scale, y * scale, z * scale

    b.cylinder(
        base, point(0, 0, 0.052), 0.095 * scale, 0.05 * scale, "dark", collision=True
    )
    yaw = b.moving(
        base,
        prefix + "base_yaw",
        point(0, 0, 0.115),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=3 * scale,
        kp=250,
    )
    b.cylinder(yaw, point(0, 0, 0), 0.073 * scale, 0.028 * scale, "blue")
    shoulder = b.moving(
        yaw,
        prefix + "shoulder",
        point(0, 0, 0.075),
        (0, 1, 0),
        (-1.65, 1.65),
        kind="hinge",
        mass=2 * scale,
        kp=250,
    )
    b.cylinder(
        shoulder,
        point(0, 0, 0),
        0.065 * scale,
        0.068 * scale,
        "blue",
        quat=(0.70710678, 0.70710678, 0, 0),
    )
    b.rod(
        shoulder,
        point(0, 0, 0),
        point(0.21, 0, 0.25),
        0.046 * scale,
        "shell",
        collision=True,
    )
    b.rod(
        shoulder,
        point(0.014, -0.045, 0.018),
        point(0.2, -0.045, 0.24),
        0.008 * scale,
        "dark",
    )
    elbow = b.moving(
        shoulder,
        prefix + "elbow",
        point(0.21, 0, 0.25),
        (0, 1, 0),
        (-2.15, 2.15),
        kind="hinge",
        mass=1.4 * scale,
        kp=220,
    )
    b.cylinder(
        elbow,
        point(0, 0, 0),
        0.051 * scale,
        0.056 * scale,
        "blue",
        quat=(0.70710678, 0.70710678, 0, 0),
    )
    b.rod(
        elbow,
        point(0, 0, 0),
        point(0.24, 0, 0.075),
        0.037 * scale,
        "shell",
        collision=True,
    )
    roll = b.moving(
        elbow,
        prefix + "wrist_roll",
        point(0.24, 0, 0.075),
        (1, 0, 0),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.45 * scale,
    )
    b.cylinder(
        roll,
        point(0.019, 0, 0),
        0.038 * scale,
        0.041 * scale,
        "blue",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    pitch = b.moving(
        roll,
        prefix + "wrist_pitch",
        point(0.065, 0, 0),
        (0, 1, 0),
        (-1.9, 1.9),
        kind="hinge",
        mass=0.25 * scale,
    )
    b.cylinder(
        pitch,
        point(0, 0, 0),
        0.032 * scale,
        0.035 * scale,
        "shell",
        quat=(0.70710678, 0.70710678, 0, 0),
    )
    wrist = b.moving(
        pitch,
        prefix + "tool_yaw",
        point(0.045, 0, 0),
        (1, 0, 0),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.15 * scale,
    )
    b.cylinder(
        wrist,
        point(0.019, 0, 0),
        0.028 * scale,
        0.018 * scale,
        "metal",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    b.box(wrist, point(0.044, 0, 0), point(0.023, 0.047, 0.022), "dark")
    for suffix, sign in (("left", -1), ("right", 1)):
        finger = b.moving(
            wrist,
            prefix + f"gripper_{suffix}",
            point(0.063, sign * 0.027, 0),
            (0, sign, 0),
            (0, 0.025 * scale),
            mass=0.035 * scale,
            force=20,
        )
        b.box(
            finger,
            point(0.025, 0, 0),
            point(0.03, 0.005, 0.019),
            "metal",
            collision=True,
        )
        b.box(
            finger,
            point(0.047, -sign * 0.006, 0),
            point(0.008, 0.003, 0.018),
            "rubber",
            collision=True,
        )
        b.site(
            finger, prefix + f"finger_{suffix}", point(0.05, -sign * 0.009, 0), 0.002
        )
    b.site(wrist, prefix + "tcp", point(0.112, 0, 0))


def _robot_arm(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    _arm(b, base, scale=float(params.get("scale", 1.0)))
    b.metadata["capabilities"] = [
        "six-axis manipulator",
        "independent parallel gripper fingers",
        "tool-center and contact sites",
    ]


def _wheel(
    b: _Builder,
    base: ET.Element,
    key: str,
    pos: tuple[float, float, float],
    radius: float,
    width: float,
) -> None:
    wheel = b.moving(
        base,
        key,
        pos,
        (0, 1, 0),
        None,
        kind="hinge",
        mass=0.8,
        mode="velocity",
        force=12,
    )
    b.cylinder(
        wheel,
        (0, 0, 0),
        radius,
        width / 2,
        "rubber",
        quat=(0.70710678, 0.70710678, 0, 0),
        collision=True,
    )
    for sign in (-1, 1):
        b.cylinder(
            wheel,
            (0, sign * width * 0.51, 0),
            radius * 0.62,
            0.005,
            "metal",
            quat=(0.70710678, 0.70710678, 0, 0),
        )
        b.cylinder(
            wheel,
            (0, sign * width * 0.56, 0),
            radius * 0.17,
            0.008,
            "dark",
            quat=(0.70710678, 0.70710678, 0, 0),
        )
    for i in range(16):
        a = i * math.tau / 16
        b.box(
            wheel,
            (radius * math.sin(a), 0, radius * math.cos(a)),
            (radius * 0.1, width * 0.47, 0.003),
            "dark",
            quat=(math.cos(a / 2), 0, math.sin(a / 2), 0),
        )


def _planar_chassis(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    """Optional free base; default is anchored to avoid unattended scene drift."""
    if params.get("free_base", False):
        ET.SubElement(base, "freejoint", name=f"{b.name}__free_base")
        ET.SubElement(
            base, "inertial", pos="0 0 0.25", mass="18", diaginertia="0.9 1.2 1.5"
        )
        b.metadata["freejoint"] = f"{b.name}__free_base"
    else:
        b.metadata["limitations"].append(
            "Base is anchored by default; set free_base=True for contact-driven locomotion"
        )


def _rover(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    _planar_chassis(b, base, params)
    b.box(base, (0, 0, 0.3), (0.34, 0.25, 0.095), "cream", collision=True)
    b.box(base, (0, 0, 0.403), (0.38, 0.3, 0.015), "metal")
    for y in (-0.19, 0.19):
        b.box(base, (-0.13, y, 0.425), (0.15, 0.078, 0.006), "blue")
        for i in range(6):
            b.box(
                base, (-0.265 + i * 0.054, y, 0.432), (0.001, 0.074, 0.0005), "bright"
            )
    for side, y in (("left", -0.366), ("right", 0.366)):
        for index, x in enumerate((-0.4, 0, 0.4)):
            b.rod(base, (x * 0.65, y * 0.66, 0.325), (x, y, 0.148), 0.025, "metal")
            _wheel(b, base, f"wheel_{side}_{index + 1}", (x, y, 0.148), 0.14, 0.108)
    b.rod(base, (-0.22, 0, 0.41), (-0.22, 0, 1.0), 0.022, "metal")
    pan = b.moving(
        base,
        "mast_pan",
        (-0.22, 0, 1.026),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.5,
    )
    b.box(pan, (0, 0, 0), (0.044, 0.105, 0.041), "cream")
    for y in (-0.065, 0.065):
        b.cylinder(
            pan,
            (0.052, y, 0),
            0.023,
            0.017,
            "dark",
            quat=(0.70710678, 0, 0.70710678, 0),
        )
        b.cylinder(
            pan,
            (0.071, y, 0),
            0.018,
            0.002,
            "lens",
            quat=(0.70710678, 0, 0.70710678, 0),
        )
    b.site(pan, "stereo_camera", (0.075, 0, 0))
    arm_base = ET.SubElement(
        base, "body", name=f"{b.name}__sampling_arm_base", pos="0.24 0 0.415"
    )
    _arm(b, arm_base, prefix="arm_", scale=0.62)
    b.box(base, (-0.2, 0, 0.449), (0.1, 0.07, 0.025), "dark")
    for i in range(3):
        b.cylinder(base, (-0.26 + i * 0.06, 0, 0.478), 0.022, 0.015, "bright")
        b.site(base, f"sample_cache_{i + 1}", (-0.26 + i * 0.06, 0, 0.497))
    b.metadata["capabilities"] = [
        "six individually driven wheels",
        "camera mast pan",
        "six-axis sample arm",
        "parallel gripper",
        "three sample-cache targets",
    ]


def _mobile_robot(b: _Builder, base: ET.Element, params: dict[str, Any]) -> None:
    _planar_chassis(b, base, params)
    b.box(base, (0, 0, 0.198), (0.31, 0.207, 0.097), "shell", collision=True)
    b.box(base, (0, 0, 0.315), (0.318, 0.214, 0.018), "dark")
    for side, y in (("left", -0.228), ("right", 0.228)):
        _wheel(b, base, f"wheel_{side}", (0, y, 0.118), 0.114, 0.04)
    for x in (-0.245, 0.245):
        b.geom(base, "sphere", (0.04,), (x, 0, 0.045), "rubber", collision=True)
    b.box(base, (0.31, 0, 0.17), (0.008, 0.19, 0.025), "dark")
    b.cylinder(base, (0.22, 0, 0.365), 0.05, 0.028, "dark")
    b.cylinder(base, (0.22, 0, 0.387), 0.052, 0.005, "cyan")
    b.rod(base, (-0.2, 0, 0.33), (-0.2, 0, 0.53), 0.017, "metal")
    b.box(base, (-0.2, 0, 0.555), (0.032, 0.055, 0.025), "dark")
    b.cylinder(
        base,
        (-0.163, 0, 0.555),
        0.019,
        0.009,
        "lens",
        quat=(0.70710678, 0, 0.70710678, 0),
    )
    b.site(base, "payload_deck", (0, 0, 0.337))
    b.site(base, "front_docking", (0.33, 0, 0.19))
    b.metadata["capabilities"] = [
        "differential drive wheels",
        "payload-deck target",
        "front docking target",
    ]


_BUILDERS: dict[str, Callable[[_Builder, ET.Element, dict[str, Any]], None]] = {
    "liquid_handler": _liquid_handler,
    "sequencer": _sequencer,
    "microfluidic_station": _microfluidic_station,
    "isotope_analyzer": _isotope_analyzer,
    "deposition_chamber": _deposition_chamber,
    "phenotyping_booth": _phenotyping_booth,
    "magnetic_microrobot": _magnetic_microrobot,
    "sample_processor": _sample_processor,
    "aerosol_sampler": _aerosol_sampler,
    "rover": _rover,
    "robot_arm": _robot_arm,
    "mobile_robot": _mobile_robot,
}

EQUIPMENT_KINDS = tuple(_BUILDERS)

_SAMPLE_INTERFACES = {
    "liquid_handler": (
        "deck_3",
        (0.1, 0.07, 0.02),
        "supported",
        "empty labware deck slot",
    ),
    "sequencer": ("flowcell_1", (0.09, 0.14, 0.01), "supported", "flow-cell carrier"),
    "microfluidic_station": (
        "specimen",
        (0.006, 0.006, 0.002),
        "supported",
        "chip specimen carrier, not a biological cell",
    ),
    "isotope_analyzer": (
        "sample_seat",
        (0.012, 0.012, 0.05),
        "supported",
        "autosampler vial in empty first carousel position",
    ),
    "deposition_chamber": (
        "substrate",
        (0.04, 0.04, 0.006),
        "supported",
        "substrate coupon",
    ),
    "phenotyping_booth": (
        "plant_pot",
        (0.23, 0.23, 0.17),
        "preinstalled",
        "plant pot; disable include_sample to replace",
    ),
    "magnetic_microrobot": (
        "micro_agent",
        (0.0004, 0.00012, 0.00012),
        "preinstalled",
        "geometric microscopic agent; no magnetic actuation",
    ),
    "sample_processor": (
        "specimen_block",
        (0.036, 0.01, 0.024),
        "clamped",
        "tissue block represented in cassette clamp",
    ),
    "aerosol_sampler": (
        "filter",
        (0.037, 0.037, 0.002),
        "supported",
        "37 mm filter carrier",
    ),
    "rover": (
        "sample_cache_1",
        (0.028, 0.028, 0.05),
        "clamped",
        "sample tube in cache holder",
    ),
    "robot_arm": ("tcp", (0.035, 0.035, 0.035), "grasped", "manipulated payload"),
    "mobile_robot": (
        "payload_deck",
        (0.18, 0.18, 0.1),
        "supported",
        "deck payload; raise centroid by half payload height",
    ),
}


def add_equipment(
    root: ET.Element,
    world: ET.Element,
    kind: str,
    name: str,
    pos: Sequence[float],
    yaw: float = 0.0,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an articulated instrument and return semantic interaction handles.

    ``pos`` places the bottom support surface; ``yaw`` is in radians. Each name
    must be unique within the MJCF. Rover/mobile bases are anchored unless
    ``params={"free_base": True}`` is supplied. Namespaced local materials mean
    this function does not rely on the caller's default material definitions.
    """
    if kind in _BUILDERS:
        build, source, interface = (
            _BUILDERS[kind],
            SOURCES[kind],
            _SAMPLE_INTERFACES[kind],
        )
    else:
        from real_labs.university_extensions import instrument_specs

        extensions = instrument_specs()
        if kind not in extensions:
            raise ValueError(
                f"Unknown equipment kind {kind!r}; choose from {EQUIPMENT_KINDS + tuple(extensions)}"
            )
        build, source, interface = extensions[kind]
    if not callable(build):
        raise ValueError(f"Instrument builder is not callable: {kind}")
    if len(pos) != 3 or not all(math.isfinite(float(v)) for v in (*pos, yaw)):
        raise ValueError(
            "Equipment position and yaw must be finite metre/radian values"
        )
    if not name or any(c.isspace() for c in name):
        raise ValueError(
            "Equipment name must be a nonempty MJCF identifier without spaces"
        )
    if any(body.get("name") == name for body in world.iter("body")):
        raise ValueError(f"Duplicate equipment name: {name}")
    b = _Builder(root, name)
    base = ET.SubElement(
        world,
        "body",
        name=name,
        pos=_numbers(pos),
        quat=_numbers((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))),
    )
    build(b, base, params or {})
    b.metadata.update(
        kind=kind,
        name=name,
        body=name,
        sources=[dict(source)],
        dimensions_m=list(source["dimensions_m"]),
    )
    key, size, attachment, description = interface
    b.metadata.update(
        sample_site=b.metadata["sites"][key],
        sample_size_m=list(size),
        sample_attachment=attachment,
        sample_description=description,
    )
    return b.metadata
