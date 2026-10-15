"""Namespaced two-stage laboratory jack with an ideal screw transmission.

The rotating screw, horizontal nut and three loop closures move the actual
links through eight coordinates. There is no vertical slider beneath the deck.
Dimensions are authored within a documented laboratory-jack size class; this
is not Bochem CAD or an identification of the jack installed at EPFL.
"""

import math
import xml.etree.ElementTree as ET


LENGTH = 0.14
MIN_HEIGHT = 0.105
MAX_HEIGHT = 0.235
PIVOT_Z = 0.018
DECK_OFFSET = 0.018
ANGLE = math.asin((MIN_HEIGHT - PIVOT_Z - DECK_OFFSET) / (2 * LENGTH))
WIDTH = LENGTH * math.cos(ANGLE)
MIN_WIDTH = math.sqrt(LENGTH**2 - ((MAX_HEIGHT - PIVOT_Z - DECK_OFFSET) / 2) ** 2)
SCREW_LEAD = .00125
METRES_PER_RADIAN = SCREW_LEAD / math.tau
SCREW_Z = .0278


def numbers(values):
    return " ".join(f"{x:.15g}" for x in values)


def height_to_nut(height):
    if not MIN_HEIGHT <= height <= MAX_HEIGHT:
        raise ValueError("Height is outside the authored operating range")
    return math.sqrt(LENGTH**2 - ((height - PIVOT_Z - DECK_OFFSET) / 2) ** 2) - WIDTH


class Geometry:
    def __init__(self, root):
        self.asset = ET.SubElement(root, "asset")
        self.count = 0

    def geom(self, body, name, kind, size, pos=(0, 0, 0), **extra):
        self.count += 1
        return ET.SubElement(body, "geom", name=name, type=kind,
                             size=numbers(size), pos=numbers(pos),
                             rgba=".55 .59 .60 1", contype="1", conaffinity="1",
                             condim="3", density="1000", friction="1 .005 .0001",
                             solmix="1", solref=".02 1", solimp=".9 .95 .001 .5 2",
                             margin="0", gap="0", priority="0",
                             **{k: numbers(v) if isinstance(v, tuple) else str(v)
                                for k, v in extra.items()})

    def ring(self, body, name, centre, inner=.0035, outer=.0085, half=.0015, axis="y"):
        """Convex sectors retain a physical shaft bore through every eye."""
        for i in range(12):
            a, b = i * math.tau / 12, (i + 1) * math.tau / 12
            vertices = []
            for y in (-half, half):
                for radius, angle in ((inner, a), (outer, a), (outer, b), (inner, b)):
                    offset = (radius * math.cos(angle), y, radius * math.sin(angle))
                    if axis == "x":
                        offset = (offset[1], offset[0], offset[2])
                    elif axis != "y":
                        raise ValueError("Only X- and Y-axis bearing bores are supported")
                    vertices.append(tuple(a+b for a,b in zip(centre, offset)))
            faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7)]
            for k in range(4):
                n = (k + 1) % 4
                faces.extend(((k, n, n + 4), (k, n + 4, k + 4)))
            if axis == "y":
                faces = [tuple(reversed(face)) for face in faces]
            mesh = f"{name}_{i}"
            ET.SubElement(self.asset, "mesh", name=mesh,
                          vertex=numbers([x for v in vertices for x in v]),
                          face=" ".join(str(x) for f in faces for x in f))
            self.geom(body, mesh, "mesh", (), mesh=mesh)

    def axle(self, body, name, x=0):
        self.geom(body, name, "cylinder", (.003, .084), (x, 0, 0),
                  euler=(math.pi / 2, 0, 0))

    def link(self, body, name, sign, y):
        for side in (-1, 1):
            for index, x in enumerate((0, LENGTH / 2, LENGTH)):
                self.ring(body, f"{name}_eye_{side}_{index}", (sign * x, side * y, 0))
            for index in (0, 1):
                self.geom(body, f"{name}_web_{side}_{index}", "box",
                          (LENGTH / 4 - .007, .0015, .006),
                          (sign * (index + .5) * LENGTH / 2, side * y, 0))


def body(parent, name, pos, *, angle=0, mass=.12, centre=(0, 0, 0), inertia=None):
    element = ET.SubElement(parent, "body", name=name, pos=numbers(pos),
                            euler=numbers((0, angle, 0)), gravcomp="0")
    inertia = inertia or (.00046, .00020, .00065)
    ET.SubElement(element, "inertial", mass=str(mass), pos=numbers(centre),
                  diaginertia=numbers(inertia))
    return element


def joint(element, name, *, kind="hinge", axis=(0, 1, 0), limits=None):
    attrs = dict(name=name, type=kind, axis=numbers(axis), damping=".004",
                 armature=".00001", limited="false", frictionloss="0", stiffness="0",
                 margin="0", ref="0", springref="0", solreflimit=".02 1",
                 solimplimit=".9 .95 .001 .5 2", solreffriction=".02 1",
                 solimpfriction=".9 .95 .001 .5 2")
    if limits is not None:
        attrs.update(limited="true", range=numbers(limits))
    ET.SubElement(element, "joint", attrs)


def _make_jack_tree():
    root = ET.Element("mujoco", model="epfl_scissor_jack_components")
    geometry = Geometry(root)
    world = ET.SubElement(root, "worldbody")
    geometry.geom(world, "base_plate", "box", (.1, .1, .002), (0, 0, .002))
    for side in (-1, 1):
        geometry.geom(world, f"base_slot_wall_{side}", "box", (.041, .002, .003),
                      (.04, side * .095, .007))
        geometry.geom(world, f"base_slot_lip_{side}", "box", (.041, .004, .001),
                      (.04, side * .091, .009))
    for x in (-.094, .090):
        geometry.ring(world, f"screw_bearing_{x}", (x, 0, SCREW_Z),
                      inner=.0046, outer=.0095, half=.004, axis="x")
        geometry.geom(world, f"screw_pedestal_{x}", "box", (.004, .007, .0075),
                      (x, 0, .0115))
    base = ET.SubElement(world, "body", name="base_pivot", pos=numbers((-.075, 0, PIVOT_Z)))
    geometry.axle(base, "base_left_axle")
    for sign in (-1, 1):
        geometry.ring(base, f"base_eye_{sign}", (0, sign * .079, 0), half=.003)
        geometry.geom(base, f"base_mount_{sign}", "box", (.007, .005, .003),
                      (0, sign * .079, -.011))
    a1 = body(base, "a1", (0, 0, 0), angle=-ANGLE, centre=(LENGTH / 2, 0, 0))
    joint(a1, "a1_angle", axis=(0, -1, 0))
    geometry.link(a1, "a1", 1, .060)
    geometry.axle(a1, "lower_middle_axle", LENGTH / 2)
    geometry.axle(a1, "lower_right_axle", LENGTH)

    nut = body(base, "driven_nut", (WIDTH, 0, 0), mass=.04,
               centre=(0, 0, -.010), inertia=(.00008, .00001, .00008))
    joint(nut, "nut_translation", kind="slide", axis=(1, 0, 0), limits=(MIN_WIDTH - WIDTH, 0))
    geometry.geom(nut, "nut_carriage", "box", (.009, .092, .0015), (0, 0, -.0125))
    for side in (-1, 1):
        geometry.ring(nut, f"nut_eye_{side}", (0, side * .079, 0), half=.003)
        geometry.geom(nut, f"nut_eye_stem_{side}", "box", (.007, .003, .0025),
                      (0, side * .079, -.0085))
    geometry.axle(nut, "base_right_axle")
    geometry.ring(nut, "threaded_nut", (0, 0, SCREW_Z - PIVOT_Z),
                  inner=.0046, outer=.0063, half=.007, axis="x")
    for side in (-1, 1):
        geometry.geom(nut, f"threaded_nut_stem_{side}", "box", (.002, .006, .008),
                      (side * .008, 0, -.0035))
    b1 = body(nut, "b1", (0, 0, 0), angle=ANGLE, centre=(-LENGTH / 2, 0, 0))
    joint(b1, "b1_angle")
    geometry.link(b1, "b1", -1, .066)
    geometry.axle(b1, "lower_left_axle", -LENGTH)

    a2 = body(b1, "a2", (-LENGTH, 0, 0), angle=-2 * ANGLE, centre=(LENGTH / 2, 0, 0))
    joint(a2, "a2_angle", axis=(0, -1, 0))
    geometry.link(a2, "a2", 1, .060)
    geometry.axle(a2, "upper_middle_axle", LENGTH / 2)
    geometry.axle(a2, "upper_right_axle", LENGTH)
    b2 = body(a1, "b2", (LENGTH, 0, 0), angle=2 * ANGLE, centre=(-LENGTH / 2, 0, 0))
    joint(b2, "b2_angle")
    geometry.link(b2, "b2", -1, .066)
    geometry.axle(b2, "upper_left_axle", -LENGTH)

    deck = body(b2, "deck", (-LENGTH, 0, 0), angle=-ANGLE, mass=1.8,
                centre=(.075, 0, .053), inertia=(.003, .005, .005))
    joint(deck, "deck_pitch", axis=(0, -1, 0))
    geometry.geom(deck, "deck_plate", "box", (.1, .1, .002), (.075, 0, .016))
    for side in (-1, 1):
        geometry.ring(deck, f"deck_eye_{side}", (0, side * .079, 0), half=.003)
        geometry.geom(deck, f"deck_eye_stem_{side}", "box", (.007, .003, .004),
                      (0, side * .079, .010))
        geometry.geom(deck, f"deck_slot_wall_{side}", "box", (.043, .002, .004),
                      (.12, side * .095, .010))
        geometry.geom(deck, f"deck_slot_lip_{side}", "box", (.043, .004, .001),
                      (.12, side * .091, .007))
    geometry.geom(deck, "inert_hotplate_payload", "box", (.085, .09, .035), (.075, 0, .053))
    ET.SubElement(deck, "site", name="deck_top", pos=numbers((.075, 0, DECK_OFFSET)), size=".002")
    top_slider = body(deck, "top_slider", (WIDTH, 0, 0), mass=.04,
                      centre=(0, 0, .006), inertia=(.00008, .00001, .00008))
    joint(top_slider, "top_slot_translation", kind="slide", axis=(1, 0, 0))
    geometry.geom(top_slider, "top_slider_carriage", "box", (.009, .092, .0015), (0, 0, .0115))
    for side in (-1, 1):
        geometry.ring(top_slider, f"slider_eye_{side}", (0, side * .079, 0), half=.003)
        geometry.geom(top_slider, f"slider_eye_stem_{side}", "box", (.007, .003, .0025),
                      (0, side * .079, .0085))

    equality = ET.SubElement(root, "equality")
    for name, first, second, anchor in (
        ("lower_scissor", "a1", "b1", (LENGTH / 2, 0, 0)),
        ("upper_scissor", "a2", "b2", (LENGTH / 2, 0, 0)),
        ("deck_slot", "a2", "top_slider", (LENGTH, 0, 0)),
    ):
        ET.SubElement(equality, "connect", name=name, body1=first, body2=second,
                      anchor=numbers(anchor), solref=".004 1", solimp=".9999 .9999 .001")
    screw = body(world, "screw", (0, 0, SCREW_Z), mass=.12,
                 centre=(-.040, 0, 0), inertia=(.00001, .00045, .00045))
    joint(screw, "screw_rotation", axis=(1, 0, 0))
    screw.find("joint").set("damping", ".000002")
    screw.find("joint").set("armature", "0")
    geometry.geom(screw, "screw_shaft", "cylinder", (.004, .1065), (-.0085, 0, 0),
                  euler=(0, math.pi/2, 0))
    geometry.geom(screw, "screw_knob", "cylinder", (.023, .0045), (-.12, 0, 0),
                  euler=(0, math.pi/2, 0))
    geometry.geom(screw, "screw_knob_spoke", "box", (.0005, .007, .0012),
                  (-.125, .011, 0))
    ET.SubElement(equality, "joint", name="screw_transmission",
                  joint1="screw_rotation", joint2="nut_translation",
                  # MuJoCo's soft-equality regularization is not invariant to
                  # this equation's scale. The reverse metre-valued equation
                  # was excessively compliant with the screw's small inertia.
                  # This residual is in radians; checks convert it to metres.
                  polycoef=numbers((0, 1 / METRES_PER_RADIAN, 0, 0, 0)),
                  solref=".004 1", solimp=".9999 .9999 .001")
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "motor", name="screw_drive", joint="screw_rotation",
                  ctrllimited="true", ctrlrange="-.2 .2", forcelimited="false", gear="1")
    return root


JOINT_KEYS = (
    "a1_angle", "nut_translation", "b1_angle", "a2_angle", "b2_angle",
    "deck_pitch", "top_slot_translation", "screw_rotation",
)
EQUALITY_KEYS = ("lower_scissor", "upper_scissor", "deck_slot", "screw_transmission")


def add_scissor_jack(b, parent, *, origin=(-.28, -.005, .934)):
    """Append the tested eight-coordinate mechanism without global options.

    The returned deck has an authored 0.5 kg inertial. Its caller adds the
    separate 1.8 kg hotplate/bath payload at deck-local (.075, 0, .018).
    All model names and references are qualified by ``b.name``. The mechanism
    stays outside the builder's ordinary joint/actuator maps; its driven
    observable is registered by the hood through an explicit mechanism contract.
    """
    source = _make_jack_tree()
    deck = source.find(".//body[@name='deck']")
    deck.remove(deck.find("geom[@name='inert_hotplate_payload']"))
    inertial = deck.find("inertial")
    inertial.set("mass", ".5")
    inertial.set("pos", ".075 0 .014")
    inertial.set("diaginertia", ".002 .002 .0033")
    prefix = f"{b.name}__epfl_jack_"
    names = {element.get("name"): prefix + element.get("name")
             for element in source.iter() if element.get("name")}
    references = {"mesh", "material", "body1", "body2", "joint", "joint1", "joint2"}
    for element in source.iter():
        if element.get("name"):
            element.set("name", names[element.get("name")])
        for attr in references:
            if element.get(attr) in names:
                element.set(attr, names[element.get(attr)])
    asset = b.root.find("asset")
    if asset is None:
        asset = ET.SubElement(b.root, "asset")
    green = f"{b.name}__mat_jack_green"
    if asset.find(f"material[@name='{green}']") is None:
        ET.SubElement(asset, "material", name=green, rgba=".025 .24 .105 1",
                      specular=".25", shininess=".22")
    for element in source.find("asset"):
        asset.append(element)
    mount = ET.SubElement(parent, "body", name=prefix + "mount", pos=numbers(origin),
                          gravcomp="0")
    for element in source.find("worldbody"):
        mount.append(element)
    for geom in mount.iter("geom"):
        semantic = geom.get("name").removeprefix(prefix)
        material = "rubber" if "knob" in semantic else "metal"
        if semantic in {"base_plate", "deck_plate"}:
            material = "jack_green"
        geom.attrib.pop("rgba", None)
        geom.set("material", f"{b.name}__mat_{material}")
    for section in ("equality", "actuator"):
        target = b.root.find(section)
        if target is None:
            target = ET.SubElement(b.root, section)
        for element in source.find(section):
            target.append(element)
    return {
        "mount": mount, "deck": deck,
        "joints": {key: names[key] for key in JOINT_KEYS},
        "equalities": {key: names[key] for key in EQUALITY_KEYS},
        "actuator": names["screw_drive"], "deck_site": names["deck_top"],
        "deck_range_m": (MIN_HEIGHT, MAX_HEIGHT),
    }
