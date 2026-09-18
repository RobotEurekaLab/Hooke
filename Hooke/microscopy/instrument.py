"""Original detailed housing, visually referenced to Nikon's Ti2-E photos.

This is an independently authored reference model, not official Nikon CAD or
a dimensional reproduction. Mechanical drive and sample definitions stay in
the workstation scene. Product photos remain private research references.
"""

import numpy as np

from microscopy.geometry import body, cylinder, geom, housing, knurled_knob, rounded_box

BENCH_TOP = .705
REFERENCE_URL = "https://www.microscope.healthcare.nikon.com/products/inverted-microscopes/eclipse-ti2-series"


def build_instrument(asset, world, origin):
    scope = body(world, "microscope", (0, 0, BENCH_TOP))
    # Broad tapered stand, rather than an optical post on a flat plate.
    housing(asset, scope, "scope_lower_shell", [
        (.012, .118, .156, 0, .022), (.026, .117, .154, 0, .023),
        (.14, .105, .137, .008, .020), (.205, .086, .115, .018, .018),
    ])
    rounded_box(asset, scope, "scope_bottom_trim", (.118, .153, .009),
                (0, 0, .014), .006, "rubber")
    for x in (-.090, .090):
        for y in (-.125, .115):
            geom(scope, f"scope_foot_{x}_{y}", "cylinder", (.012, .004), (x, y, .004), "rubber")
    rounded_box(asset, scope, "upper_optical_housing", (.092, .102, .025),
                (0, .022, .222), .010, "graphite")
    rounded_box(asset, scope, "upper_lens_column", (.040, .036, .047),
                (0, -.125, .238), .009, "graphite")
    housing(asset, scope, "observation_column", [
        (.150, .038, .034, -.139, .010), (.205, .034, .029, -.155, .010),
        (.265, .035, .030, -.169, .009),
    ], "graphite")
    rounded_box(asset, scope, "binocular_prism", (.047, .041, .029),
                (0, -.188, .286), .011, "graphite", euler=".65 0 0")
    for x in (-.032, .032):
        start = np.array([x, -.203, .303])
        axis = np.array([0, -.71, .704])
        for suffix, distance, length, radius, material in (
            ("barrel", .021, .052, .0142, "graphite"),
            ("diopter", .041, .015, .0160, "rubber"),
            ("rim", .053, .006, .0167, "rubber"),
            ("lens", .057, .001, .0122, "lens"),
        ):
            centre = start + axis*distance
            if suffix == "diopter":
                knurled_knob(scope, f"eyepiece_{x}_{suffix}", centre, axis, radius, length, ribs=24)
            else:
                cylinder(scope, f"eyepiece_{x}_{suffix}", centre-axis*length/2,
                         centre+axis*length/2, radius, material)
    # Long rear illuminator track and large sloping, ventilated lamp head.
    rounded_box(asset, scope, "illumination_spine", (.039, .028, .214),
                (0, .117, .364), .007, "graphite", euler="-.06 0 0")
    for x in (-.036, .036):
        cylinder(scope, f"illumination_track_{x}", (x, .151, .195), (x, .174, .580), .004)
    housing(asset, scope, "illumination_head", [
        (.554, .072, .105, .035, .012), (.582, .073, .105, .035, .013),
        (.642, .070, .085, .055, .014),
    ], "graphite")
    for i, x in enumerate(np.linspace(-.060, .060, 32)):
        geom(scope, f"lamp_vent_{i}", "box", (.00065, .0004, .025), (x, .141, .610), "rubber")
    cylinder(scope, "field_iris", (0, origin[1], .549), (0, origin[1], .532), .030, "graphite")
    cylinder(scope, "field_iris_trim", (0, origin[1], .531), (0, origin[1], .526), .028)
    rounded_box(asset, scope, "condenser_slide", (.037, .038, .023),
                (0, .065, .433), .005, "graphite")
    rounded_box(asset, scope, "condenser_arm", (.035, .075, .008),
                (0, -.006, .404), .006, "graphite")
    cylinder(scope, "condenser_turret", (0, origin[1], .391), (0, origin[1], .413), .041, "graphite")
    cylinder(scope, "condenser_turret_trim", (0, origin[1], .388), (0, origin[1], .394), .039)
    cylinder(scope, "condenser_optics", (0, origin[1], .344), (0, origin[1], .388), .023, "graphite")
    for i in range(4):
        z = .35+i*.010
        cylinder(scope, f"condenser_ring_{i}", (0, origin[1], z), (0, origin[1], z+.002), .0245, "bronze")
    cylinder(scope, "condenser_lens", (0, origin[1], .343), (0, origin[1], .344), .019, "lens")
    knurled_knob(scope, "condenser_focus", (.052, .050, .435), (1, 0, 0), .017, .023, ribs=24)
    # Orange UV screen is outside the tool approach plane.
    rounded_box(asset, scope, "filter_screen_mount", (.037, .005, .004),
                (-.088, -.137, .298), .002, "graphite")
    geom(scope, "orange_filter_screen", "box", (.035, .0015, .048),
         (-.088, -.137, .350), "filter_orange")
    for side in (-1, 1):
        knurled_knob(scope, f"coarse_focus_{side}", (side*.118, -.018, .146),
                     (1, 0, 0), .030, .021, ribs=40)
        cylinder(scope, f"focus_silver_ring_{side}", (side*.130, -.018, .146),
                 (side*.134, -.018, .146), .032)
        knurled_knob(scope, f"fine_focus_{side}", (side*.142, -.018, .146),
                     (1, 0, 0), .016, .015, ribs=32)
        for y, z in ((-.078, .164), (.023, .132), (.065, .188)):
            geom(scope, f"side_button_{side}_{y}", "box", (.001, .006, .003),
                 (side*.107, y, z), "graphite")
    cylinder(scope, "camera_port", (.088, .040, .223), (.131, .040, .223), .022, "graphite")
    cylinder(scope, "camera_adapter", (.128, .040, .223), (.143, .040, .223), .024)
    rounded_box(asset, scope, "camera_sensor", (.037, .039, .037),
                (.174, .040, .223), .004, "graphite")
    for y in (-.030, .025, .080):
        rounded_box(asset, scope, f"camera_heat_rib_{y}", (.002, .025, .027),
                    (.212, y*.35+.040, .223), .001, "rubber")
    # Front selector knobs and status LEDs form the characteristic instrument fascia.
    for x in (-.043, .043):
        cylinder(scope, f"selector_outer_{x}", (x, -.150, .055), (x, -.161, .055), .016, "ivory")
        cylinder(scope, f"selector_ring_{x}", (x, -.160, .055), (x, -.162, .055), .013)
        geom(scope, f"selector_handle_{x}", "capsule", (.004,), material="ivory",
             fromto=f"{x-.010} -.164 .055 {x+.010} -.164 .055")
    for i, (x, z) in enumerate(((-.057, .130), (0, .144), (.057, .130),
                                (-.040, .106), (.040, .106), (0, .084))):
        geom(scope, f"status_led_{i}", "box", (.0025, .0006, .0013), (x, -.137, z), "led_green")
    geom(scope, "model_nameplate", "box", (.024, .0008, .009), (0, -.148, .189), "metal")
    return scope


def detail_objectives(focus, origin, *, placements=None):
    # Silver engraved barrels, coloured magnification collars and a lens at the top.
    placements = ((.024, 0), (-.012, .021), (-.012, -.021), (0, 0)) if placements is None else placements
    for i, (x, y) in enumerate(placements):
        top = -.010 if i == 3 else -.020
        for suffix, lower, upper, radius, material in (
            ("thread", -.051, -.047, .0105, "bronze"),
            ("barrel", -.047, top-.004, .0095 if i < 3 else .011, "metal"),
            ("collar", top-.009, top-.006, .010 if i < 3 else .0115, "blue" if i == 3 else "amber"),
            ("nose", top-.004, top, .0068 if i < 3 else .009, "graphite"),
            ("lens", top, top+.0005, .0045 if i < 3 else .0065, "lens"),
        ):
            cylinder(focus, f"objective_{i}_{suffix}", origin+[x,y,lower], origin+[x,y,upper], radius, material)
