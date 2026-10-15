"""Source-informed EPFL hood, supported rack and true screw-driven thermal stage.

The historical photograph supports the twin hood silhouettes, service fascia,
clamp lattice and asymmetric supply density. Dimensions, offsets, coupler
internals and guide travel are authored. No installed manufacturer is identified.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount
from real_labs import epfl_apparatus_geometry as apparatus
from real_labs.epfl_jack_geometry import add_scissor_jack


BENCH_TOP = .934
RACK_Y = .30
POST_XS = (-.60, -.28, .16, .59)
BAR_ZS = (1.24, 1.72, 2.13)
BAR_Y = .330


def name(b, key):
    return f"{b.name}__hood_context_{key}"


def box(b, parent, key, pos, half, material="shell", **kwargs):
    return b.box(parent, pos, half, material, collision=True,
                 name=name(b, key), **kwargs)


def cylinder(b, parent, key, pos, radius, half, material="bright", **kwargs):
    return b.cylinder(parent, pos, radius, half, material, collision=True,
                      name=name(b, key), **kwargs)


def rod(b, parent, key, first, last, radius=.008, material="bright"):
    return b.rod(parent, first, last, radius, material, collision=True,
                 name=name(b, key))


def add_hood_shell(b, base):
    """Original hood envelope and 0–480 mm sash travel, with supported details.

    Main bounds remain X +/- .87, back Y=.47, roof Z=2.595. Moving handle
    front Y=-.559 and original sash body position (0,-.50,1.64) remain intact.
    Panel junctions are trimmed to their supporting faces instead of overlapping.
    """
    box(b, base, "plinth", (0, 0, .07), (.85, .43, .07), "dark")
    box(b, base, "cabinet", (0, .02, .455), (.84, .40, .315))
    box(b, base, "fascia", (0, -.01, .828), (.87, .46, .058))
    box(b, base, "bench", (0, 0, .91), (.86, .44, .024), "dark")
    for x in (-.42, .42):
        box(b, base, f"cabinet_door_{x}", (x, -.397, .45),
            (.40, .017, .29), "cream")
        rod(b, base, f"cabinet_handle_{x}", (x-.10, -.448, .63),
            (x+.10, -.448, .63), .009)
        for dx in (-.075, .075):
            cylinder(b, base, f"cabinet_handle_mount_{x}_{dx}",
                     (x+dx, -.4265, .63), .006, .0125,
                     euler=(math.pi/2, 0, 0))
    lower, upper = BENCH_TOP, 2.465
    box(b, base, "back", (0, .435, (lower+upper)/2),
        (.87, .035, (upper-lower)/2), "cream")
    for x in (-.84, .84):
        box(b, base, f"side_{x}", (x, -.005, (lower+upper)/2),
            (.027, .405, (upper-lower)/2))
        box(b, base, f"guide_post_{x}", (x, -.46, 1.78),
            (.025, .025, .81), "metal")
        # Guide post feet bear on the fascia and meet the bench front edge.
        box(b, base, f"guide_foot_{x}", (x, -.455, .928),
            (.025, .015, .042), "metal")
        for z in (1.03, 2.43):
            box(b, base, f"guide_bridge_{x}_{z}", (x, -.4225, z),
                (.020, .0125, .035), "metal")
        # Captured C-channel with 0.5 mm lateral running clearance. The original
        # prismatic joint represents its guided bearing; no rubbing contact is
        # forced at the exact aligned pose. All channel collisions remain on.
        sign = 1 if x > 0 else -1
        box(b, base, f"guide_liner_{x}", (sign*.81375, -.4915, 1.8775),
            (.00125, .0085, .6075), "dark")
        box(b, base, f"guide_channel_web_{x}", (sign*.81625, -.5085, 1.8775),
            (.00125, .0235, .6075), "metal")
        for edge, y in (("front", -.52775), ("rear", -.47225)):
            box(b, base, f"guide_channel_{edge}_{x}", (sign*.801, y, 1.8775),
                (.014, .00425, .6075), "metal")
    box(b, base, "roof", (0, .015, 2.53), (.87, .43, .065))
    for row in range(5):
        box(b, base, f"inert_vent_mark_{row}", (0, -.417, 2.482+row*.012),
            (.65, .002, .002), "dark")
    sash = b.moving(base, "sash_height", (0, -.50, 1.64), (0, 0, 1),
                    (0, .48), mass=1.4)
    box(b, sash, "sash_glass", (0, 0, 0), (.788, .006, .345), "glass")
    for z in (-.36, .36):
        box(b, sash, f"sash_crossrail_{z}", (0, 0, z),
            (.81, .023, .015), "bright")
    for x in (-.80, .80):
        box(b, sash, f"sash_siderail_{x}", (x, 0, 0),
            (.012, .018, .345), "bright")
    rod(b, sash, "sash_handle", (-.23, -.045, -.36),
        (.23, -.045, -.36), .014, "metal")
    for x in (-.20, .20):
        cylinder(b, sash, f"sash_handle_mount_{x}", (x, -.027, -.36),
                 .008, .004, "metal", euler=(math.pi/2, 0, 0))
    # Service disks touch the fascia; inert levers touch their disk fronts.
    for index in range(7):
        x = -.66 + index*.22
        cylinder(b, base, f"service_disk_{index}", (x, -.480, .840),
                 .024, .010, euler=(math.pi/2, 0, 0))
        box(b, base, f"service_lever_{index}", (x, -.503, .840),
            (.009, .013, .025), "cyan" if index % 2 else "blue")
        if index in (1, 4):
            box(b, base, f"inert_socket_{index}", (x, -.476, .788),
                (.025, .006, .014), "dark")
    b.site(sash, "hood_sash_handle", (0, -.045, -.36))
    b.metadata["capabilities"].append("Vertical sash access with continuous supported side guides")
    b.metadata["limitations"].append(
        "Historical 2019 hood silhouette; dimensions, guide details and 480 mm travel authored. "
        "Services and vents are inert geometry; no extraction airflow or chemistry.")
    return {"sash_body": sash, "bench_top_m": BENCH_TOP,
            "sash_joint": b.metadata["joints"]["sash_height"],
            "sash_range_m": (0, .48), "rack_y_m": RACK_Y,
            "minimum_guide_running_clearance_m": .0005}


def add_supported_rack(b, base):
    """Ground all four posts and separate orthogonal rods by 30 mm in Y.

    Two bored collars and a small fused connecting block form each coupler.
    The rods have 14 mm surface separation at every crossing; no solid boss
    or tube is buried in the other tube. All supports bear on the hood bench.
    """
    grounded = apparatus.add_grounded_rack(
        b, base, bench_z=BENCH_TOP, rack_y=RACK_Y, post_xs=POST_XS)
    for z in BAR_ZS:
        rod(b, base, f"rack_crossbar_{z}", (-.65, BAR_Y, z),
            (.65, BAR_Y, z), apparatus.RACK_RADIUS)
        for x in POST_XS:
            key = f"coupler_{x}_{z}"
            for axis, center in (("z", (x, RACK_Y, z)), ("x", (x, BAR_Y, z))):
                annular_mount(b, base, key+"_"+axis, center, .008, .013, .011,
                              axis=axis, proxies=True, material="metal",
                              namespace="hood_context")
            box(b, base, key+"_bridge", (x, (RACK_Y+BAR_Y)/2, z),
                (.006, .003, .006), "metal")
    return {"grounded_posts": grounded, "post_xs": POST_XS,
            "post_y": RACK_Y, "crossbar_y": BAR_Y, "crossbar_zs": BAR_ZS,
            "orthogonal_rod_surface_gap_m": BAR_Y-RACK_Y-2*apparatus.RACK_RADIUS}


def _restrained_loop(b, base, key, points):
    """Inert source-category utility tubing, with both ends visibly anchored."""
    for endpoint, p in enumerate((points[0], points[-1])):
        box(b, base, f"{key}_anchor_{endpoint}", (p[0], .389, p[2]),
            (.016, .011, .016), "metal")
    for index, (first, last) in enumerate(zip(points, points[1:])):
        rod(b, base, f"{key}_segment_{index}", first, last, .007, "rubber")


def add_source_context(b, base, *, side="left", props=True):
    """Asymmetric inert props/utility density; requires add_supported_rack first.

    The secondary hotplate and retained jack add no actuated lift mechanism.
    Keep props=False when a caller has already added these supported props.
    """
    if side not in ("left", "right"):
        raise ValueError("Expected side left or right")
    if props:
        # The frozen right tray was authored for a minimal one-post fixture.
        # Move the complete supported prop group 60 mm forward to clear the
        # newly present .59 post's grounded socket; keep its bottle on the tray.
        props_parent = base
        if side == "right":
            props_parent = ET.SubElement(base, "body", name=name(b, "right_props"),
                                         pos="0 -.06 0")
        apparatus.add_asymmetric_props(b, props_parent, side=side, bench_z=BENCH_TOP,
                                      rack_y=RACK_Y, back_y=.46 if side == "right" else .40)
    _restrained_loop(b, base, "left_utility_loop", (
        (-.72, .371, 1.48), (-.66, .22, 1.38), (-.72, .18, 1.10),
        (-.77, .24, 1.02), (-.72, .371, 1.10)))
    if side == "left":
        _restrained_loop(b, base, "right_utility_loop", (
            (.70, .371, 1.55), (.74, .22, 1.35), (.69, .17, 1.07),
            (.70, .371, 1.18)))
    return {"side": side, "props_added": props,
            "props_y_offset_m": -.06 if side == "right" else 0,
            "restrained_utility_loops": 2 if side == "left" else 1}


def _explicit_local_defaults(base, existing):
    """Keep the checked local mechanics independent of scene-wide defaults."""
    for element in base.iter():
        if id(element) in existing:
            continue
        if element.tag == "geom":
            for key, value in dict(solref=".02 1", solimp=".9 .95 .001 .5 2",
                                   solmix="1", margin="0", gap="0", priority="0",
                                   condim="3", friction="1 .005 .0001",
                                   density="1000").items():
                element.attrib.setdefault(key, value)
        elif element.tag == "joint":
            for key, value in dict(frictionloss="0", stiffness="0", margin="0",
                                   ref="0", springref="0", solreflimit=".02 1",
                                   solimplimit=".9 .95 .001 .5 2",
                                   solreffriction=".02 1",
                                   solimpfriction=".9 .95 .001 .5 2").items():
                element.attrib.setdefault(key, value)


def add_synthesis_hood(b, base, params):
    """Compose one independently namespaced hood and its mechanical contract.

    ``bath_height`` is the bath rim above the worktop datum. It is neither a
    slider joint nor flask elevation: the flask remains fixed on the rack.
    The mechanism controller drives screw torque through its original eight
    coordinates and four equalities. Ordinary builder controls expose only
    ``sash_height``; closed-chain joints are registered under ``mechanisms``.
    """
    side = params.get("side", "left")
    if side not in {"left", "right"}:
        raise ValueError("EPFL hood side must be left or right")
    existing = {id(element) for element in base.iter()}
    shell = add_hood_shell(b, base)
    add_supported_rack(b, base)
    add_source_context(b, base, side=side)
    axis = (-.28, -.005)
    glassware = apparatus.add_fixed_glassware(b, base, axis=axis, bench_z=BENCH_TOP)
    jack = add_scissor_jack(b, base, origin=(*axis, BENCH_TOP))
    payload = apparatus.add_hotplate_open_bath(b, jack["deck"])
    presentation = apparatus.refine_assembly_presentation(b, jack["mount"], payload)
    datum_site = b.site(base, "bath_worktop_datum", (*axis, BENCH_TOP))
    observation_site = b.metadata["sites"]["main_bath_rim"]
    b.metadata.setdefault("mechanisms", {})["bath_height"] = {
        "kind": "scissor_jack_v1", "joints": jack["joints"],
        "actuator": jack["actuator"], "equalities": jack["equalities"],
        "deck_site": jack["deck_site"], "datum_site": datum_site,
        "observation_site": observation_site, "range": [.245, .375],
        "deck_to_observation_offset_m": .140,
        "observation": "Bath rim Z relative to the fixed worktop datum",
    }
    b.metadata["interface_revision"] = 2
    b.metadata["retired_controls"] = {
        "flask_lift": "flask_lift was retired in interface revision 2: the flask is fixed; "
                      "use bath_height for the bath rim height above the worktop."
    }
    b.metadata["capabilities"] = [
        "Vertical sash access through supported guides",
        "Screw-driven bath-rim positioning below independently clamped glassware",
    ]
    b.metadata["epfl_geometry"] = {
        "side": side, "bench_top_m": BENCH_TOP, "fixture_axis_m": list(axis),
        "minimum_guide_running_clearance_m": shell["minimum_guide_running_clearance_m"],
        "authored_masses_kg": {"jack_deck": .5, "hotplate_and_bath": 1.8},
        "fixed_glassware": glassware, "presentation": presentation,
    }
    b.metadata["limitations"].extend([
        "Original dimensions and inertias are authored from historical exterior cues and a laboratory-jack size class; no installed brand, OEM CAD or rated-load reproduction.",
        "The 0.5 kg deck, 1.8 kg thermal payload, ideal 1.25 mm screw lead and motor law are authored working assumptions.",
        "Glassware is statically attached to the rack. Clamp friction, glass fracture, liquid contents and chemistry are not simulated.",
        "The flask wall uses conservative convex cells from the hollow visual mesh. A rounded-probe interior check is not general insertion, filling or full-tool qualification.",
        "Sash uses the existing gravity-compensated prismatic bearing with 0.5 mm running clearance; no OEM counterweight or extraction model.",
        "Selected apparatus only: secondary hotplate/jack stacks visible in both source hoods are not yet reconstructed.",
    ])
    _explicit_local_defaults(base, existing)
