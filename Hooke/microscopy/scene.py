"""Original, SI-unit instrument assets; no proprietary microscope CAD required."""

import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import body, camera, cylinder, frustum, geom, numbers, slide
from microscopy.instrument import BENCH_TOP, build_instrument, detail_objectives

ORIGIN = np.array([0.0, -0.07, 0.995])
RADIUS = 0.0006
BEAD_DAMPING = 1e-10
HOMES = {
    "probe": np.array([-0.006, -0.002, 0.004]),
    "gripper": np.array([0.006, 0.0, 0.004]),
    "injector": np.array([0.003, 0.006, 0.004]),
    "holder": np.array([-0.006, -0.002, 0.004]),
}
DEFAULT_TOOLS = ("probe", "gripper", "injector")
APPROACHES = {"probe": np.array([-1., 0, 0]), "gripper": np.array([1., 0, 0]),
              "injector": np.array([2**-.5, 2**-.5, 0]), "holder": np.array([-1., 0, 0])}
PUSH_TARGET = np.array([-0.0003, -0.0012])
PICK_TARGET = np.array([0.0025, -0.0025])
WELL = np.array([0.003, 0.0025])


def manipulator(world, actuators, tool, asset=None, cad=None, *, compact=False, cell_pipettes=False,
                parallel=None, equality=None):
    home = HOMES[tool]
    direction = APPROACHES[tool]
    back = np.r_[direction[:2]/2**.5, 1/2**.5]
    if cell_pipettes and tool in ("injector", "holder"):
        from microscopy.pipettes import pipette_back
        back = pipette_back(tool)
    if parallel is not None:
        from microscopy.parallel_mount import mount_parallel

        parent, evidence = mount_parallel(world, asset, actuators, equality, tool, home, ORIGIN, BENCH_TOP,
                                          back, parallel)
        tool_geometry(parent, actuators, tool, housed_gripper=tool == "gripper")
        if tool == "gripper":
            evidence["gripper"] = dict(profile="original-estimated-electric-gripper",
                actuators=["jaw_a", "jaw_b"], vendor_cad=False,
                appearance_only_housing=True, shank_rise_m=.002)
        if tool != "gripper":
            quat = np.r_[1-back[2], np.cross([0., 0., -1.], back)]
            quat /= np.linalg.norm(quat)
            parent.find(f"geom[@name='{tool}_taper']").set("quat", numbers(quat))
        return evidence
    if cad is not None:
        parent = cad.zaber(asset, world, actuators, tool, home, ORIGIN, BENCH_TOP, direction,
                           compact=compact, back=back)
        if tool == "gripper":
            from microscopy.grippers import parallel_gripper
            parallel_gripper(cad, asset, parent, actuators)
        else:
            tool_geometry(parent, actuators, tool)
        if tool != "gripper":
            quat = np.r_[1-back[2], np.cross([0., 0., -1.], back)]
            quat /= np.linalg.norm(quat)
            parent.find(f"geom[@name='{tool}_taper']").set("quat", numbers(quat))
        return
    mount = body(world, tool + "_mount", ORIGIN + home)
    reach = .35 if tool == "injector" else .21
    base = direction * reach + [0, 0, BENCH_TOP+.012-(ORIGIN[2]+home[2])]
    geom(mount, tool + "_foot", "box", (.06, .048, .012), base, "graphite")
    cylinder(mount, tool + "_post", base, direction * reach + [0, 0, -.025], .013)
    parent = mount
    for i, axis in enumerate(np.eye(3)):
        parent = slide(parent, actuators, f"{tool}_{'xyz'[i]}", axis, (-.016, .016))
        offset = direction * (reach - .02 - i * .012) + [0, 0, -.035 + i * .017]
        geom(parent, f"{tool}_motor_{i}", "box", (.026, .024, .014), offset, "graphite")
        geom(parent, f"{tool}_carriage_{i}", "box", (.019, .020, .006),
             offset + [0, 0, .019], "metal")
        cylinder(parent, f"{tool}_screw_{i}", offset + [-.027, -.02, .018],
                 offset + [.027, -.02, .018], .0025)
    cylinder(parent, tool + "_boom", direction * (reach-.04), direction * .035, .005)
    cylinder(parent, tool + "_collet", direction * .035, direction * .009, .003)
    ET.SubElement(parent, "site", name=tool + "_tcp", size=".00005", group="5")
    tool_geometry(parent, actuators, tool)


def tool_geometry(parent, actuators, tool, *, housed_gripper=False):
    if tool == "gripper":
        if housed_gripper:
            from microscopy.electric_gripper import housing, carriage
            housing(parent)
        for side, sign in (("a", 1), ("b", -1)):
            jaw = body(parent, "jaw_" + side, (0, sign * .0002, 0), gravcomp="1")
            ET.SubElement(jaw, "inertial", pos="0 0 0", mass=".0004",
                          diaginertia="1e-9 1e-9 1e-9")
            ET.SubElement(jaw, "joint", name="jaw_" + side, type="slide",
                          axis=f"0 {sign} 0", range="0 .0016", damping=".001")
            ET.SubElement(actuators, "position", name="jaw_" + side + "_drive",
                          joint="jaw_" + side, kp="50", kv=".02", ctrlrange="0 .0016",
                          forcelimited="true", forcerange="-.005 .005")
            geom(jaw, "jaw_" + side + "_pad", "box", (.00065, .00012, .00035),
                 material="metal", contype="1", conaffinity="1", friction="2 .00001 .000001")
            cylinder(jaw, "jaw_" + side + "_shank", (.00065, 0, 0),
                     (.009, sign * .001, .002 if housed_gripper else 0), .00018)
            if housed_gripper:
                carriage(jaw, side, sign)
    else:
        quaternion = "0.707106781 0 0.707106781 0" if tool in ("probe", "holder") else (
            ".653281482 .653281482 -.27059805 -.27059805")
        ET.SubElement(parent, "geom", name=tool + "_taper", type="mesh",
                      mesh=tool + "_taper_mesh", quat=quaternion,
                      material="glass" if tool == "injector" else "metal")
        geom(parent, tool + "_tip", "sphere", (.00012 if tool == "probe" else .00004,),
             material="cyan" if tool == "probe" else "glass", contype="1", conaffinity="1")


def build_xml(tools=None, *, focal_reference_m=RADIUS, cell_pipettes=False):
    from microscopy.cad_assets import requested_assets
    tools = DEFAULT_TOOLS if tools is None else tuple(tools)
    if not tools or len(set(tools)) != len(tools) or not set(tools) <= HOMES.keys():
        raise ValueError("Select distinct supported microscopy tools")
    cad = requested_assets()
    from microscopy.parallel_mount import requested_parallel
    parallel = requested_parallel()
    from microscopy.stand import stand_profile
    instrument = stand_profile(cad is not None)
    compact = instrument == "te2000-s-reference" and (cad is not None or parallel is not None)
    reference_40x = instrument == "te2000-s-reference" and cell_pipettes
    glass_thickness_m = .0012 if reference_40x else .0005
    root = ET.Element("mujoco", model="motorized_microscopy_workstation")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".001", integrator="implicitfast",
                  gravity="0 0 -9.81", cone="elliptic", iterations="100", noslip_iterations="3")
    ET.SubElement(root, "size", nuserdata="2")
    ET.SubElement(root, "statistic", center="0 0 .94", extent=".85")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "map", znear=".00005", zfar="10", shadowclip="2")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".55 .55 .55",
                  specular=".15 .15 .15")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", mass=".001",
                  solref=".003 1", solimp=".95 .99 .0001")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1=".18 .24 .31", rgb2=".06 .10 .15", width="512", height="3072")
    for name, color, shiny in (
        ("ivory", ".82 .84 .83 1", .25), ("graphite", ".15 .17 .19 1", .3),
        ("metal", ".48 .57 .65 1", .8), ("cyan", ".08 .60 .65 1", .4),
        ("glass", ".65 .83 .88 .3", .9), ("amber", ".95 .48 .12 1", .2),
        ("blue", ".10 .38 .86 1", .4), ("violet", ".54 .25 .75 1", .4),
        ("rubber", ".045 .048 .052 1", .08), ("bronze", ".48 .31 .19 1", .7),
        ("lens", ".04 .09 .12 .80", .95), ("filter_orange", "1 .32 .015 .45", .5),
        ("led_green", ".1 .95 .25 1", .2),
        ("hose", ".58 .69 .72 .65", .35),
    ):
        ET.SubElement(asset, "material", name=name, rgba=color,
                      specular=str(shiny), shininess=str(shiny))
    frustum(asset, "probe_taper_mesh", .009, .0011, .00012)
    frustum(asset, "injector_taper_mesh", .012 if compact else .009, .0005 if compact else .001, .00004)
    if "holder" in tools:
        frustum(asset, "holder_taper_mesh", .009, .001, .00004)
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    ET.SubElement(world, "light", pos="-.8 -.8 2", dir=".3 .3 -1", diffuse=".9 .9 .9")
    ET.SubElement(world, "light", pos=".7 .4 1.8", dir="-.3 -.3 -1",
                  diffuse=".6 .7 .8", castshadow="false")
    geom(world, "floor", "plane", (3, 3, .01), material="graphite")
    geom(world, "bench", "box", (.63, .40, .025), (0, 0, BENCH_TOP-.025), "metal")
    for x in (-.54, .54):
        for y in (-.31, .31):
            cylinder(world, f"leg_{x}_{y}", (x, y, .02), (x, y, BENCH_TOP-.05), .027, "graphite")
    # Optical breadboard pattern is visual; samples collide only with their chamber.
    for x in np.arange(-.55, .56, .05):
        for y in np.arange(-.32, .33, .05):
            geom(world, f"hole_{x:.2f}_{y:.2f}", "cylinder", (.002, .0002),
                 (x, y, BENCH_TOP+.0003), "graphite")
    scope = None
    if instrument == "te2000-s-reference":
        from microscopy.commercial_stand import build_stand
        from microscopy.stage import stage_profile
        scope, evidence = build_stand(asset, world, ORIGIN, BENCH_TOP,
                                     stage_supports=stage_profile() == "reference",
                                     collection_reference=reference_40x)
        if cad is not None:
            cad.evidence["microscope"] = evidence
        else:
            custom = ET.SubElement(root, "custom")
            ET.SubElement(custom, "text", name="microscopy_assets", data=json.dumps(
                dict(profile="reference", microscope=evidence, assemblies=[])))
    elif cad is None:
        build_instrument(asset, world, ORIGIN)
    else:
        scope = cad.openframe(asset, world, ORIGIN, BENCH_TOP)
    focus = slide(world, actuators, "focus", (0, 0, 1), (-.001, .001), .03, 600, 4)
    from microscopy.cad_optics import camera_profile
    if cad is not None and camera_profile() == "assembled":
        from microscopy.focus_assembly import mount_focus
        mount_focus(cad, asset, scope, focus, ORIGIN, BENCH_TOP, focal_reference_m)
    elif reference_40x:
        from microscopy.reference_objective import build_reference_objective
        objective = build_reference_objective(asset, focus, ORIGIN, focal_reference_m)
        from microscopy.reference_focus_mount import build_focus_mount
        objective["focus_mount"] = build_focus_mount(asset, scope, focus, ORIGIN, BENCH_TOP,
                                                    objective["legacy_shoulder_offset_m"], cad)
        from microscopy.inspection_lighting import add_inspection_fill
        objective["inspection_lighting"] = add_inspection_fill(world, ORIGIN)
        from microscopy.reference_detection import build_detection_path
        objective["detection_path"] = build_detection_path(asset, scope, focus, ORIGIN, BENCH_TOP,
            objective["legacy_shoulder_offset_m"], evidence["shell_profiles_m"])
        camera(world, "objective_detail", ORIGIN+[-.15, -.18, -.14],
               ORIGIN+[-.025, 0., -.040], 28)
        if cad is not None:
            cad.evidence["objective"] = objective
        else:
            metadata = root.find("custom/text[@name='microscopy_assets']")
            evidence = json.loads(metadata.get("data"))
            evidence["objective"] = objective
            metadata.set("data", json.dumps(evidence))
    else:
        cylinder(focus, "nosepiece", ORIGIN + [0, 0, -.06], ORIGIN + [0, 0, -.05], .040)
        detail_objectives(focus, ORIGIN)
    from microscopy.stage import build_stage
    stage = build_stage(asset, world, actuators, ORIGIN, cad, instrument,
                        glass_thickness_m=glass_thickness_m,
                        support_arm_height_m=-.074 if reference_40x else -.064)
    ET.SubElement(stage, "site", name="sample_plane", pos="0 0 0", group="5", size=".00005")
    geom(stage, "sample_glass", "box", (.023, .023, glass_thickness_m/2),
         (0, 0, -glass_thickness_m/2),
         "glass", contype="1", conaffinity="1", friction=".5 .00001 .000001")
    for i in range(48):
        angle = i * 2 * math.pi / 48
        cylinder(stage, f"dish_wall_{i}", (.023 * math.cos(angle), .023 * math.sin(angle), 0),
                 (.023 * math.cos(angle), .023 * math.sin(angle), .003), .001, "glass")
    for i in range(16):
        angle = i * 2 * math.pi / 16
        geom(stage, f"well_wall_{i}", "box", (.00025, .00016, .00035),
             (* (WELL + .0012 * np.array([np.cos(angle), np.sin(angle)])), .00035),
             "violet", contype="1", conaffinity="1", euler=f"0 0 {angle + math.pi / 2}")
    ET.SubElement(stage, "site", name="well_center", pos=numbers([*WELL, .00035]), group="5", size=".00005")
    parallel_evidence = []
    equality = ET.SubElement(root, "equality") if parallel is not None else None
    for tool in tools:
        evidence = manipulator(world, actuators, tool, asset, cad, compact=compact, cell_pipettes=cell_pipettes,
                               parallel=parallel, equality=equality)
        if evidence:
            parallel_evidence.append(evidence)
    for name, xy, material in (("bead_push", (-.0026, -.0012), "amber"),
                               ("bead_pick", (.0008, -.0008), "blue")):
        sample = body(world, name, ORIGIN + [*xy, RADIUS])
        # A small passive resistance damps free spin consistently through the
        # shared generalized-force bridge. It is an uncalibrated demo parameter.
        ET.SubElement(sample, "joint", type="free", name=name + "_free", damping=str(BEAD_DAMPING))
        geom(sample, name + "_geom", "sphere", (RADIUS,), material=material,
             mass=str(2500 * 4 / 3 * math.pi * RADIUS**3), contype="1", conaffinity="1",
             condim="6", friction="1 .00001 .000001")
    # Pressure controller, electrical cables and pneumatic tubing.
    if compact:
        from microscopy.pneumatics import compact_console
        compact_console(world, ORIGIN, HOMES, tools, BENCH_TOP)
    else:
        geom(world, "pressure_controller", "box", (.068, .055, .025), (.37, .15, BENCH_TOP+.030), "ivory")
        geom(world, "pressure_display", "box", (.028, .0008, .009), (.35, .094, BENCH_TOP+.035), "graphite")
        cylinder(world, "pressure_dial", (.405, .092, BENCH_TOP+.035), (.405, .084, BENCH_TOP+.035), .008, "cyan")
    if cad is not None:
        from microscopy.cad_assets import TOOL_EXTENSIONS
        extension = .152 if compact else TOOL_EXTENSIONS["injector"]
        pneumatic_end = tuple(ORIGIN+HOMES["injector"]+np.array([.5, .5, 2**-.5])*extension)
    else:
        pneumatic_end = (.225, .158, .999)
    cable_paths = []
    if "injector" in tools and not compact:
        cable_paths.extend((((.37, .15, BENCH_TOP+.055), (.35, .26, .88)),
                            ((.35, .26, .88), pneumatic_end)))
    if "probe" in tools:
        cable_paths.append(((-.21, -.08, .90), (-.31, .19, BENCH_TOP+.005)))
    if "gripper" in tools:
        cable_paths.append(((.21, -.07, .90), (.31, .19, BENCH_TOP+.005)))
    for i, (start, end) in enumerate(cable_paths):
        cylinder(world, f"cable_{i}", start, end, .0025, "cyan" if i < 2 else "graphite")
    cylinder(world, "monitor_pillar", (-.40, .23, BENCH_TOP+.005), (-.40, .23, .96), .015, "graphite")
    geom(world, "monitor", "box", (.135, .014, .085), (-.40, .23, 1.045), "graphite")
    geom(world, "monitor_panel", "box", (.12, .001, .069), (-.40, .215, 1.045), "lens")
    # Decorative display layout; live instrument readings are in the webpage.
    for i, height in enumerate((.018, .024, .035, .043, .029, .052)):
        geom(world, f"display_bar_{i}", "box", (.008, .0002, height/2),
             (-.48+i*.017, .2138, 1.007+height/2), "cyan")
    for i, width in enumerate((.048, .033, .025)):
        geom(world, f"display_line_{i}", "box", (width, .0002, .001),
             (-.389, .2138, 1.095-i*.006), "metal")
    camera(world, "workstation_overview", (.78, -.99, 1.52), np.array([0, 0, 1.04]), 45)
    camera(world, "instrument_closeup", (.50, -.78, 1.42), np.array([0, -.025, 1.035]), 45)
    from microscopy.lab_appearance import add_lab_appearance
    add_lab_appearance(root)
    if reference_40x:
        from microscopy.reference_detection import add_detection_view
        add_detection_view(world, ORIGIN, BENCH_TOP)
    if cad is not None:
        cad.evidence["assemblies"].extend(parallel_evidence)
        cad.record(root)
    elif parallel_evidence:
        text = root.find("custom/text[@name='microscopy_assets']")
        if text is None:
            custom = root.find("custom")
            if custom is None:
                custom = ET.SubElement(root, "custom")
            text = ET.SubElement(custom, "text", name="microscopy_assets")
            evidence = dict(profile="reference", assemblies=[])
        else:
            evidence = json.loads(text.get("data"))
        evidence["assemblies"].extend(parallel_evidence)
        text.set("data", json.dumps(evidence))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
