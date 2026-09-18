"""XY stages preserve the sample frame independently of the instrument stand.

The optional vendor meshes depict rigid parts. Their grouping uses the vendor
STEP and virtual-component feature labels; drive dynamics and original sample
fixtures remain uncalibrated. Public downloads do not establish reuse rights.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import body, cylinder, geom, rounded_box, slide

STAGE_SHA256 = "837759b532e1544c4a9bb481edc33ead5ba06999e54ef9aa7940c2f44d7b9d71"
ADAPTER_SHA256 = "b166f5a346be5950360b01ad2b1873c871aaecfbe209af8419fb3e560815d13c"
ASSEMBLY_SHA256 = "6ae994af70a2899ada83f6e88b1c48364cfe932025bf1321159cfbc0d992d926"
# CAD Y is up. The installation drawing places the controller on the right
# and adapters fore/aft: lower CAD-Z motion becomes world Y, upper CAD-X
# becomes world X. Body frames keep Cartesian controller axes unrotated.
STAGE_TO_WORLD = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
ADAPTER_TO_STAGE = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
SOURCE_TOP_PLANE_M = .0272
SAMPLE_SEAT_HEIGHT_M = .003  # Original insert estimate, not a vendor measurement.
# A rigid 20 mm forward mounting offset clears the rear motor cover from the
# reference illumination pillar at the sampled lower-axis endpoints. The
# original insert keeps the glass/sample origin on the microscope axis.
WINDOW_OFFSET_M = .020
STAGE_OFFSET_M = np.array([0., -WINDOW_OFFSET_M, -SOURCE_TOP_PLANE_M-SAMPLE_SEAT_HEIGHT_M])
OPERATING_LIMITS_M = {"stage_x": [-.004, .004], "stage_y": [-.004, .004]}
PART_LINKS = {
    "fixed": (0, 1, 2, 3, 26, 27, 28, 29, 30, 31),
    "lower": (*range(4, 21), *range(32, 38)),
    "upper": tuple(range(21, 26)),
}


def stage_profile():
    profile = os.environ.get("HOOKE_MICROSCOPY_STAGE", "reference")
    if profile not in ("reference", "x-asr100"):
        raise ValueError("HOOKE_MICROSCOPY_STAGE must be reference or x-asr100")
    return profile


def build_stage(asset, world, actuators, origin, cad, instrument, *, glass_thickness_m=.0005,
                support_arm_height_m=-.064):
    profile = stage_profile()
    if profile == "x-asr100":
        if cad is None or instrument != "te2000-s-reference":
            raise ValueError("The X-ASR/AP114 local assembly requires CAD assets and the TE2000-S reference stand")
        return _vendor_stage(asset, world, actuators, origin, cad, glass_thickness_m,
                             support_arm_height_m)
    return _reference_stage(asset, world, actuators, origin, cad)


def _reference_stage(asset, world, actuators, origin, cad):
    stage = body(world, "stage_base", origin)
    if cad is not None:
        for x in (-.070, .070):
            for y in (-.055, .055):
                cylinder(stage, f"stage_riser_{x}_{y}", (x, y, -.049), (x, y, -.014), .007)
    for i, axis in enumerate(np.eye(2, 3)):
        stage = slide(stage, actuators, f"stage_{'xy'[i]}", axis, (-.004, .004), .2, 8000, 70)
    for i, (size, pos) in enumerate((((.120, .035, .006), (0, .071, -.007)),
                                    ((.120, .035, .006), (0, -.071, -.007)),
                                    ((.035, .036, .006), (.085, 0, -.007)),
                                    ((.035, .036, .006), (-.085, 0, -.007)))):
        rounded_box(asset, stage, f"stage_rail_{i}", size, pos, .004, "graphite")
    for x in (-.105, .105):
        for y in (-.085, .085):
            geom(stage, f"stage_screw_{x}_{y}", "cylinder", (.003, .0006), (x, y, -.0005), "metal")
    if cad is not None:
        for i, axis in enumerate(("x", "y")):
            node = world.find(f".//body[@name='stage_{axis}']")
            geom(node, f"stage_motor_{axis}", "box", (.027, .017, .016),
                 (-.146 if i == 0 else -.076, .070 if i == 0 else -.132, -.026), "graphite")
    return stage


def _vendor_stage(asset, world, actuators, origin, cad, glass_thickness_m, support_arm_height_m):
    report = cad.read("x-asr100", STAGE_SHA256)
    adapter = cad.read("ap114", ADAPTER_SHA256)
    if len(report["parts"]) != 38 or len(adapter["parts"]) != 2:
        raise ValueError("Unexpected X-ASR/AP114 STEP solid counts")
    fixed = body(world, "stage_base", origin)
    # MuJoCo requires a finite centre of mass even on this fixed, massless
    # visual parent. This numerical anchor has no moving joint or OEM mass.
    ET.SubElement(fixed, "inertial", pos="0 0 0", mass=".000001", diaginertia=".00000001 .00000001 .00000001")
    lower = slide(fixed, actuators, "stage_y", (0, 1, 0), (-.050, .050), .2, 8000, 70)
    upper = slide(lower, actuators, "stage_x", (1, 0, 0), (-.060, .060), .2, 8000, 70)
    nodes = dict(fixed=fixed, lower=lower, upper=upper)
    for link, indices in PART_LINKS.items():
        for i in indices:
            material = "graphite" if i in (9, 19, 20, 26, 32) else "metal"
            cad.attach(asset, nodes[link], "x-asr100", i, STAGE_OFFSET_M, STAGE_TO_WORLD,
                       material, prefix="stage")
    for i in range(2):
        cad.attach(asset, fixed, "ap114", i, STAGE_OFFSET_M,
                   STAGE_TO_WORLD @ ADAPTER_TO_STAGE, "graphite", prefix="stage_adapter")
    _reference_mounts(asset, fixed, support_arm_height_m)
    _sample_insert(asset, upper, glass_thickness_m)
    cad.evidence["stage"] = dict(
        profile="x-asr100", official_cad=True, one_to_one_verified=False,
        stand_mount_accepted=False,
        stand_mount_status="Revised drawing-based orientation; reference-stand installation requires a fresh surface and motion audit",
        stage_source="https://www.zaber.com/fs/series/X-ASR-E/7/X-ASR100B120B-SE03D12/docs/X-ASR100B120B-SE03D12.step",
        stage_sha256=STAGE_SHA256,
        adapter_source="https://www.zaber.com/fs/accessories/AP114/docs/AP114.step",
        adapter_sha256=ADAPTER_SHA256,
        assembly_source="https://www.zaber.com/fs/series/X-ASR-E/7/docs/ASR.sldasm",
        assembly_sha256=ASSEMBLY_SHA256,
        partition_basis="Virtual ASR_base/mid/top and NMS11_motor feature labels, STEP bounds and rail interfaces; no native mates replay",
        part_links={k: list(v) for k, v in PART_LINKS.items()},
        metre_meshes=40, geometry_scale=1., source_length_unit="mm",
        stage_to_world_rotation=STAGE_TO_WORLD.tolist(), translation_m=STAGE_OFFSET_M.tolist(),
        adapter_to_stage_rotation=ADAPTER_TO_STAGE.tolist(), adapter_translation_m=[0., 0., 0.],
        travel_m=dict(stage_x=.120, stage_y=.100),
        operating_limits_m=OPERATING_LIMITS_M,
        operating_limit_basis="Conservative central XY envelope; modeled objectives intersect carriage at some full-travel poses. Coarse objective retraction is not implemented; hardware travel is retained.",
        zero_pose="STEP nominal actuator pose; original insert offsets the sample from the window centre by 20 mm; not factory homing calibration",
        dynamics="Uncalibrated existing 0.2 kg slide masses, position gains and rigid joints; fixed parent uses a 1 mg numerical anchor; not vendor precision, inertia or speed",
        estimated_parts=["160 x 110 mm open sample insert", "stand supports and arms", "fasteners"],
        mounting_scope="Six stage/adapter coaxial holes at nominal STEP pose; Nikon stand interfaces, insert and full travel clearance not verified",
        glass_thickness_m=glass_thickness_m, support_arm_height_m=support_arm_height_m,
        rights="Public vendor downloads; explicit reuse license not established; optional private local inspection",
    )
    return upper


def _reference_mounts(asset, fixed, arm_z):
    # AP114's outer holes are a 280 x 80 mm pattern. The support posts and
    # bolts are independent geometry, not an OEM Nikon mounting assembly.
    for x in (-.040, .040):
        for mount_y in (-.140, .140):
            y = mount_y-WINDOW_OFFSET_M
            inner = np.copysign(.068, mount_y)-WINDOW_OFFSET_M
            cylinder(fixed, f"stage_support_column_{x}_{y}", (x, inner, -.117),
                     (x, inner, arm_z), .011, "metal", mass="0")
            rounded_box(asset, fixed, f"stage_support_arm_{x}_{y}",
                        (.014, .043, .004), (x, (y+inner)/2, arm_z), .001,
                        "graphite", mass="0")
            cylinder(fixed, f"stage_support_{x}_{y}", (x, y, arm_z+.004), (x, y, -.0302),
                     .007, "metal", mass="0")
            geom(fixed, f"stage_mount_head_{x}_{y}", "cylinder", (.0045, .0015),
                 (x, y, -.0302+.00794-.0015), "metal", mass="0")
    for x in (-.075, 0., .075):
        for z in (-.086, .086):
            point = STAGE_TO_WORLD @ np.array([x, .0062, z]) + STAGE_OFFSET_M
            geom(fixed, f"stage_adapter_bolt_{x}_{z}", "cylinder", (.003, .00145),
                 point, "metal", mass="0")


def _sample_insert(asset, stage, glass_thickness_m):
    # Original 160 x 110 mm insert bridges the open stage to the existing
    # glass bottom. The outer frame follows the shifted OEM window; its
    # 40 mm square optical opening and the glass remain on the sample axis.
    # Seat thickness and support shape are estimates; sample Z stays at zero.
    front, rear = -.055-WINDOW_OFFSET_M, .055-WINDOW_OFFSET_M
    halfheight = (SAMPLE_SEAT_HEIGHT_M-glass_thickness_m)/2
    z = -(SAMPLE_SEAT_HEIGHT_M+glass_thickness_m)/2
    for i, (size, pos) in enumerate((((.020, (-.020-front)/2, halfheight),
                                     (0, (front-.020)/2, z)),
                                    ((.020, (rear-.020)/2, halfheight),
                                     (0, (rear+.020)/2, z)),
                                    ((.030, .055, halfheight), (-.050, -WINDOW_OFFSET_M, z)),
                                    ((.030, .055, halfheight), (.050, -WINDOW_OFFSET_M, z)))):
        rounded_box(asset, stage, f"stage_insert_{i}", size, pos, .001, "graphite", mass="0")
