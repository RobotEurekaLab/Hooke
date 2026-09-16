"""CPU-only structural preflight; support is distinct from parity qualification."""

import xml.etree.ElementTree as ET
import numpy as np

VERSION = "hooke-native-capabilities-v2"
KNOWN_PLUGINS = {"mjlab.sdf.thread", "mjlab.passive.detent"}
FEATURES = {
    "free_hinge_slide_joints": "Implemented; compound joints add declared auxiliary inertia.",
    "body_wrenches": "Forwarded through observed-state kinematic Jacobians.",
    "connect_equalities": "Compliant native linear springs; impedance mapping is approximate.",
    "weld_equalities": "Native fixed joints; source soft weld compliance is uncalibrated.",
    "static_heightfields": "Source samples become a closed triangle terrain mesh; contact parity is unqualified.",
    "affine_joint_and_fixed_tendon_actuators": "Implemented within supported gain, bias and transmission types.",
    "thread_sdf": "Original visual mesh; native voxel SDF collision remains approximate.",
    "textures_cameras_liquid_surfaces": "Original assets and dynamic overlays retained; pixels differ.",
    "thermal": "Opt-in two-node energy model, uncalibrated.",
    "capillary_flow": "Opt-in laminar filled-tube flow and conservative reservoir volumes, uncalibrated.",
}
UNSUPPORTED = {
    "ball_joints": "Quaternion joint/velocity adapter required.",
    "moving_heightfields": "Moving terrain collision/dynamics adapter required.",
    "flexible_bodies": "Native deformable state and force adapter required.",
    "nonlinear_dynamic_actuators": "Actuator state/dynamics adapter required.",
    "arbitrary_plugins": "Explicit native/source-force plugin adapter required.",
    "actuator_or_sensor_plugins": "Explicit actuator/sensor plugin adapter required.",
    "actuator_transmission": "Only joint and fixed-tendon transmissions are implemented.",
    "joint_actuator_gearing": "Joint actuators require a unit scalar gear.",
    "spatial_tendon_actuators": "Spatial tendon actuator forces require another adapter.",
    "equality_type": "Only connect, weld and affine joint equalities are implemented.",
    "nonlinear_joint_equality": "Nonlinear joint coupling requires another adapter.",
}


def registry():
    return {
        "version": VERSION,
        "features": FEATURES,
        "unsupported": UNSUPPORTED,
        "parity_qualified": False,
        "scientific_process_validated": False,
    }


def preflight(model, xml):
    blockers = []
    if np.any(model.jnt_type == 1):
        blockers.append("ball_joints")
    terrain = np.flatnonzero(model.geom_type == 1)
    if np.any(model.body_weldid[model.geom_bodyid[terrain]] != 0):
        blockers.append("moving_heightfields")
    if model.nflex:
        blockers.append("flexible_bodies")
    plugins = {
        p.get("plugin") for p in ET.fromstring(xml).findall("./extension/plugin")
    }
    unknown = plugins - KNOWN_PLUGINS
    if unknown:
        blockers.append("arbitrary_plugins")
    if np.any(model.actuator_plugin >= 0) or np.any(model.sensor_plugin >= 0):
        blockers.append("actuator_or_sensor_plugins")
    if (
        np.any(model.actuator_dyntype != 0)
        or np.any(model.actuator_gaintype != 0)
        or np.any(~np.isin(model.actuator_biastype, [0, 1]))
    ):
        blockers.append("nonlinear_dynamic_actuators")
    if np.any(~np.isin(model.actuator_trntype, [0, 3])):
        blockers.append("actuator_transmission")
    tendons = model.actuator_trnid[model.actuator_trntype == 3, 0]
    for tendon in np.unique(tendons):
        start = model.tendon_adr[tendon]
        count = model.tendon_num[tendon]
        if np.any(model.wrap_type[start : start + count] != 1):
            blockers.append("spatial_tendon_actuators")
            break
    joint_actuators = np.flatnonzero(model.actuator_trntype == 0)
    if np.any(
        model.actuator_gear[joint_actuators] != np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ):
        blockers.append("joint_actuator_gearing")
    if np.any(~np.isin(model.eq_type, [0, 1, 2])):
        blockers.append("equality_type")
    if np.any((model.eq_type == 2) & np.any(model.eq_data[:, 2:5] != 0, axis=1)):
        blockers.append("nonlinear_joint_equality")
    return dict(
        version=VERSION,
        can_attempt_native=not blockers,
        blockers=blockers,
        unknown_plugins=sorted(unknown),
        parity_qualified=False,
        scope="Structural rejection before native startup; runtime limits still apply.",
    )


def require_native(task):
    report = preflight(task.model, task.spec.to_xml())
    if report["blockers"]:
        raise NotImplementedError(
            "Native preflight rejected: " + ", ".join(report["blockers"])
        )
    return report
