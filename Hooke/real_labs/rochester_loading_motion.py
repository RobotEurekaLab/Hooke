"""Contact-only load, release and retrieve of Rochester's free inert plate.

This bounded operation drives an authored manual-tool surrogate. It performs no
pose assignment, equality switching or implicit reset, and leaves the original
pipetting axes and sequencer controls unchanged. It is not a liquid process.
"""

import hashlib
import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state

PREFIX = "rochester_loading__"
SEQUENCE = (
    ("settle", None, 0),
    ("unpark_raise", "loader_z", 0.105),
    ("unpark_retract", "loader_y", -0.82),
    ("unpark_lower", "loader_z", 0),
    ("open", "front_door_hinge", 1.75),
    ("fork_under_stage", "loader_y", -0.46),
    ("lift_stage", "loader_z", 0.070),
    ("align_x", "loader_x", -0.70),
    ("insert", "loader_y", -0.1635),
    ("deposit", "loader_z", 0),
    ("withdraw", "loader_y", -0.82),
    ("park_x", "loader_x", 0),
    ("park_raise", "loader_z", 0.105),
    ("park_forward", "loader_y", 0),
    ("park_lower", "loader_z", 0),
    ("close", "front_door_hinge", 0),
    ("reopen", "front_door_hinge", 1.75),
    ("retrieve_unpark_raise", "loader_z", 0.105),
    ("retrieve_unpark_retract", "loader_y", -0.82),
    ("retrieve_unpark_lower", "loader_z", 0),
    ("realign", "loader_x", -0.70),
    ("approach_plate", "loader_y", -0.1635),
    ("pick_from_deck", "loader_z", 0.070),
    ("remove", "loader_y", -0.46),
    ("return_x", "loader_x", 0),
    ("return_to_stage", "loader_z", 0),
    ("park_fork", "loader_y", -0.82),
    ("finish_raise", "loader_z", 0.105),
    ("finish_forward", "loader_y", 0),
    ("finish_lower", "loader_z", 0),
    ("close_finish", "front_door_hinge", 0),
)


def run_rochester_plate_loading(session, *, observe=None):
    """Execute the validated loading route from home and return measured evidence.

    ``observe(phase, model, data)`` optionally records the actual phase-end state.
    It must not mutate the live model or state. Failure holds current measured
    positions for the four task controls and raises; it never resets the scene.
    Call ``InstrumentSession.reset`` separately to restore the initial placement.
    """
    m, d = session.model, session.data
    if d.model is not m or not np.isfinite(d.time):
        raise ValueError("The loading session needs its own finite model/data")
    interface = session.equipment["liquid_handler"].get("plate_loading_interface")
    if not interface or interface.get("revision") != 1:
        raise ValueError("Rochester contact-loading interface revision 1 required")
    keys = ("front_door_hinge", "loader_x", "loader_y", "loader_z")
    controls = [session.controls["liquid_handler", k] for k in keys]
    indices = np.array([c.actuator for c in controls])
    addresses = np.array([m.jnt_qposadr[c.joint] for c in controls])
    expected_ranges = ((0, 1.75), (-0.70, 0), (-0.82, 0), (0, 0.105))
    for c, span, units in zip(controls, expected_ranges, ("rad", "m", "m", "m")):
        if (
            c.mode != "position"
            or c.units != units
            or not np.allclose((c.lower, c.upper), span)
        ):
            raise ValueError("The versioned hinge/XYZ controls do not match this route")
    if not np.allclose(d.qpos[addresses], 0, atol=1e-5) or not np.allclose(
        d.ctrl, 0, atol=1e-8
    ):
        raise ValueError("Loading must begin with all authored controls at home")
    plate = m.body(interface["body"]).id
    plate_geom = m.geom(interface["collision_geom"]).id
    fj = m.joint(interface["freejoint"]).id
    if m.jnt_type[fj] != mujoco.mjtJoint.mjJNT_FREE or m.jnt_bodyid[fj] != plate:
        raise ValueError("The plate must be genuinely free")
    if any(plate in (int(a), int(b)) for a, b in zip(m.eq_obj1id, m.eq_obj2id)):
        raise ValueError("No plate equality attachment is permitted")
    stage, deck = np.array(interface["stage_world"]), np.array(interface["dock_world"])
    mujoco.mj_forward(m, d)
    if np.linalg.norm(d.xpos[plate] - stage) > 0.001:
        raise ValueError("Free plate is not on the external staging supports")
    fork = m.body(PREFIX + "body_loader_z").id
    signature = _model_signature(m)
    warning0 = d.warning.number.copy()
    expected = d.ctrl.copy()
    original = [
        m.body(session.equipment[k]["process"]["sample_body"]).id
        for k in ("liquid_handler", "sequencer")
    ]
    original_pos = d.xpos[original].copy()
    held = [c for c in session.controls.values() if c.actuator not in indices]
    if np.max(np.abs(d.qvel)) > 0.01:
        raise ValueError("Loading requires a settled initial state")
    names = [m.geom(i).name for i in range(m.ngeom)]
    new = np.array([n.startswith(PREFIX) for n in names])
    force = np.zeros(6)
    phases, trace = [], hashlib.sha256()
    steps = 0
    max_penetration = 0.0
    peak_plate_force = 0.0
    start_time = float(d.time)
    phase_steps = math.ceil(3.6 / m.opt.timestep)
    ramp_steps = math.ceil(2.0 / m.opt.timestep)

    def check():
        nonlocal max_penetration, peak_plate_force
        if not np.array_equal(d.ctrl, expected):
            raise RuntimeError("A control was changed outside the loading operation")
        if (
            not np.array_equal(d.warning.number, warning0)
            or not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
        ):
            raise RuntimeError("Physics warning or nonfinite state during loading")
        if np.max(np.linalg.norm(d.xpos[original] - original_pos, axis=1)) > 0.002:
            raise RuntimeError("An original preloaded carrier was disturbed")
        if d.xpos[plate, 2] < stage[2] - 0.035 or abs(d.xquat[plate, 0]) < 0.995:
            raise RuntimeError("The free plate was dropped or tilted")
        for c in held:
            if abs(d.qpos[m.jnt_qposadr[c.joint]]) > (
                0.002 if c.units == "m" else 0.01
            ):
                raise RuntimeError("An inactive control drifted during loading")
        for i, c in enumerate(d.contact):
            a, b = int(c.geom1), int(c.geom2)
            if not (new[a] or new[b]):
                continue
            mujoco.mj_contactForce(m, d, i, force)
            max_penetration = max(max_penetration, max(0.0, -float(c.dist)))
            other = b if a == plate_geom else a if b == plate_geom else None
            supported = other is not None and (
                names[other].startswith(PREFIX + "stage_pad_")
                or names[other].startswith(PREFIX + "deck_pad_")
                or m.geom_bodyid[other] == fork
            )
            if supported:
                peak_plate_force = max(peak_plate_force, float(force[0]))
                if c.dist < -0.0005 or force[0] > 3:
                    raise RuntimeError("Excess plate contact force or penetration")
            elif force[0] > 0.10 or c.dist < -0.0001:
                raise RuntimeError(
                    f"Unexpected loading contact: {names[a]} / {names[b]}"
                )

    destination = stage
    try:
        for phase, key, target in SEQUENCE:
            if _model_signature(m) != signature:
                raise RuntimeError("The model was changed during loading")
            check()
            c = session.controls["liquid_handler", key] if key else None
            start = expected[c.actuator] if c else 0.0
            for k in range(phase_steps):
                check()
                if c:
                    u = min((k + 1) / ramp_steps, 1.0)
                    expected[c.actuator] = start + (target - start) * u * u * (
                        3 - 2 * u
                    )
                    d.ctrl[c.actuator] = expected[c.actuator]
                before_time, before_warnings = float(d.time), _warning_state(d)
                mujoco.mj_step(m, d)
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, {"ctrl": expected}
                )
                if event is not None:
                    raise RuntimeError(
                        f"Native physics interruption during loading: {event}"
                    )
                steps += 1
                check()
                trace.update(np.r_[d.time, d.qpos, d.qvel, d.ctrl].tobytes())
            measured = d.qpos[addresses].copy()
            targets = expected[indices]
            if np.max(np.abs(measured - targets)) > 0.0007:
                raise RuntimeError(f"Loading axis did not reach {phase}")
            carried = phase in {
                "lift_stage",
                "align_x",
                "insert",
                "pick_from_deck",
                "remove",
                "return_x",
            }
            if phase == "deposit":
                destination = deck
            elif phase == "return_to_stage":
                destination = stage
            target_pos = (
                d.xpos[fork] + np.array([0, 0, 0.015]) if carried else destination
            )
            error = float(np.linalg.norm(d.xpos[plate] - target_pos))
            if error > 0.002:
                raise RuntimeError(f"Plate placement failed at {phase}: {error} m")
            support_names = []
            for i, contact in enumerate(d.contact):
                if plate_geom not in (contact.geom1, contact.geom2):
                    continue
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] > 0.001:
                    support_names.append(
                        names[
                            int(
                                contact.geom2
                                if contact.geom1 == plate_geom
                                else contact.geom1
                            )
                        ]
                    )
            if not support_names:
                raise RuntimeError(f"Plate lacks measured support at {phase}")
            row = dict(
                phase=phase,
                time_s=float(d.time),
                plate_position=d.xpos[plate].tolist(),
                plate_quaternion=d.xquat[plate].tolist(),
                position_error_m=error,
                measured_controls=dict(zip(keys, measured.tolist())),
                support_geoms=sorted(set(support_names)),
            )
            phases.append(row)
            if observe is not None:
                q, v, ctrl = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
                time = float(d.time)
                observe(phase, m, d)
                if (
                    time != d.time
                    or not np.array_equal(q, d.qpos)
                    or not np.array_equal(v, d.qvel)
                    or not np.array_equal(ctrl, d.ctrl)
                ):
                    raise RuntimeError(
                        "The observation callback mutated the live state"
                    )
        if _model_signature(m) != signature:
            raise RuntimeError("The observation callback changed the model")
        return dict(
            passed=True,
            revision=1,
            steps=steps,
            elapsed_simulation_s=float(d.time) - start_time,
            trajectory_sha256=trace.hexdigest(),
            phases=phases,
            max_plate_position_error_m=max(x["position_error_m"] for x in phases),
            maximum_new_contact_penetration_m=max_penetration,
            peak_single_plate_contact_force_n=peak_plate_force,
            reset_called=False,
            pose_assignment=False,
            equality_switching=False,
            scope="Contact-only inert plate loading, door closure/reopening and retrieval; no liquid/tip/biological process",
        )
    except Exception:
        d.ctrl[indices] = np.clip(
            d.qpos[addresses],
            m.actuator_ctrlrange[indices, 0],
            m.actuator_ctrlrange[indices, 1],
        )
        raise
