"""Measured geometric probe approach and line scan; no optical response model.

The surface frame comes from the actual clamped coupon geometry. Fine motion is
observed in metres after every CPU physics step. Reported numerical errors are
simulation state measurements, not hardware precision or calibrated AFM data.
"""

import math
import hashlib
from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports


CONTROLS = (
    "specimen_translation",
    "inspection_head_height",
    "fine_scan_x",
    "fine_approach_z",
)


@lru_cache(maxsize=1)
def _canonical_binding():
    """Bind this bounded demo to its checked authored scene, without live edits.

    Arbitrary model variants need a separately validated operation contract.
    Comparing the compiled model covers meshes, body attachments, pair filters,
    actuator limits and solver options, including normally hidden geometry.
    """
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene("lmu_hybrid_snom")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    expected = InstrumentSession(model, manifest)
    return {
        "model_signature": _model_signature(model),
        "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "controls": expected.controls,
        "contype": model.geom_contype.copy(),
        "conaffinity": model.geom_conaffinity.copy(),
        "disableflags": int(model.opt.disableflags),
        "enableflags": int(model.opt.enableflags),
    }


def _validate_binding(session, equipment):
    if equipment != "snom_table" or session.manifest.get("id") != "lmu_hybrid_snom":
        raise ValueError("This operation is bound to the authored LMU assembly")
    if session.data.model is not session.model:
        raise ValueError("Physics state belongs to a different model")
    expected = _canonical_binding()
    if (
        int(session.model.opt.disableflags) != expected["disableflags"]
        or int(session.model.opt.enableflags) != expected["enableflags"]
    ):
        raise ValueError("Unsupported physics flags for the checked LMU operation")
    if session.controls != expected["controls"]:
        raise ValueError("Control bindings differ from the checked LMU interface")
    if _model_signature(session.model) != expected["model_signature"]:
        raise ValueError("Physical model differs from the checked LMU assembly")
    declared = session.manifest.get("scene_sha256")
    if declared is not None and declared != expected["scene_sha256"]:
        raise ValueError("Scene metadata does not match the checked physical model")
    return expected


class _Stop(RuntimeError):
    pass


class SurfaceObserver:
    """Read a real planar coupon and check all articulated contact geometry."""

    def __init__(self, session, equipment="snom_table"):
        self.binding = _validate_binding(session, equipment)
        self.session = session
        self.model, self.data = session.model, session.data
        self.equipment = equipment
        self.controls = {name: session.controls[equipment, name] for name in CONTROLS}
        self.coupon = self.model.geom(f"{equipment}__snom_inert_coupon").id
        self.tip = self.model.geom(f"{equipment}__snom_probe_tip").id
        self.surface = self.model.site(f"{equipment}__site_snom_coupon").id
        self.apex = self.model.site(f"{equipment}__site_probe_apex").id
        if self.model.site_bodyid[self.surface] != self.model.geom_bodyid[self.coupon]:
            raise ValueError("Surface site is not attached to the physical coupon")
        if self.model.site_bodyid[self.apex] != self.model.geom_bodyid[self.tip]:
            raise ValueError("Apex site is not attached to the physical tip")
        local = self.model.site_pos[self.surface] - self.model.geom_pos[self.coupon]
        if not np.allclose(
            local, (0, 0, self.model.geom_size[self.coupon, 2]), atol=1e-12
        ):
            raise ValueError("Surface registration does not match coupon top centre")
        # Include colliding child/parent pairs explicitly; never change masks.
        moving_bodies = {
            int(self.model.jnt_bodyid[c.joint]) for c in self.controls.values()
        }
        moving = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in moving_bodies and self.model.geom_contype[i]
        ]
        instrument = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith(equipment + "__")
            and self.model.geom_contype[i]
        ]
        self.pairs = np.array(
            [
                (i, j)
                for i in moving
                for j in instrument
                if self.model.geom_bodyid[i] != self.model.geom_bodyid[j]
                and (j not in moving or i < j)
            ],
            dtype=int,
        )
        self.executed_steps = 0
        self.checked_steps = 0
        self.minimum_clearance = None

    def read(self):
        # A site is only the registration frame: the measured plane itself is
        # reconstructed from the physical coupon centre, rotation and thickness.
        rot = self.data.geom_xmat[self.coupon].reshape(3, 3)
        surface = (
            self.data.geom_xpos[self.coupon]
            + rot[:, 2] * self.model.geom_size[self.coupon, 2]
        )
        relative = rot.T @ (self.data.site_xpos[self.apex] - surface)
        return {
            "x_m": float(relative[0]),
            "y_m": float(relative[1]),
            "clearance_m": float(relative[2]),
            "over_coupon": bool(
                np.all(np.abs(relative[:2]) <= self.model.geom_size[self.coupon, :2])
            ),
            "surface_world_m": surface.tolist(),
            "probe_world_m": self.data.site_xpos[self.apex].tolist(),
        }

    def check(self):
        m, d = self.model, self.data
        if (
            int(m.opt.disableflags) != self.binding["disableflags"]
            or int(m.opt.enableflags) != self.binding["enableflags"]
            or not np.array_equal(m.geom_contype, self.binding["contype"])
            or not np.array_equal(m.geom_conaffinity, self.binding["conaffinity"])
        ):
            raise _Stop("physics_contact_configuration_changed")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise _Stop("nonfinite_physics_state")
        for c in self.controls.values():
            q = float(d.qpos[m.jnt_qposadr[c.joint]])
            if not c.lower - 1e-7 <= q <= c.upper + 1e-7:
                raise _Stop("joint_limit_violation")
        obs = self.read()
        if obs["clearance_m"] < 3e-6:
            raise _Stop("surface_clearance_violation")
        for contact in d.contact:
            if contact.dist < -1e-7:
                raise _Stop("physical_contact_penetration")
        # Vectorized exact primitive/mesh AABBs only select candidate pairs;
        # actual signed distances decide collision, including filtered parents.
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        overlap = np.minimum(center[a] + half[a], center[b] + half[b]) - np.maximum(
            center[a] - half[a], center[b] - half[b]
        )
        for i, j in self.pairs[np.all(overlap > 1e-7, axis=1)]:
            distance = mujoco.mj_geomDistance(m, d, int(i), int(j), 0.1, None)
            if distance < -1e-7:
                raise _Stop(
                    "filtered_pair_penetration:"
                    + m.geom(int(i)).name
                    + ":"
                    + m.geom(int(j)).name
                )
        self.checked_steps += 1
        gap = obs["clearance_m"]
        self.minimum_clearance = (
            gap if self.minimum_clearance is None else min(gap, self.minimum_clearance)
        )
        return obs

    def coverage(self):
        return {
            "executed_steps": self.executed_steps,
            "checked_steps": self.checked_steps,
            "coverage_complete": self.executed_steps > 0
            and self.executed_steps == self.checked_steps,
            "minimum_checked_clearance_m": self.minimum_clearance,
        }


def inspect_lmu(
    session,
    *,
    offsets_m=(-100e-6, -50e-6, 0.0, 50e-6, 100e-6),
    clearance_m=10e-6,
    equipment="snom_table",
):
    """Align, approach, scan and retract through actual bounded servo commands.

    Requests and model bindings are validated before mutation. Execution failure
    stops further legs and holds bounded commands at measured positions; it does
    not teleport, explicitly reset or claim a safe return. Caller controls episode reset.
    """
    offsets = tuple(float(v) for v in offsets_m)
    if (
        not offsets
        or len(offsets) > 101
        or not all(math.isfinite(v) and abs(v) <= 120e-6 for v in offsets)
    ):
        raise ValueError("Choose 1–101 finite scan offsets within +/-120 micrometres")
    if not math.isfinite(clearance_m) or not 8e-6 <= clearance_m <= 80e-6:
        raise ValueError("Geometric scan clearance must be 8–80 micrometres")
    observer = SurfaceObserver(session, equipment)
    m, d = session.model, session.data
    if not math.isfinite(float(d.time)):
        raise ValueError("Surface inspection requires finite simulation time")
    mujoco.mj_forward(m, d)
    start = observer.read()
    if start["clearance_m"] < 0.002:
        raise ValueError(
            "Begin from a retracted pose at least 2 mm above the sample plane"
        )
    controls = observer.controls
    q = lambda name: float(d.qpos[m.jnt_qposadr[controls[name].joint]])
    initial = {name: q(name) for name in CONTROLS}

    def measured_state():
        return {
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "ctrl": d.ctrl.tolist(),
            "time_s": float(d.time),
        }

    report = {
        "scene_sha256": observer.binding["scene_sha256"],
        "model_signature": observer.binding["model_signature"],
        "operation": "registered_geometric_line_scan",
        "passed": False,
        "science_model": "not_simulated",
        "calibration": "authored model geometry only",
        "requested_clearance_m": clearance_m,
        "requested_offsets_m": list(offsets),
        "initial_q_m": initial,
        "initial_state": measured_state(),
        "start_observation": start,
        "legs": [],
        "scan": [],
        "reset_performed": False,
        "explicit_reset_performed": False,
        "automatic_reset_detected": False,
        "time_discontinuity_detected": False,
        "physics_events": [],
        "return_attempted": False,
        "returned": False,
    }

    def move(name, target, label):
        c = controls[name]
        if not c.lower <= target <= c.upper:
            raise _Stop("unreachable_" + label)
        d.ctrl[c.actuator] = target
        before = observer.executed_steps
        steps = math.ceil(0.5 / m.opt.timestep)
        for _ in range(steps):
            before_time, before_warnings = float(d.time), _warning_state(d)
            reset_inputs = {
                key: getattr(d, key).copy()
                for key in ("ctrl", "qfrc_applied", "xfrc_applied")
            }
            mujoco.mj_step(m, d)
            observer.executed_steps += 1
            mujoco.mj_forward(m, d)
            event, _ = _physics_event(m, d, before_time, before_warnings, reset_inputs)
            if event is not None:
                report["physics_events"].append(
                    dict(phase=label, executed_step=observer.executed_steps, **event)
                )
                report.update(combine_reset_reports(report["physics_events"]))
                report["time_discontinuity_detected"] = any(
                    not e["time_continuous"] for e in report["physics_events"]
                )
                raise _Stop("native_physics_warning_or_time_discontinuity")
            observer.check()
        tolerance = 2e-8 if name.startswith("fine_") else 2e-7
        error = q(name) - target
        if abs(error) > tolerance:
            raise _Stop("tracking_failure_" + label)
        obs = observer.read()
        report["legs"].append(
            {
                "phase": label,
                "control": name,
                "target_m": target,
                "measured_m": q(name),
                "error_m": error,
                "executed_steps": observer.executed_steps - before,
                "observation": obs,
            }
        )
        return obs

    try:
        # A separate preflight check does not count as an executed-step check.
        observer.check()
        observer.checked_steps = 0
        observer.minimum_clearance = None
        move("fine_approach_z", 0.0, "centre_fine_z_while_retracted")
        move("fine_scan_x", 0.0, "centre_fine_x_while_retracted")
        obs = observer.read()
        if abs(obs["y_m"]) > 1e-7:
            raise _Stop("sample_y_registration_unreachable")
        move(
            "specimen_translation",
            q("specimen_translation") + obs["x_m"],
            "register_sample_x",
        )
        obs = observer.read()
        move(
            "inspection_head_height",
            q("inspection_head_height") + clearance_m + 93e-6 - obs["clearance_m"],
            "coarse_surface_approach",
        )
        obs = observer.read()
        move(
            "fine_approach_z",
            q("fine_approach_z") + clearance_m - obs["clearance_m"],
            "fine_surface_approach",
        )
        for offset in offsets:
            obs = observer.read()
            obs = move("fine_scan_x", q("fine_scan_x") + offset - obs["x_m"], "scan")
            if (
                not obs["over_coupon"]
                or abs(obs["x_m"] - offset) > 3e-8
                or abs(obs["clearance_m"] - clearance_m) > 3e-8
            ):
                raise _Stop("registered_scan_tolerance_failure")
            report["scan"].append(
                dict(requested_x_m=offset, state=measured_state(), **obs)
            )
        report["return_attempted"] = True
        move("fine_approach_z", 0.0, "fine_retract")
        move(
            "inspection_head_height",
            initial["inspection_head_height"],
            "coarse_retract",
        )
        move("fine_scan_x", initial["fine_scan_x"], "restore_fine_x")
        move(
            "specimen_translation", initial["specimen_translation"], "restore_sample_x"
        )
        move("fine_approach_z", initial["fine_approach_z"], "restore_fine_z")
        report["returned"] = all(
            abs(q(name) - initial[name]) < 2e-7 for name in CONTROLS
        )
        report["passed"] = report["returned"]
    except _Stop as error:
        report["failure"] = str(error)
        for c in controls.values():
            measured = float(d.qpos[m.jnt_qposadr[c.joint]])
            if math.isfinite(measured):
                d.ctrl[c.actuator] = np.clip(measured, c.lower, c.upper)
        report["bounded_measured_hold_commanded"] = True
    report.update(observer.coverage())
    report["final_q_m"] = {name: q(name) for name in CONTROLS}
    report["final_observation"] = observer.read()
    report["final_state"] = measured_state()
    return report
