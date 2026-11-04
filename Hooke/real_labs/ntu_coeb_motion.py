"""Measured coarse registration of an inert coupon below a supported probe.

This dry mechanical inspection has no near-field optical, AFM feedback or
spectroscopy model. Captures contain actual state; live movement never resets.
"""

from dataclasses import replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "ntu_coeb_nanophotonics"
EQUIPMENT = "snom_station"
CONTROLS = ("sample_x", "sample_y", "probe_clearance")


class CoebInspection(RigidAccess):
    """Bound model/state with direct enabled-pair checks after completed steps."""

    def __init__(self, session, controls=CONTROLS):
        if tuple(controls) not in (CONTROLS, CONTROLS[:2], CONTROLS[2:]):
            raise ValueError("Choose the complete inspection, XY stage, or probe axis")
        super().__init__(session, SCENE, EQUIPMENT, tuple(controls), minimum_m=0.00025)
        self.selected_controls = tuple(controls)
        self.motion.resolution[:] = 0.0005

    def move(self, target, *, step_observer=None):
        self.preflight()
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        before = self.state()
        self.reset_counts()
        try:
            self.check(self.model, self.data)
            plan = self.motion.plan_joint_target(target)
            # Keep the original modest servos and shared tracking tolerance.
            # Subdivide physical targets rather than teleporting or increasing gains.
            increment = 0.8 * 0.025 * (self.motion.upper - self.motion.lower)
            points = [plan.waypoints[0]]
            for a, b in zip(plan.waypoints, plan.waypoints[1:]):
                a, b = np.asarray(a), np.asarray(b)
                count = max(1, int(np.ceil(np.max(np.abs(b - a) / increment))))
                points.extend(
                    tuple(a + (b - a) * k / count) for k in range(1, count + 1)
                )
            points.extend([points[-1]] * 4)
            plan = replace(plan, waypoints=tuple(points))
        except (PlanningError, RuntimeError) as error:
            self.reset_counts()
            result = dict(
                status="blocked",
                failure=str(error),
                physics_steps=0,
                hold_requested=self.motion._request_hold(),
                reset_performed=False,
            )
        else:
            self.reset_counts()

            def observe(model, data):
                self.check(model, data)
                if step_observer is not None:
                    step_observer(model, data)

            result = self.motion.execute(plan, step_observer=observe)
        steps = result["physics_steps"]
        complete = steps > 0 and steps == self.checked
        result.update(
            before=before,
            after=self.state(),
            direct_clearance=dict(
                executed_steps=steps,
                observer_entries=self.entries,
                checked_steps=self.checked,
                unchecked_steps=steps - self.checked,
                coverage_complete=complete,
                pair_count=len(self.pairs),
                direct_near_queries=self.queries,
                clearance_cap_m=self.cap_m,
                checked_steps_clearance_lower_bound_m=(
                    self.minimum if self.checked else None
                ),
                whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
            ),
        )
        return result

    def observe_registration(self):
        self.preflight()
        interface = self.session.equipment[EQUIPMENT]["inspection_interface"]
        m, d = self.model, self.data
        tip = d.site_xpos[m.site(interface["tip_site"]).id].copy()
        target = d.site_xpos[m.site(interface["target_site"]).id].copy()
        error = float(np.linalg.norm((tip - target)[:2]))
        gap = float(tip[2] - target[2])
        hit = np.array([-1], dtype=np.int32)
        distance = float(
            mujoco.mj_ray(
                m,
                d,
                tip - [0, 0, 0.000001],
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                hit,
            )
        )
        name = m.geom(int(hit[0])).name if hit[0] >= 0 else None
        return dict(
            tip_world_m=tip.tolist(),
            target_world_m=target.tolist(),
            projected_error_m=error,
            standoff_m=gap,
            first_hit=name,
            first_hit_distance_m=distance,
            registered=bool(
                error < 0.00002
                and 0.0015 < gap < 0.0025
                and name == interface["target_geom"]
            ),
            actual_state=self.state(),
            scope="Noncontact registration of an authored blunt fibre over an inert marker; no optical result",
        )

    def run(self, *, step_observer=None):
        self.preflight()
        if self.selected_controls != CONTROLS:
            raise ValueError("The full registration sequence requires all three axes")
        start = self.motion.measured().copy()
        if np.max(np.abs(start)) > 0.00002:
            raise PlanningError("Start from the measured authored home pose")
        initial = self.state()
        interface = self.session.equipment[EQUIPMENT]["inspection_interface"]
        tip = self.data.site_xpos[self.model.site(interface["tip_site"]).id]
        mark = self.data.site_xpos[self.model.site(interface["target_site"]).id]
        alignment = [
            float(
                start[i]
                + np.dot(
                    tip - mark,
                    self.data.xaxis[self.session.controls[EQUIPMENT, control].joint],
                )
            )
            for i, control in enumerate(CONTROLS[:2])
        ]
        target = (*alignment, -0.009)
        phases, observation = [], None
        for name, q in (
            ("raise_probe", (0, 0, 0.010)),
            ("register_retained_coupon", (*alignment, 0.010)),
            ("inspect_registration", target),
            ("retract_probe", (*alignment, 0.010)),
            ("return_coupon", (0, 0, 0.010)),
            ("return_probe", start),
        ):
            result = self.move(q, step_observer=step_observer)
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
            if name == "inspect_registration":
                observation = self.observe_registration()
                if not observation["registered"]:
                    phases[-1].update(
                        status="failed",
                        failure="registration_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
                    break
        error = np.abs(self.motion.measured() - start)
        success = len(phases) == 6 and all(p["status"] == "succeeded" for p in phases)
        if success and np.max(error) >= 0.00002:
            success = False
            phases[-1].update(
                status="failed",
                failure="measured_return_error",
                hold_requested=self.motion._request_hold(),
            )
        return dict(
            status="succeeded" if success else "failed",
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_state=initial,
            derived_alignment_target_m=list(target),
            phases=phases,
            observation=observation,
            final_state=self.state(),
            executed_steps=sum(p["physics_steps"] for p in phases),
            checked_steps=sum(p["direct_clearance"]["checked_steps"] for p in phases),
            measured_return_error_m=error.tolist(),
            **combine_reset_reports(phases),
            limitations=[
                "Mechanically retained 12 mm inert coupon; no loading, release or deformable attachment",
                "Noncontact millimetre standoff; no near-field light, AFM feedback, optical image or spectroscopy",
                "Estimated instrument dimensions, blunt fibre geometry and hidden guides; no exact OEM reconstruction",
                "Ideal joints and gravity compensation retain original authored gains/masses; no calibrated precision",
                "Finite planning and completed-step checks are not continuous-time collision certification",
                "One exclusive session owner; callbacks must be read-only and concurrent state writers are unsupported",
                "Failure requests a servo hold without resetting; it is not an instantaneous certified stop",
            ],
        )


def inspect_coeb_registration(session):
    return CoebInspection(session).run()
