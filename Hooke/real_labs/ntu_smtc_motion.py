"""Measured cold valve positioning and retained-vial access on SMTC skids."""

import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "ntu_smtc_membranes"
EQUIPMENT = ("membrane_skid_primary", "membrane_skid_secondary")
CONTROLS = ("isolation_valve", "sample_carriage")


class SkidInspection(RigidAccess):
    """Use the original handles and carriage drives; samples stay retained."""

    def __init__(self, session, equipment=EQUIPMENT[0]):
        if equipment not in EQUIPMENT:
            raise ValueError("Unknown SMTC membrane skid")
        super().__init__(session, SCENE, equipment, CONTROLS)
        self.equipment = equipment

    def observe(self):
        self.preflight()
        m, d, equipment = self.model, self.data, self.equipment
        positions = self.motion.measured()
        vial = m.site(equipment + "__site_water_vial").id
        lever = m.site(equipment + "__site_valve_handle_tip").id
        return dict(
            equipment=equipment,
            valve_angle_rad=float(positions[0]),
            carriage_extension_m=float(positions[1]),
            vial_center_world_m=d.site_xpos[vial].tolist(),
            handle_tip_world_m=d.site_xpos[lever].tolist(),
            sample_attachment="Three rigid retained closed vials; no grasping or liquid transfer",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.any(np.abs(start) > 0.0002):
            raise PlanningError("Skid inspection requires the measured home pose")
        initial = self.observe()
        angle = float(self.motion.upper[0])
        phases, observations = [], []
        for name, target in (
            ("quarter_turn_handle", (angle, 0)),
            ("extend_retained_vials", (angle, 0.15)),
            ("retract_retained_vials", (angle, 0)),
            ("return_handle", start),
        ):
            result = self.move(target)
            if result["status"] == "succeeded":
                observation = self.observe()
                observations.append(dict(phase=name, **observation))
                if np.any(np.abs(self.motion.measured() - target) > 0.0002):
                    result.update(
                        status="failed",
                        failure="skid_not_at_target",
                        hold_requested=self.motion._request_hold(),
                    )
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
        return dict(
            status=(
                "succeeded"
                if len(phases) == 4 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            equipment=self.equipment,
            phases=phases,
            observations=observations,
            initial=initial,
            **combine_reset_reports(phases),
            measured_return_error_rad_m=(self.motion.measured() - start).tolist(),
            scope="Cold mechanical quarter-turn handle and guided retained-vial access",
            limitations=[
                "Handle angle is not measured fluid isolation; pressure, flow and filtration are not implemented",
                "Gauge needles are static exterior details, not simulated measurements",
                "Original surrogate drive gains, masses and ideal guides are not calibrated hardware",
                "Failure requests a measured-position hold without teleporting or resetting",
            ],
        )


def inspect_ntu_smtc(session):
    reports = []
    for equipment in EQUIPMENT:
        report = SkidInspection(session, equipment).run()
        reports.append(report)
        if report["status"] != "succeeded":
            break
    return dict(
        scene=SCENE,
        status=(
            "succeeded"
            if len(reports) == 2 and all(r["status"] == "succeeded" for r in reports)
            else "failed"
        ),
        scene_sha256=reports[0]["scene_sha256"],
        skids=reports,
        **combine_reset_reports(reports),
    )
