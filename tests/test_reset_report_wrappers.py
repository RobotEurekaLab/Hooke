"""Public aggregate reports preserve real native resets and uncertain clocks."""

import importlib
import unittest
import xml.etree.ElementTree as ET

import mujoco

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


# Each entry exercises an actual public task, including both levels of the
# two-skid report and both devices accepted by the NTU characterization task.
CASES = (
    ("alberta_coater_motion", "CoaterAccess", "run", None),
    # Boston revision 3 has its own source-bound executor and actual aggregate
    # native-reset/clock regression in test_boston_sem_access.py.
    ("caltech_motion", "HoodAccess", "open_inspect_return", None),
    ("cambridge_motion", "CambridgeAccess", "run", None),
    ("cornell_loading", "CornellLoading", "run", None),
    ("harvard_motion", "WaferInspection", "inspect_and_return", None),
    ("heidelberg_alignment_motion", "AlignmentInspection", "run", None),
    ("jhu_furnace_motion", "FurnaceInspection", "run", None),
    ("ntu_coeb_motion", "CoebInspection", "run", None),
    ("ntu_sgsr_motion", "PrecisionInspection", "inspect_and_return", "dimension_icon"),
    (
        "ntu_sgsr_motion",
        "PrecisionInspection",
        "inspect_and_return",
        "mechanical_tester",
    ),
    ("ntu_smtc_motion", "SkidInspection", "inspect_ntu_smtc", None),
    ("purdue_motion", "PurdueInspection", "run", None),
    ("sydney_tank_motion", "TankInspection", "run", None),
    ("ucla_probe_motion", "ProbeAccess", "run", None),
    ("uq_aquarium_motion", "AquariumAccess", "run", None),
    ("yonsei_probe_motion", "ProbeInspection", "inspect_and_return", None),
    ("wire_alignment", "WireAlignment", "run", None),
)


def load_case(case):
    module_name, class_name, method, equipment = case
    module = importlib.import_module("real_labs." + module_name)
    scene = getattr(
        module,
        "SCENE",
        (
            "birmingham_advanced_micro_edm"
            if module_name == "wire_alignment"
            else "caltech_stoltz_schlinger"
        ),
    )
    root, manifest = build_scene(scene)
    session = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    cls = getattr(module, class_name)
    operation = cls(session, equipment) if equipment else cls(session)
    run = lambda: (
        module.inspect_ntu_smtc(session)
        if method == "inspect_ntu_smtc"
        else getattr(operation, method)()
    )
    return session, operation, run


def first_phase(report):
    if "skids" in report:
        report = report["skids"][0]
    return report.get("phases", report.get("legs"))[0]


class ResetReportWrapperTests(unittest.TestCase):
    def check_failure(self, case, *, native_reset):
        session, operation, run = load_case(case)
        session.data.time = 1.0 if native_reset else 1e30
        if native_reset:
            # This finite external load induces a real MuJoCo BADQACC reset;
            # no step mocking, state teleportation or altered physics options.
            session.data.qfrc_applied[operation.motion.dadr[0]] = 1e25
        report = run()
        phase = first_phase(report)
        expected = True if native_reset else None
        self.assertEqual(report["status"], "failed")
        self.assertEqual(phase["status"], "failed")
        self.assertEqual(phase["physics_steps"], 1)
        self.assertEqual(phase["checked_steps"], 0)
        self.assertFalse(phase["coverage_complete"])
        self.assertTrue(phase["hold_requested"])
        self.assertIs(report["reset_performed"], expected)
        self.assertIs(report["reset_performed"], phase["reset_performed"])
        self.assertFalse(report["explicit_reset_performed"])
        self.assertEqual(report["automatic_reset_detected"], native_reset)
        if "skids" in report:
            self.assertIs(report["skids"][0]["reset_performed"], expected)
        if native_reset:
            self.assertEqual(session.data.time, session.model.opt.timestep)
            self.assertGreater(
                session.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number, 0
            )
        else:
            self.assertEqual(session.data.time, 1e30)

    def test_public_tasks_propagate_actual_native_resets(self):
        for case in CASES:
            with self.subTest(case=case):
                self.check_failure(case, native_reset=True)

    def test_public_tasks_preserve_ambiguous_clock_status(self):
        for case in CASES:
            with self.subTest(case=case):
                self.check_failure(case, native_reset=False)


if __name__ == "__main__":
    unittest.main()
