"""Spatial errors must fail independently of an instrument's motion checks."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.catalog import scene, scenes
from real_labs.builder import build_scene
from real_labs.layout_audit import (
    audit_layout,
    model_obstacles,
    doorway_obstacle_warnings,
)
from real_labs.runtime import InstrumentSession


class LayoutChecks(unittest.TestCase):
    def test_compiled_fixture_bounds_leave_authored_entries_clear(self):
        for identifier in scenes():
            with self.subTest(scene=identifier):
                root, manifest = build_scene(identifier)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                obstacles = model_obstacles(model, data)
                self.assertTrue(obstacles)
                self.assertEqual(doorway_obstacle_warnings(manifest, obstacles), [])

    def test_projection_accounts_for_rotation_and_rejects_invalid_pose(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><compiler angle="radian"/><worldbody><geom name="obstacle" type="box" size="1 .5 .4" pos="1 2 .5" euler="0 0 .7853981633974483"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        obstacle = model_obstacles(model, data)[0]
        extent = 1.5 / np.sqrt(2)
        np.testing.assert_allclose(obstacle["minimum_xy"], np.array([1, 2]) - extent)
        np.testing.assert_allclose(obstacle["maximum_xy"], np.array([1, 2]) + extent)
        data.geom_xpos[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model_obstacles(model, data)

    def test_authored_layouts_pass_with_unverified_support_exposed(self):
        for definition in scenes().values():
            with self.subTest(scene=definition["id"]):
                report = audit_layout(definition)
                self.assertTrue(report["passed"], report["checks"])
                self.assertIn("not measured-source accuracy", report["scope"])
                self.assertTrue(definition["layout_rationale"])
        nasa = audit_layout(scene("nasa_planetary"))
        self.assertFalse(nasa["all_equipment_origins_classified"])
        self.assertEqual(
            [row["equipment"] for row in nasa["manual_inspection"]], ["rover"]
        )

    def test_rotated_bench_outside_room_is_rejected(self):
        definition = scene("penn_pathology")
        definition["benches"][0]["pos"][0] = -definition["room"]["size"][0] / 2
        report = audit_layout(definition)
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                c["check"] == "bench_inside_room" and not c["passed"]
                for c in report["checks"]
            )
        )

    def test_overlapping_benches_and_obstructed_door_are_rejected(self):
        definition = scene("rochester_genomics")
        definition["benches"][1]["pos"] = definition["benches"][0]["pos"].copy()
        self.assertFalse(audit_layout(definition)["passed"])
        definition = scene("rochester_genomics")
        door = next(
            o
            for o in definition["room"]["architecture"]["openings"]
            if o["kind"] == "door"
        )
        definition["benches"].append(
            dict(
                id="obstruction",
                pos=[door["offset"], definition["room"]["size"][1] / 2 - 0.4, 0],
                size=[0.5, 0.5, 0.9],
            )
        )
        report = audit_layout(definition)
        self.assertTrue(
            any(
                c["check"] == "doorway_approach_clear_of_benches" and not c["passed"]
                for c in report["checks"]
            )
        )

    def test_unsupported_instrument_cannot_be_reported_as_supported(self):
        definition = scene("rochester_genomics")
        definition["equipment"][0]["pos"] = [0, 0, 0.92]
        report = audit_layout(definition)
        self.assertIn(
            "liquid_handler", [r["equipment"] for r in report["manual_inspection"]]
        )

    def test_rotated_balance_door_clears_bench_across_travel(self):
        root, manifest = build_scene("eth_air_quality")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        for target in (-0.08, -0.02):
            result = session.command("analytical_balance", "draft_door_joint", target)
            self.assertLess(abs(result["error"]), 0.0015, result)


if __name__ == "__main__":
    unittest.main()
