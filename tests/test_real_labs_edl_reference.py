"""The photo-informed EDL assembly keeps supported parts and sample access."""

from pathlib import Path
from itertools import combinations
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.usability import camera_target_screening


class EdlReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("cmu_edl_electrochemical_testing")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def geom(self, key):
        return self.model.geom("electrochemical_test__edl_" + key).id

    def distance(self, first, second):
        return mujoco.mj_geomDistance(
            self.model, self.data, self.geom(first), self.geom(second), 1.0, None
        )

    def test_cases_and_service_hardware_have_continuous_physical_supports(self):
        contacts = [
            ("service_foot", "service_bracket"),
            ("service_bracket", "service_shelf"),
            ("service_shelf", "service_canister"),
            ("service_canister", "service_top_port"),
            ("blue_base", "blue_case_0"),
            ("blue_case_0", "blue_case_1"),
        ]
        for cabinet in ("tester", "pressure"):
            contacts.extend(
                (cabinet + "_cabinet", f"{cabinet}_foot_{i}_{j}")
                for i in range(2)
                for j in range(2)
            )
        for level, support in ((0, "tester_cabinet"), (1, "stack_0_case")):
            for foot in range(2):
                contacts.extend(
                    (
                        (support, f"stack_{level}_foot_{foot}"),
                        (f"stack_{level}_foot_{foot}", f"stack_{level}_case"),
                    )
                )
        for first, second in contacts:
            with self.subTest(first=first, second=second):
                ids = [self.geom(first), self.geom(second)]
                self.assertTrue(np.all(self.model.geom_contype[ids] > 0))
                self.assertTrue(np.all(self.model.geom_conaffinity[ids] > 0))
                self.assertLessEqual(self.distance(first, second), 1e-6)

    def test_service_leads_have_attached_and_supported_parked_ends(self):
        table_height = self.data.xpos[self.model.body("electrochemical_test").id, 2]
        for route, port in (
            ("white_left", "hose_port_0"),
            ("white_right", "hose_port_1"),
            ("red", "red_socket"),
            ("black", "black_socket"),
        ):
            with self.subTest(route=route):
                self.assertLessEqual(self.distance(route + "_route_0", port), 1e-6)
                connector = route + "_parked_connector"
                self.assertLessEqual(
                    self.distance(route + "_route_23", connector), 1e-6
                )
                geom = self.geom(connector)
                rotation = self.data.geom_xmat[geom].reshape(3, 3)
                bottom = (
                    self.data.geom_xpos[geom, 2]
                    - (np.abs(rotation) @ self.model.geom_aabb[geom, 3:])[2]
                )
                self.assertAlmostEqual(bottom, table_height, places=6)
        for route, port in (
            ("service_route_0", "service_lower_port"),
            ("service_route_23", "service_top_port"),
        ):
            self.assertLessEqual(self.distance(route, port), 1e-6)
        # All appearance-only curves remain above the table and outside the
        # drawer's complete swept footprint, including the connecting capsules.
        for i in range(self.model.ngeom):
            if (
                "__edl_" not in self.model.geom(i).name
                or "_route_" not in self.model.geom(i).name
            ):
                continue
            rotation = self.data.geom_xmat[i].reshape(3, 3)
            extent = np.abs(rotation) @ self.model.geom_aabb[i, 3:]
            local = (
                self.data.geom_xpos[i]
                - self.data.xpos[self.model.body("electrochemical_test").id]
            )
            self.assertGreaterEqual(local[2] - extent[2], -1e-6)
            overlaps_drawer_xy = (
                local[0] + extent[0] > -0.27
                and local[0] - extent[0] < 0.17
                and local[1] + extent[1] > -0.67
                and local[1] - extent[1] < -0.26
            )
            self.assertFalse(overlaps_drawer_xy, self.model.geom(i).name)

    def test_blue_electronics_no_longer_intersect_pressure_cabinet(self):
        for level in range(2):
            self.assertGreater(
                self.distance("pressure_cabinet", f"blue_case_{level}"), 0.010
            )

    def test_fixed_leads_do_not_pass_through_each_other(self):
        # These curves are visual-only, so a dynamic contact audit cannot catch
        # a crossing. Query the actual capsule surfaces independent of masks.
        for first, second in combinations(
            ("white_left", "white_right", "red", "black"), 2
        ):
            closest = min(
                self.distance(f"{first}_route_{i}", f"{second}_route_{j}")
                for i in range(24)
                for j in range(24)
            )
            self.assertGreater(closest, 0.0005, (first, second, closest))

    def test_workstation_camera_keeps_sample_line_of_sight(self):
        rows = camera_target_screening(self.model, self.data, self.manifest)
        row = next(item for item in rows if item["equipment"] == "electrochemical_test")
        camera = next(
            item for item in row["cameras"] if item["camera"] == "workstation"
        )
        self.assertTrue(camera["in_frame"])
        self.assertTrue(camera["opaque_line_of_sight"])
        self.assertGreater(camera["estimated_target_diameter_px"], 10)


if __name__ == "__main__":
    unittest.main()
