"""Integrated laboratory assets, honest evidence, controls and portable exports."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from archetypes.task_catalog import CATALOG, CatalogEntry
from real_labs.builder import build_scene
from real_labs.catalog import scenes
from real_labs.export import export_scene
from real_labs.runtime import InstrumentSession
from webui.task_navigation import task_navigation


EXPECTED = {
    "rochester_genomics",
    "indiana_microfluidics",
    "uw_isolab",
    "uva_deposition",
    "purdue_phenotyping",
    "epfl_microbiorobotics",
    "waterloo_robohub",
    "penn_pathology",
    "eth_air_quality",
    "nasa_planetary",
    "harvard_cns_cambridge",
    "caltech_stoltz_schlinger",
    "cornell_schlom_mbe",
    "stanford_biomechatronics_gait",
    "anu_shrimp_geochronology",
    "oxford_bonilla_semiconductor",
    "ntu_sgsr_characterization",
    "ntu_smtc_membranes",
    "tsinghua_rush3d",
    "uq_moreton_aquarium",
}


EXPANSION = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "Hooke/university/batches/qs2027_100_labs_20261003.json"
    ).read_text()
)
EXPECTED |= set(EXPANSION["scene_ids"])


class RealLaboratorySceneTests(unittest.TestCase):
    def test_catalogue_preserves_source_and_estimate_disclosures(self):
        definitions = scenes()
        self.assertEqual(set(definitions), EXPECTED)
        for identifier, definition in definitions.items():
            with self.subTest(scene=identifier):
                self.assertTrue(definition["source_urls"])
                self.assertTrue(definition["reference_notes"])
                self.assertIn(
                    definition["layout_fidelity"],
                    {
                        "estimated_from_public_reference",
                        "workflow_based_design",
                    },
                )
                self.assertTrue(all(size > 0 for size in definition["room"]["size"]))
                equipment_ids = [item["id"] for item in definition["equipment"]]
                self.assertEqual(len(equipment_ids), len(set(equipment_ids)))
                self.assertIn(definition["task"]["primary_equipment"], equipment_ids)
                for item in definition["equipment"]:
                    self.assertTrue(item["evidence"])
                    self.assertTrue(item["role"])

    def test_all_scenes_compile_and_every_control_tracks_observed_motion(self):
        for identifier in sorted(EXPECTED):
            with self.subTest(scene=identifier):
                root, manifest = build_scene(identifier)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                for camera in ("overview", "workstation"):
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera), 0
                    )
                self.assertGreater(model.nu, 0)
                self.assertIn("not a surveyed", manifest["reconstruction_status"])
                session = InstrumentSession(model, manifest)
                self.assertEqual(len(session.controls), model.nu)
                self.assertTrue(
                    all(c.lower < c.upper for c in session.controls.values())
                )
                report = session.smoke_test()
                failures = [row for row in report["controls"] if not row["passed"]]
                self.assertTrue(report["passed"], failures)
                self.assertEqual(session.data.time, 0)

    def test_every_lab_has_distinct_navigation_without_loading_simulators(self):
        entries = [entry for entry in CATALOG.values() if entry.category == "real_labs"]
        self.assertEqual(len(entries), len(EXPECTED))
        destinations = set()
        with patch.object(
            CatalogEntry, "load_classes", side_effect=AssertionError("Loaded simulator")
        ):
            for entry in entries:
                navigation = task_navigation(entry)
                url = urlsplit(navigation["url"])
                self.assertEqual(url.path, "/real-labs")
                self.assertEqual(navigation["category"], "real_labs")
                identifier = parse_qs(url.query)["scene"][0]
                self.assertIn(identifier, EXPECTED)
                self.assertEqual(entry.name, "real_lab_" + identifier)
                self.assertEqual(
                    navigation["task_label"], scenes()[identifier]["title"]
                )
                destinations.add(identifier)
        self.assertEqual(destinations, EXPECTED)

    def test_free_specimen_carriers_settle_on_real_support_surfaces(self):
        checked = 0
        for identifier in sorted(EXPECTED):
            root, manifest = build_scene(identifier)
            if not manifest["specimens"]:
                continue
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            session = InstrumentSession(model, manifest)
            starting_heights = {
                item["body"]: float(session.data.body(item["body"]).xpos[2])
                for item in manifest["specimens"]
            }
            mujoco.mj_step(model, session.data, nstep=round(1.0 / model.opt.timestep))
            mujoco.mj_forward(model, session.data)
            for item in manifest["specimens"]:
                with self.subTest(scene=identifier, sample=item["body"]):
                    body = model.body(item["body"]).id
                    joint = model.body_jntadr[body]
                    self.assertEqual(model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
                    self.assertLess(
                        session.data.xpos[body, 2],
                        starting_heights[item["body"]] - 0.001,
                    )
                    observed = session.process_readiness(item["equipment"])
                    self.assertTrue(observed["ready"], observed)
                    process = session.equipment[item["equipment"]]["process"]
                    if "access_control" in process:
                        control = process["access_control"]
                        spec = session.controls[item["equipment"], control]
                        closed = process["closed_value"]
                        limit = max(
                            (spec.lower, spec.upper),
                            key=lambda value: abs(value - closed),
                        )
                        session.command(
                            item["equipment"], control, closed + (limit - closed) * 0.35
                        )
                        open_state = session.process_readiness(item["equipment"])
                        self.assertFalse(open_state["ready"])
                        self.assertIn("access_open", open_state["reasons"])
                    checked += 1
        self.assertEqual(checked, 7)

    def test_export_reloads_after_relocation_with_relative_resource_paths(self):
        artifact_root = Path(__file__).resolve().parents[1] / "temp"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            directory = Path(directory)
            export_scene("eth_air_quality", directory / "source", validate=False)
            shutil.copytree(directory / "source", directory / "relocated")
            xml = directory / "relocated/scene.xml"
            root = ET.parse(xml).getroot()
            for asset in root.findall("./asset/*[@file]"):
                self.assertFalse(Path(asset.get("file")).is_absolute())
                self.assertTrue((xml.parent / asset.get("file")).is_file())
            metadata = json.loads((xml.parent / "manifest.json").read_text())
            session = InstrumentSession.from_files(xml, xml.parent / "manifest.json")
            self.assertEqual(session.model.nu, metadata["model"]["actuators"])
            self.assertEqual(session.manifest["id"], "eth_air_quality")
            self.assertTrue((directory / "source/snapshot").is_dir())


if __name__ == "__main__":
    unittest.main()
