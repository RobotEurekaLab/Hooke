"""Hooke/integrations/alchemi.py builds the documented ALCHEMI wire protocol
and parses it back correctly; no live NIM is contacted."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from integrations.alchemi import (
    AlchemiConfig,
    AlchemiRemoteError,
    AlchemiTransportError,
    AtomicStructure,
    MDConfig,
    health,
    relax_geometry,
    run_molecular_dynamics,
)


def _opened(json_body):
    """A context-manager object matching what urllib.request.urlopen returns on success."""
    response = MagicMock()
    response.status = 200
    response.read.return_value = json.dumps(json_body).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class AtomicStructureValidation(unittest.TestCase):
    def test_coord_length_must_match_three_times_atom_count(self):
        with self.assertRaises(ValueError):
            AtomicStructure(numbers=[1, 1], coord=[0.0, 0.0, 0.0])

    def test_cell_must_have_nine_entries(self):
        with self.assertRaises(ValueError):
            AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0], cell=[1.0, 0.0, 0.0])

    def test_active_mask_must_have_one_entry_per_atom(self):
        with self.assertRaises(ValueError):
            AtomicStructure(numbers=[1, 1], coord=[0.0] * 6, active_mask=[True])

    def test_to_payload_omits_unset_optional_fields(self):
        structure = AtomicStructure(numbers=[1, 1], coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.74])
        payload = structure.to_payload()
        self.assertNotIn("cell", payload)
        self.assertNotIn("mass", payload)
        self.assertNotIn("velocity", payload)
        self.assertNotIn("active_mask", payload)
        self.assertNotIn("structure_id", payload)
        self.assertEqual(payload["numbers"], [1, 1])

    def test_to_payload_includes_set_optional_fields(self):
        structure = AtomicStructure(
            numbers=[1, 1],
            coord=[0.0, 0.0, 0.0, 0.0, 0.0, 0.74],
            cell=[10, 0, 0, 0, 10, 0, 0, 0, 10],
            structure_id="h2",
        )
        payload = structure.to_payload()
        self.assertEqual(payload["cell"], [10, 0, 0, 0, 10, 0, 0, 0, 10])
        self.assertEqual(payload["structure_id"], "h2")


class AlchemiConfigValidation(unittest.TestCase):
    def test_rejects_non_http_url(self):
        with self.assertRaises(ValueError):
            AlchemiConfig(bgr_url="not-a-url")

    def test_rejects_non_http_bmd_url(self):
        with self.assertRaises(ValueError):
            AlchemiConfig(bmd_url="localhost:8001")

    def test_strips_trailing_slash_from_explicit_url(self):
        config = AlchemiConfig(bgr_url="http://bgr:8000/")
        self.assertEqual(config.bgr_url, "http://bgr:8000")

    def test_rejects_non_positive_timeout(self):
        with self.assertRaises(ValueError):
            AlchemiConfig(timeout_s=0)

    def test_reads_env_vars(self):
        env = {
            "ALCHEMI_BGR_URL": "http://bgr.internal:9000",
            "ALCHEMI_BMD_URL": "http://bmd.internal:9001",
            "ALCHEMI_API_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=False):
            config = AlchemiConfig()
        self.assertEqual(config.bgr_url, "http://bgr.internal:9000")
        self.assertEqual(config.bmd_url, "http://bmd.internal:9001")
        self.assertEqual(config.api_key, "secret")

    def test_falls_back_to_shared_nvidia_api_key(self):
        import os

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "shared"}, clear=False):
            os.environ.pop("ALCHEMI_API_KEY", None)
            config = AlchemiConfig()
        self.assertEqual(config.api_key, "shared")


# Taken verbatim from https://docs.nvidia.com/nim/alchemi/alchemi-bgr/api-reference
BGR_EXAMPLE_RESPONSE = {
    "atoms": [
        {
            "coord": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "numbers": [1, 1],
            "cell": [10, 0, 0, 0, 10, 0, 0, 0, 10],
            "pbc": [True, True, True],
            "energy": -1.5,
            "forces": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "stress": [0.0],
            "charges": [0.0, 0.0],
            "converged": True,
            "optimizer_nsteps": 25,
            "structure_id": "string",
        }
    ],
    "status": "Success",
    "info": "string",
}


class RelaxGeometry(unittest.TestCase):
    def test_rejects_empty_structures(self):
        with self.assertRaises(ValueError):
            relax_geometry(AlchemiConfig(), [])

    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_posts_batched_atoms_and_parses_response(self, mock_urlopen):
        mock_urlopen.return_value = _opened(BGR_EXAMPLE_RESPONSE)
        structure = AtomicStructure(numbers=[1, 1], coord=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], structure_id="H2")
        config = AlchemiConfig(bgr_url="http://bgr:8000")

        result = relax_geometry(config, [structure], force_tolerance=0.002, optimize_cell=True)

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://bgr:8000/v1/infer")
        self.assertEqual(request.get_method(), "POST")
        sent = json.loads(request.data)
        self.assertEqual(sent["atoms"], [structure.to_payload()])
        self.assertEqual(sent["opttol"], 0.002)
        self.assertTrue(sent["cellopt"])
        self.assertNotIn("Authorization", request.headers)

        self.assertEqual(result.status, "Success")
        self.assertEqual(len(result.structures), 1)
        relaxed = result.structures[0]
        self.assertTrue(relaxed.converged)
        self.assertEqual(relaxed.optimizer_nsteps, 25)
        self.assertEqual(relaxed.energy, -1.5)
        self.assertEqual(relaxed.structure_id, "string")

    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_attaches_bearer_auth_when_api_key_set(self, mock_urlopen):
        mock_urlopen.return_value = _opened(BGR_EXAMPLE_RESPONSE)
        config = AlchemiConfig(bgr_url="http://bgr:8000", api_key="nvapi-xyz")
        relax_geometry(config, [AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0])])
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer nvapi-xyz")

    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_raises_remote_error_on_non_success_status(self, mock_urlopen):
        mock_urlopen.return_value = _opened({**BGR_EXAMPLE_RESPONSE, "status": "Failed"})
        with self.assertRaises(AlchemiRemoteError):
            relax_geometry(AlchemiConfig(), [AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0])])

    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_raises_transport_error_on_network_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(AlchemiTransportError):
            relax_geometry(AlchemiConfig(), [AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0])])

    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_raises_remote_error_on_non_200_http_status(self, mock_urlopen):
        error = urllib.error.HTTPError(
            "http://bgr:8000/v1/infer", 503, "Service Unavailable", {}, MagicMock()
        )
        error.read = MagicMock(return_value=b"backends_alive=0")
        mock_urlopen.side_effect = error
        with self.assertRaises(AlchemiRemoteError):
            relax_geometry(AlchemiConfig(), [AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0])])


# Taken verbatim (shape) from https://docs.nvidia.com/nim/alchemi/alchemi-bmd/api-reference
BMD_EXAMPLE_RESPONSE = {
    "trajectory": [
        {
            "coord": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "energy": -0.9,
            "mass": [1.0],
            "cell": None,
            "stress": None,
            "charges": [0.0],
            "istep": 0,
            "md_time": 0.0,
        },
        {
            "coord": [0.0, 0.0, 0.1],
            "velocity": [0.0, 0.0, 0.05],
            "energy": -0.92,
            "mass": [1.0],
            "cell": None,
            "stress": None,
            "charges": [0.0],
            "istep": 100,
            "md_time": 0.1,
        },
    ],
    "config": {},
    "status": "Success",
    "info": "string",
}


class RunMolecularDynamics(unittest.TestCase):
    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_posts_single_atoms_object_not_a_list(self, mock_urlopen):
        mock_urlopen.return_value = _opened(BMD_EXAMPLE_RESPONSE)
        structure = AtomicStructure(numbers=[1], coord=[0.0, 0.0, 0.0])
        config = AlchemiConfig(bmd_url="http://bmd:8001")

        result = run_molecular_dynamics(config, structure, MDConfig(temperature_k=298.15))

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://bmd:8001/v1/infer")
        sent = json.loads(request.data)
        self.assertIsInstance(sent["atoms"], dict)
        self.assertEqual(sent["atoms"], structure.to_payload())
        self.assertEqual(sent["config"]["temperature"], 298.15)
        self.assertTrue(sent["config"]["nvt"])

        self.assertEqual(len(result.trajectory), 2)
        self.assertEqual(result.final_energy, -0.92)
        self.assertEqual(result.trajectory[-1].md_time, 0.1)

    def test_final_energy_rejects_empty_trajectory(self):
        from integrations.alchemi import BMDResult

        with self.assertRaises(ValueError):
            BMDResult(trajectory=[], status="Success").final_energy

    def test_md_config_rejects_unknown_ensemble(self):
        with self.assertRaises(ValueError):
            MDConfig(ensemble="plasma")


class Health(unittest.TestCase):
    @patch("integrations.alchemi.urllib.request.urlopen")
    def test_parses_ready_response(self, mock_urlopen):
        mock_urlopen.return_value = _opened({"status": "ready", "backends_alive": 2, "backends_total": 2})
        result = health(AlchemiConfig(bgr_url="http://bgr:8000"), "bgr")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://bgr:8000/v1/health/ready")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(result["status"], "ready")

    def test_rejects_unknown_service(self):
        with self.assertRaises(ValueError):
            health(AlchemiConfig(), "bgrd")


if __name__ == "__main__":
    unittest.main()
