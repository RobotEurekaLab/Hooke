"""Hooke/integrations/bionemo.py builds the documented GenMol wire protocol
and parses it back correctly; no live NIM is contacted."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from integrations.bionemo import (
    BioNeMoConfig,
    BioNeMoRemoteError,
    BioNeMoTransportError,
    generate_molecules,
)


def _opened(json_body):
    response = MagicMock()
    response.status = 200
    response.read.return_value = json.dumps(json_body).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


# Taken verbatim from https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html
GENMOL_EXAMPLE_REQUEST_SCAFFOLD = "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1O.[*{15-15}]"
GENMOL_EXAMPLE_RESPONSE = {
    "status": "success",
    "molecules": [
        {"smiles": "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1OC(C)=O", "score": 0.82},
        {"smiles": "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1OCC", "score": 0.77},
    ],
}


class GenerateMolecules(unittest.TestCase):
    @patch("integrations.bionemo.urllib.request.urlopen")
    def test_posts_documented_payload_and_parses_response(self, mock_urlopen):
        mock_urlopen.return_value = _opened(GENMOL_EXAMPLE_RESPONSE)
        config = BioNeMoConfig(genmol_url="http://genmol:8000")

        result = generate_molecules(
            config, GENMOL_EXAMPLE_REQUEST_SCAFFOLD, num_molecules=2, temperature=1.0, noise=1.0, scoring="QED"
        )

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://genmol:8000/generate")
        sent = json.loads(request.data)
        self.assertEqual(
            sent,
            {
                "smiles": GENMOL_EXAMPLE_REQUEST_SCAFFOLD,
                "num_molecules": 2,
                "temperature": 1.0,
                "noise": 1.0,
                "scoring": "QED",
            },
        )
        self.assertNotIn("Authorization", request.headers)

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.molecules), 2)
        self.assertEqual(result.molecules[0].smiles, "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1OC(C)=O")
        self.assertEqual(result.molecules[0].score, 0.82)

    @patch("integrations.bionemo.urllib.request.urlopen")
    def test_attaches_bearer_auth_when_api_key_set(self, mock_urlopen):
        mock_urlopen.return_value = _opened(GENMOL_EXAMPLE_RESPONSE)
        config = BioNeMoConfig(genmol_url="http://genmol:8000", api_key="nvapi-xyz")
        generate_molecules(config, GENMOL_EXAMPLE_REQUEST_SCAFFOLD)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer nvapi-xyz")

    @patch("integrations.bionemo.urllib.request.urlopen")
    def test_raises_remote_error_on_failure_status(self, mock_urlopen):
        mock_urlopen.return_value = _opened({"status": "failed", "error": "invalid SMILES"})
        with self.assertRaises(BioNeMoRemoteError):
            generate_molecules(BioNeMoConfig(), "not-a-real-smiles")

    @patch("integrations.bionemo.urllib.request.urlopen")
    def test_raises_remote_error_on_non_200_http_status(self, mock_urlopen):
        error = urllib.error.HTTPError("http://genmol:8000/generate", 500, "Internal Error", {}, MagicMock())
        error.read = MagicMock(return_value=b"internal error")
        mock_urlopen.side_effect = error
        with self.assertRaises(BioNeMoRemoteError):
            generate_molecules(BioNeMoConfig(), GENMOL_EXAMPLE_REQUEST_SCAFFOLD)

    @patch("integrations.bionemo.urllib.request.urlopen")
    def test_raises_transport_error_on_network_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(BioNeMoTransportError):
            generate_molecules(BioNeMoConfig(), GENMOL_EXAMPLE_REQUEST_SCAFFOLD)


class BioNeMoConfigValidation(unittest.TestCase):
    def test_rejects_non_positive_timeout(self):
        with self.assertRaises(ValueError):
            BioNeMoConfig(timeout_s=-1)

    def test_rejects_non_http_url(self):
        with self.assertRaises(ValueError):
            BioNeMoConfig(genmol_url="not-a-url")

    def test_strips_trailing_slash_from_explicit_url(self):
        config = BioNeMoConfig(genmol_url="http://genmol:8000/")
        self.assertEqual(config.genmol_url, "http://genmol:8000")

    def test_falls_back_to_shared_nvidia_api_key(self):
        import os

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "shared"}, clear=False):
            os.environ.pop("BIONEMO_API_KEY", None)
            config = BioNeMoConfig()
        self.assertEqual(config.api_key, "shared")


if __name__ == "__main__":
    unittest.main()
