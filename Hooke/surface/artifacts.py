"""Default local output locations shared by CLI tools and the web gallery."""

from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[2] / "temp/surface_missions"
SCENES = OUTPUT / "scenes"
MEDIA = OUTPUT / "media"
SUMMARY = OUTPUT / "summary.json"
TEAM_SUMMARY = OUTPUT / "team-summary.json"
GAIT_ROOT = OUTPUT / "gait" / "unitree-g1"
GAIT_POLICY = GAIT_ROOT / "policy.npz"
