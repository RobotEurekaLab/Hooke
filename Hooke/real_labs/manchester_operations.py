"""Public collision-checked access to the Manchester fixed-hand workspace.

The imported joint strokes remain available to private response diagnostics.
Public clients use the measured inspection/return route because independent
in-range joint requests can drive the complete fixed hand into the table.
"""

from real_labs.manchester_motion import inspect_manchester
from real_labs.rigid_access import require_binding


SCENE = "manchester_collaborative_robotics"
OWNER = "collaborative_arm"
OPERATION = "empty_tool_inspection"


class OperationRequired(ValueError):
    """The requested joint motion needs a complete collision-checked route."""

    def __init__(self):
        self.operation = OPERATION
        self.reason = "Independent arm motion has no checked table clearance"
        super().__init__(f"{self.reason}; use the {self.operation} operation")


def validate_command(session, equipment_id, control, target):
    """Reject unplanned arm requests before applying any input or physics step."""
    if session.manifest.get("id") == SCENE and equipment_id == OWNER:
        require_binding(session, SCENE)
        raise OperationRequired()


def run_inspection(session):
    """Retain the measured route report and expose its explicit success value."""
    require_binding(session, SCENE)
    result = inspect_manchester(session)
    if not isinstance(result, dict) or result.get("status") not in ("succeeded", "failed"):
        raise RuntimeError("Inspection returned no terminal route result")
    return dict(result, success=result["status"] == "succeeded")
