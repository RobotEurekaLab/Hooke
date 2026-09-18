"""Motorized microscopy workstations and demonstrator experiments."""

OPERATIONS = ("push", "pick_place", "injection", "cell_injection", "suction_injection")
BACKENDS = ("mujoco", "isaac")


def model_family(operation):
    if operation not in OPERATIONS:
        raise ValueError("Unknown microscopy experiment")
    return operation
