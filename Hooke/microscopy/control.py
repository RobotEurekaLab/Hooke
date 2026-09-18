"""Validated controls shared by source and native microscopy sessions."""

from microscopy import OPERATIONS, model_family


def experiment_request(task, request, default_operation="push"):
    operation = request.get("operation", task.operation if task is not None else default_operation)
    seed = request.get("seed", 0)
    if operation not in OPERATIONS or type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError("Invalid operation or seed")
    return operation, seed


def apply_request(task, request):
    command = request.get("command")
    if command == "reset":
        operation, seed = experiment_request(task, request)
        if model_family(operation) != model_family(task.operation):
            raise ValueError("This operation requires reloading the workstation model")
        task.operation = operation
        task.task = "microscopy_"+operation
        task.reset(seed)
    elif command == "step":
        seconds = float(request.get("seconds", .2))
        if not 0 < seconds <= 2:
            raise ValueError("Step duration must be from 0 to 2 seconds")
        values = request.get("actuators", {})
        if not isinstance(values, dict):
            raise ValueError("Actuators must be a name/value mapping")
        pressure = float(request.get("pressure_pa", task.data.userdata[0]))
        volume_key = "target_"+task.volume_unit.lower()
        other_key = "target_nl" if volume_key == "target_pl" else "target_pl"
        if other_key in request:
            raise ValueError("Target volume unit does not match the active experiment")
        volume = float(request.get(volume_key, task.data.userdata[1]))
        compensation = float(request.get("compensation_pa", getattr(task.mechanics, "compensation_pa", 0.)))
        clogged = request.get("needle_clogged", getattr(task.mechanics, "needle_clogged", False))
        if type(clogged) is not bool or ("needle_clogged" in request and task.volume_unit != "pL"):
            raise ValueError("Needle occlusion requires a boolean in the cell experiment")
        holding_pressure = float(request.get("holding_pressure_pa", getattr(task.mechanics, "holding_pressure_pa", 0.)))
        holding_occluded = request.get("holding_occluded", getattr(task.mechanics, "holding_occluded", False))
        if (("holding_pressure_pa" in request or "holding_occluded" in request)
                and task.operation != "suction_injection"):
            raise ValueError("Holding controls require the suspended-cell experiment")
        if not -5000 <= holding_pressure <= 0 or type(holding_occluded) is not bool:
            raise ValueError("Holding vacuum must be from -5000 to 0 Pa and occlusion a boolean")
        if not 0 <= pressure <= 20000 or not 0 <= volume <= task.maximum_target_volume or not 0 <= compensation <= 2000:
            raise ValueError("Pressure or target volume exceeds the demonstrator range")
        # Validate all axis commands before changing pressure or advancing time.
        task.validate_commands(values)
        task.data.userdata[:] = [pressure, volume]
        if hasattr(task.mechanics, "compensation_pa"):
            task.mechanics.compensation_pa = compensation
            task.mechanics.needle_clogged = clogged
        if hasattr(task.mechanics, "holding_pressure_pa"):
            task.mechanics.holding_pressure_pa = holding_pressure
            task.mechanics.holding_occluded = holding_occluded
        task.completed = False
        task.phase("manual_control")
        task.command(values, seconds)
    elif command == "autofocus":
        task.autofocus()
    elif command == "demo":
        operation, seed = experiment_request(task, request)
        if model_family(operation) != model_family(task.operation):
            raise ValueError("This operation requires reloading the workstation model")
        task.operation = operation
        task.task = "microscopy_" + operation
        task.reset(seed)
        task.execute()
    else:
        raise ValueError("Unknown workstation command")
