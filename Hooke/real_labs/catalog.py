"""Read-only scene definitions; importing the catalogue never starts a simulator."""

import copy
import json
from pathlib import Path


def _merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def scenes():
    path = Path(__file__).with_name("catalog.json")
    rows = json.loads(path.read_text())["scenes"]
    for source in sorted(Path(__file__).with_name("scenes").glob("*.json")):
        rows.extend(json.loads(source.read_text())["scenes"])
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reference laboratory identifiers")
    result = {row["id"]: row for row in rows}
    patched = set()
    for path in sorted(Path(__file__).with_name("layouts").glob("*.json")):
        for identifier, patch in json.loads(path.read_text()).items():
            if identifier not in result or identifier in patched:
                raise ValueError(f"Unknown or duplicate layout patch: {identifier}")
            patched.add(identifier)
            patch = copy.deepcopy(patch)
            positions = patch.pop("equipment_positions", {})
            definition = result[identifier]
            _merge(definition, patch)
            equipment = {item["id"]: item for item in definition["equipment"]}
            for name, placement in positions.items():
                if name not in equipment:
                    raise ValueError(f"Unknown equipment in {identifier}: {name}")
                _merge(equipment[name], placement)
            definition["layout_revision"] = 2
    for path in sorted(Path(__file__).with_name("evidence").glob("*.json")):
        for identifier, evidence in json.loads(path.read_text()).items():
            if identifier not in result or not isinstance(evidence, dict):
                raise ValueError(
                    f"Invalid reference-detail evidence in {path.name}: {identifier}"
                )
            definition = result[identifier]
            definition.setdefault("reference_detail_evidence", {})[path.stem] = evidence
            definition["detail_revision"] = 3
    return result


def scene(identifier):
    try:
        return copy.deepcopy(scenes()[identifier])
    except KeyError:
        raise ValueError(f"Unknown reference laboratory: {identifier}") from None
