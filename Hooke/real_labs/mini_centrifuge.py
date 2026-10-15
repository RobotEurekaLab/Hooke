"""Copy-only mechanical adaptation of the existing miniature centrifuge asset.

The final local convex resources retain loading bores and the open hinge. The
upstream instrument files, joint frames, inertia and existing drive contracts
remain intact. These are mechanical inspection assets, not separation physics.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from real_labs.reuse_drives import MINI_CENTRIFUGE


RESOURCE_DIR = Path(__file__).with_name("resources") / "mini_centrifuge"
SCENES = ("rochester_genomics", "indiana_microfluidics", "penn_pathology")
CCD_TOLERANCE = 1e-10


def _numbers(values):
    return " ".join(format(float(value), ".13g") for value in np.asarray(values).flat)


@lru_cache(maxsize=1)
def _resources():
    metadata = json.loads((RESOURCE_DIR / "provenance.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported miniature centrifuge resource schema")
    for filename in ("contacts.npz", "lid_visual.obj"):
        actual = hashlib.sha256((RESOURCE_DIR / filename).read_bytes()).hexdigest()
        if actual != metadata["resource_sha256"][filename]:
            raise ValueError(f"Miniature centrifuge resource hash changed: {filename}")
    with np.load(RESOURCE_DIR / "contacts.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"names", "bodies", "vertices", "offsets"}:
            raise ValueError("Unexpected miniature centrifuge contact arrays")
        names, bodies, vertices, offsets = (
            archive[name].copy() for name in ("names", "bodies", "vertices", "offsets")
        )
    count = metadata["contact_count"]
    if (
        names.shape != (count,)
        or names.dtype.kind != "U"
        or len(set(names)) != count
        or any(not name or not str(name).replace("_", "").isalnum() for name in names)
        or bodies.shape != (count,)
        or bodies.dtype.kind not in "iu"
        or np.any(bodies < 0)
        or np.any(bodies > 2)
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.dtype.kind != "f"
        or not np.isfinite(vertices).all()
        or offsets.shape != (count + 1,)
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or offsets[-1] != len(vertices)
        or np.any(np.diff(offsets) < 4)
    ):
        raise ValueError("Invalid miniature centrifuge contact geometry")
    expected_counts = metadata["body_contact_counts"]
    if [int(np.count_nonzero(bodies == i)) for i in range(3)] != [
        expected_counts[name] for name in ("body", "lid", "rotor")
    ]:
        raise ValueError("Miniature centrifuge contact ownership changed")
    for array in (names, bodies, vertices, offsets):
        array.flags.writeable = False
    return metadata, names, bodies, vertices, offsets


def _verify_upstream(metadata):
    repository = Path(__file__).resolve().parents[2]
    for name, expected in metadata["source_files"].items():
        if not name.startswith(("Hooke/model/", "Hooke/assets/")):
            continue
        actual = hashlib.sha256((repository / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Review miniature centrifuge adaptation after source change: {name}"
            )


def adapt_geometry(source):
    """Adapt an unnamespaced copy; source files and inertial geoms are untouched."""
    metadata, names, owners, vertices, offsets = _resources()
    _verify_upstream(metadata)
    assets = source.find("asset")
    bodies = [
        source.find(f".//body[@name='{name}']") for name in ("body", "lid", "rotor")
    ]
    if assets is None or any(body is None for body in bodies):
        raise ValueError("Miniature centrifuge source hierarchy changed")
    lid_mesh = assets.find("./mesh[@name='lid-visual-0']")
    if lid_mesh is None:
        raise ValueError("Miniature centrifuge source lid is missing")
    lid_mesh.set("file", str((RESOURCE_DIR / "lid_visual.obj").resolve()))
    for geom in list(bodies[1].findall("geom")):
        if geom.get("class") == "collision":
            bodies[1].remove(geom)
    for index, name in enumerate(names):
        ET.SubElement(
            assets,
            "mesh",
            name=str(name),
            vertex=_numbers(vertices[offsets[index] : offsets[index + 1]]),
        )
        ET.SubElement(
            bodies[int(owners[index])],
            "geom",
            name=str(name),
            type="mesh",
            mesh=str(name),
            density="0",
            contype="1",
            conaffinity="1",
            group="3",
            friction=".5 .005 .0001",
        )
    for index, port in enumerate(metadata["ports"]):
        axis = np.asarray(port["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        quat = np.array([1 + axis[2], -axis[1], axis[0], 0])
        quat /= np.linalg.norm(quat)
        ET.SubElement(
            bodies[2],
            "site",
            name=f"port_{index}",
            pos=_numbers(port["center"]),
            quat=_numbers(quat),
            size=".0003",
            rgba=".25 .5 .65 1",
        )
    return (
        "Retained original mass, inertia and joint frames; replaced the filled lid proxy "
        "with open detailed contacts and added real rotor-bore contacts. Local rear-rim "
        "and bearing-fit relief matches the repaired visible lid; its original split "
        "normals are restored. This is an authored mechanical fit, not certified OEM CAD."
    )


def configure(root, name):
    """Declare the copied asset's precision requirement and observable port datums."""
    metadata, *_ = _resources()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    previous = float(option.get("ccd_tolerance", "1e-6"))
    applied = min(previous, CCD_TOLERANCE)
    option.set("ccd_tolerance", format(applied, ".13g"))
    return dict(
        revision=metadata["revision"],
        source_asset=MINI_CENTRIFUGE,
        resource_sha256=dict(metadata["resource_sha256"]),
        contact_count=metadata["contact_count"],
        fixed_body=name + "__body",
        lid_body=name + "__lid",
        rotor_body=name + "__rotor",
        port_sites=[f"{name}__port_{i}" for i in range(len(metadata["ports"]))],
        precision=dict(
            previous_ccd_tolerance=previous,
            applied_ccd_tolerance=applied,
            reason=metadata["numerics"]["basis"],
        ),
        payload="Canonical scene is unloaded; inert free-gauge fixtures are separate variants",
        scope=metadata["scope"],
        limitations=[
            "Four original coarse main-body proxies remain outside the detailed hinge patch",
            "Original mass/inertia remain despite the approximately 1.827% local lid material removal",
            "No calibrated imbalance sensing, lid interlock or scientific centrifugation model",
        ],
    )
