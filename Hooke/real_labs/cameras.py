"""Validate instrument-mounted cameras without converting them into room views."""

import math


def _vector(value, length, label):
    values = tuple(float(v) for v in value)
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Invalid mounted camera {label}")
    return values


def _matches(first, second):
    return all(abs(a - b) <= 1e-8 for a, b in zip(first, second))


def mounted_camera_records(world, equipment, room_views):
    """Bind declared names, mounts and local optics to the actual XML nodes.

    Local poses are mounting geometry. World poses come from the compiled model
    at observation time; a moving camera must never acquire a duplicate fixed
    camera through the room-view dictionary.
    """
    parents = {child: parent for parent in world.iter() for child in parent}
    cameras = {}
    sites = {}
    for node in world.iter():
        if node.tag in ("camera", "site"):
            table = cameras if node.tag == "camera" else sites
            table.setdefault(node.get("name"), []).append(node)
    names = set(room_views)
    records = []
    for item in equipment:
        for record in item.get("mounted_cameras", []):
            name, body = record["name"], record["body"]
            nodes = cameras.get(name, [])
            if name in names or len(nodes) != 1:
                raise ValueError(f"Missing or duplicate mounted camera: {name}")
            camera = nodes[0]
            parent = parents[camera]
            if (
                parent.tag != "body"
                or parent.get("name") != body
                or not body.startswith(item["id"] + "__")
            ):
                raise ValueError(f"Mounted camera has a different body: {name}")
            if camera.get("mode", "fixed") != "fixed" or camera.get("target"):
                raise ValueError(f"Mounted camera must use its body's frame: {name}")
            position = _vector(record["local_position_m"], 3, "position")
            rotation = _vector(record["local_quaternion_wxyz"], 4, "quaternion")
            if abs(sum(v * v for v in rotation) - 1) > 1e-8:
                raise ValueError(
                    f"Mounted camera quaternion must be unit length: {name}"
                )
            xml_position = _vector(
                camera.get("pos", "0 0 0").split(), 3, "XML position"
            )
            if any(
                key in camera.attrib
                for key in ("euler", "axisangle", "xyaxes", "zaxis")
            ):
                raise ValueError(f"Declare mounted camera rotation with quat: {name}")
            xml_rotation = _vector(
                camera.get("quat", "1 0 0 0").split(), 4, "XML quaternion"
            )
            if not _matches(position, xml_position) or not (
                _matches(rotation, xml_rotation)
                or _matches(rotation, tuple(-v for v in xml_rotation))
            ):
                raise ValueError(
                    f"Mounted camera local pose disagrees with XML: {name}"
                )
            fovy = float(record["fovy_deg"])
            xml_fovy = float(camera.get("fovy", "45"))
            if (
                not 0 < fovy < 180
                or not 0 < xml_fovy < 180
                or abs(xml_fovy - fovy) > 1e-8
            ):
                raise ValueError(
                    f"Mounted camera field of view disagrees with XML: {name}"
                )
            if any(
                key in camera.attrib
                for key in ("orthographic", "focal", "focalpixel", "sensorsize")
            ):
                raise ValueError(
                    f"Mounted camera must declare perspective optics with fovy: {name}"
                )
            site_nodes = sites.get(record["site"], [])
            if len(site_nodes) != 1 or parents[site_nodes[0]] is not parent:
                raise ValueError(
                    f"Mounted camera datum belongs to another frame: {name}"
                )
            site_position = _vector(
                site_nodes[0].get("pos", "0 0 0").split(), 3, "site position"
            )
            if not _matches(position, site_position):
                raise ValueError(
                    f"Mounted camera is displaced from its declared datum: {name}"
                )
            names.add(name)
            records.append(dict(record, equipment=item["id"]))
    return records
