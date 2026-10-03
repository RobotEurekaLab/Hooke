"""Free specimen carriers settled by contact, not attached by hidden welds."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def add_specimens(root, equipment):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    records = []
    for item in equipment:
        if item.get("sample_attachment") != "supported":
            continue
        size = np.asarray(item["sample_size_m"], dtype=float)
        site = model.site(item["sample_site"]).id
        pos = data.site_xpos[site].copy()
        pos[2] += 0.01 + (size[2] / 2 if item["kind"] == "mobile_robot" else 0)
        name = item["id"] + "__specimen"
        body = ET.SubElement(
            root.find("worldbody"), "body", name=name, pos=" ".join(map(str, pos))
        )
        ET.SubElement(body, "freejoint", name=name + "__free")
        ET.SubElement(
            body,
            "geom",
            name=name + "__collision",
            type="box",
            size=" ".join(map(str, size / 2)),
            rgba=".35 .54 .62 1",
            mass=".005",
            friction="1 .01 .001",
            solref=".006 1",
            solimp=".95 .99 .0001",
        )
        process = dict(
            sample_site=item["sample_site"],
            sample_body=name,
            load_tolerance=max(0.015, float(size[2] / 2 + 0.006)),
        )
        access = next(
            (
                key
                for key in ("door", "lid", "chamber_lid", "flowcell_tray")
                if key in item["actuators"]
            ),
            None,
        )
        if access:
            process.update(access_control=access, closed_value=0, closed_tolerance=0.01)
        item["process"] = process
        records.append(
            dict(
                body=name,
                equipment=item["id"],
                description=item["sample_description"],
                dimensions_m=size.tolist(),
                fidelity="rigid specimen carrier; not a scientific specimen model",
            )
        )
    return records
