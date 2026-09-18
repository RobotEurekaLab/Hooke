"""Estimated pressure console and encoder-driven visual hose routing.

Pressure dynamics live in the task mechanics. These visual hoses carry no
contact or mass and do not simulate elastic tube motion or distributed flow.
"""

import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from microscopy.geometry import cylinder, geom, numbers
from microscopy.needle_holder import HOLDER_LENGTH_M


def hose_points(controls):
    controls = np.asarray(controls, dtype=float)
    t = np.linspace(0., 1., 49)[:, None]
    return ((1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]
            +3*(1-t)*t**2*controls[2]+t**3*controls[3])


def hose_visuals(model, data):
    shapes = []
    for tool in ("injector", "holder"):
        names = [f"pneumatic_{tool}_anchor{i}" for i in range(4)]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
        if ids[0] < 0:
            continue
        if any(index < 0 for index in ids):
            raise ValueError(f"Incomplete pneumatic route for {tool}")
        points = hose_points(data.site_xpos[ids])
        for start, end in zip(points, points[1:]):
            vector = end-start
            length = float(np.linalg.norm(vector))
            if length < 1e-10:
                raise ValueError("Degenerate pneumatic hose segment")
            z = vector/length
            helper = np.eye(3)[int(np.argmin(abs(z)))]
            x = np.cross(helper, z)
            x /= np.linalg.norm(x)
            matrix = np.column_stack((x, np.cross(z, x), z))
            shapes.append(dict(type=5, role="pneumatic_hose", size=[.0015, length/2, 0.],
                pos=((start+end)/2).tolist(), mat=matrix.ravel().tolist(),
                rgba=[.58, .69, .72, .65], surface=dict(roughness=.65, specular_color=[.03]*3)))
    return shapes


def compact_console(world, origin, homes, tools, bench_top):
    centre = np.array([.37, .15, bench_top+.045])
    geom(world, "pressure_controller", "box", (.068, .055, .040), centre, "ivory")
    for index, tool in enumerate(tool for tool in ("injector", "holder") if tool in tools):
        x = .340+.045*index
        geom(world, f"pressure_display_{tool}", "box", (.016, .0008, .008),
             (x, .094, bench_top+.054), "graphite")
        cylinder(world, f"pressure_dial_{tool}", (x, .093, bench_top+.029),
                 (x, .085, bench_top+.029), .006, "graphite")
        holder = world.find(f".//body[@name='{tool}_needle_holder']")
        if holder is not None:
            endpoint = np.array([0., 0., HOLDER_LENGTH_M])
            from scipy.spatial.transform import Rotation
            quaternion = np.fromstring(holder.get("quat"), sep=" ")
            rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            end = origin+homes[tool]+np.fromstring(holder.get("pos"), sep=" ")+rotation@endpoint
        else:
            # Reference manipulators connect at their existing collet, without
            # implying that the optional commercial holder is present.
            holder = next(node for node in world.iter("body")
                          if node.find(f"site[@name='{tool}_tcp']") is not None)
            collet = holder.find(f"geom[@name='{tool}_collet']")
            endpoint = np.fromstring(collet.get("fromto"), sep=" ")[:3]
            rotation = np.eye(3)
            end = origin+homes[tool]+endpoint
        port = np.array([x, .173, bench_top+.087])
        cylinder(world, f"pressure_port_{tool}", port-[0, 0, .007], port, .0025)
        # Separate rear loops avoid the observation and needle-approach region.
        for index, point in enumerate((port, [x+.035, .30, bench_top+.26])):
            ET.SubElement(world, "site", name=f"pneumatic_{tool}_anchor{index}",
                          pos=numbers(point), size=".0001", group="5")
        control = endpoint+rotation.T@np.array([.04, .37-end[1], .06])
        for index, point in ((2, control), (3, endpoint)):
            ET.SubElement(holder, "site", name=f"pneumatic_{tool}_anchor{index}",
                          pos=numbers(point), size=".0001", group="5")
