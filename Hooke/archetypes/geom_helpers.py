"""Shared MJCF-generation helpers for procedurally-built display instruments.

Every breadth batch since `static_items2.py` (see private/technical-log.md)
built its instruments with a small set of primitive-geometry helper
functions (`geom`, `moving_body`, `chamber_static`/`chamber_door`,
`boxy_static`) plus two string templates, copy-pasted into a throwaway
scratchpad generator script each time rather than committed anywhere.
That was fine while each batch was a one-off; `protocol_to_task.py` needs
the same generation logic at *run time* (to turn a parsed protocol step
into an actual instrument), so it's promoted here as a real, importable,
committed module instead of being re-derived a fifth time.
"""
from __future__ import annotations

SCENE_TEMPLATE = """<mujoco model="{title}">

    <option integrator="implicitfast" impratio="10" cone="elliptic" noslip_iterations="2">
        <flag multiccd="enable"/>
    </option>
    <visual>
        <global azimuth="220" elevation="-30" offwidth="1280" offheight="960"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
        <model name="table" file="../misc/simple_table.xml" content_type="text/xml" />
        <model name="{instrument}" file="../instrument/{instrument}.xml" content_type="text/xml" />
        <model name="ur5e" file="../robot/ur5e_gripper.xml" content_type="text/xml" />
    </asset>

    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 0" size="2.5 2 0.05" type="plane" material="groundplane"/>
        <body name="table" pos="0 0 0." quat="1 0 0 1">
            <attach model="table" body="vention table" prefix="/"/>
            <site name="instrument_site" pos="0.0 0.0 0.824" size="0.01" rgba="1 0 0 1" group="3" />
            <body name="{instrument}" pos="0.0 0.0 0.824">
                <attach model="{instrument}" body="base" prefix="/{instrument}:" />
            </body>
            <site name="arm1_site" pos="0.5 -0.0 0.824" quat="0 0 0 1" size="0.01" rgba="0 1 0 1" group="3" />
            <body name="1/ur5e" pos="0.5 -0. 0.824" quat="0 0 0 1">
                <attach model="ur5e" body="world" prefix="/ur:"/>
            </body>
            <camera name="table_cam_front" pos="0. -1.5 1.65" quat="0.819 0.574 0 0" fovy="45" resolution="1280 960"/>
            <camera name="table_cam_left" pos="-1.2 0. 1.65" quat="0.579 0.406 -0.406 -0.579" fovy="45" resolution="1280 960"/>
        </body>
    </worldbody>

</mujoco>
"""

INSTRUMENT_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <default class="body"><geom contype="1" conaffinity="1"/></default>
  </default>
  <worldbody>
    <body name="base">
{static_geoms}
{moving_body}
    </body>
  </worldbody>
</mujoco>
"""


def geom(type_, size, pos, rgba, quat=None, name=None):
    attrs = f'type="{type_}" size="{size}" pos="{pos}" rgba="{rgba}" class="body"'
    if quat:
        attrs += f' quat="{quat}"'
    if name:
        attrs = f'name="{name}" ' + attrs
    return f'      <geom {attrs}/>'


def moving_body(body_name, jtype, axis, jrange, pos, geoms, damping="1.0", frictionloss="0.3"):
    inner = "\n".join(f"  {g.strip()}" for g in geoms)
    return (
        f'      <body name="{body_name}" pos="{pos}">\n'
        f'        <joint name="{body_name}_joint" type="{jtype}" axis="{axis}" range="{jrange}" '
        f'damping="{damping}" frictionloss="{frictionloss}"/>\n'
        f'{inner}\n'
        f'      </body>'
    )


def boxy_static(body_size, body_color, n_static_knobs=1, screen=True, screen_color="0.9 0.7 0.1 1"):
    hx, hy, hz = body_size
    g = [geom("box", f"{hx} {hy} {hz}", f"0 0 {hz}", body_color)]
    if screen:
        g.append(geom("box", f"{hx*0.3} {hy*0.05} {hz*0.25}", f"{-hx*0.5} {-hy*0.9} {hz*1.5}", screen_color))
    knob_colors = ["0.8 0.1 0.1 1", "0.1 0.1 0.8 1"]
    for i in range(n_static_knobs):
        xo = hx * 0.5 * (0.6 if i == 0 else 0.2)
        g.append(geom("cylinder", "0.01 0.008", f"{xo} {hy*0.3} {2*hz+0.005}", knob_colors[i % 2]))
    return g


def chamber_static(footprint, height, wall_color):
    hx, hy = footprint
    hz = height / 2
    return [
        geom("box", f"{hx} 0.01 {hz}", f"0 {hy} {hz}", wall_color),
        geom("box", f"0.01 {hy} {hz}", f"{-hx} 0 {hz}", wall_color),
        geom("box", f"0.01 {hy} {hz}", f"{hx} 0 {hz}", wall_color),
        geom("box", f"{hx} {hy} 0.01", f"0 0 {height}", wall_color),
    ]


def chamber_door(hx, hz, window_color="0.6 0.8 0.9 0.35"):
    return moving_body("door", "hinge", "0 0 1", "0 1.8", f"{-hx} 0.0 {hz}",
                        [geom("box", f"{hx} 0.005 {hz}", f"{hx} 0 0", window_color)],
                        damping="0.6")


def write_instrument_and_scene(name: str, static_geoms: list[str], moving: str, model_root) -> tuple[str, str]:
    """Writes model/instrument/{name}.xml and model/scene/mani_{name}.xml
    under `model_root` (a Path to the `model/` directory), returns their
    paths as strings."""
    inst_path = model_root / "instrument" / f"{name}.xml"
    inst_path.write_text(INSTRUMENT_HEADER.format(
        name=name, static_geoms="\n".join(static_geoms), moving_body=moving,
    ))
    scene_path = model_root / "scene" / f"mani_{name}.xml"
    scene_path.write_text(SCENE_TEMPLATE.format(title=name, instrument=name))
    return str(inst_path), str(scene_path)


MULTI_INSTRUMENT_SCENE_TEMPLATE = """<mujoco model="{title}">

    <option integrator="implicitfast" impratio="10" cone="elliptic" noslip_iterations="2">
        <flag multiccd="enable"/>
    </option>
    <visual>
        <global azimuth="220" elevation="-30" offwidth="1280" offheight="960"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
        <model name="table" file="../misc/simple_table.xml" content_type="text/xml" />
{instrument_assets}
        <model name="ur5e" file="../robot/ur5e_gripper.xml" content_type="text/xml" />
    </asset>

    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 0" size="2.5 2 0.05" type="plane" material="groundplane"/>
        <body name="table" pos="0 0 0." quat="1 0 0 1">
            <attach model="table" body="vention table" prefix="/"/>
            <site name="instrument_site" pos="0.0 0.0 0.824" size="0.01" rgba="1 0 0 1" group="3" />
{instrument_bodies}
            <site name="arm1_site" pos="0.5 -0.0 0.824" quat="0 0 0 1" size="0.01" rgba="0 1 0 1" group="3" />
            <body name="1/ur5e" pos="0.5 -0. 0.824" quat="0 0 0 1">
                <attach model="ur5e" body="world" prefix="/ur:"/>
            </body>
            <camera name="table_cam_front" pos="0. -1.5 1.65" quat="0.819 0.574 0 0" fovy="45" resolution="1280 960"/>
            <camera name="table_cam_left" pos="-1.2 0. 1.65" quat="0.579 0.406 -0.406 -0.579" fovy="45" resolution="1280 960"/>
        </body>
    </worldbody>

</mujoco>
"""


def write_multi_instrument_scene(scene_name: str, instrument_names: list[str], offsets: list[float], model_root) -> str:
    """Writes model/scene/mani_{scene_name}.xml combining several already-
    existing instruments (each already built under model/instrument/), each
    placed at a distinct attach-body offset (meters, along the axis that
    empirically maps to world X -- see private/technical-log.md's first
    cross-instrument composite entry for how this was confirmed, not
    assumed). Does not touch model/instrument/ at all -- the instruments
    themselves must already exist. Returns the scene path as a string.

    Callers are responsible for spacing `offsets` far enough apart that the
    instruments' own footprints don't overlap (checked empirically per
    scene when this is used, not enforced here)."""
    assert len(instrument_names) == len(offsets)
    assets = "\n".join(
        f'        <model name="{n}" file="../instrument/{n}.xml" content_type="text/xml" />'
        for n in instrument_names
    )
    bodies = "\n".join(
        f'            <body name="{n}" pos="0.0 {off} 0.824">\n'
        f'                <attach model="{n}" body="base" prefix="/{n}:" />\n'
        f'            </body>'
        for n, off in zip(instrument_names, offsets)
    )
    scene_path = model_root / "scene" / f"mani_{scene_name}.xml"
    scene_path.write_text(MULTI_INSTRUMENT_SCENE_TEMPLATE.format(
        title=scene_name, instrument_assets=assets, instrument_bodies=bodies,
    ))
    return str(scene_path)
