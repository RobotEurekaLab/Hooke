"""Visual finishes for fixed lab accessories, independent of instrument mechanics."""

from pathlib import Path
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from microscopy.geometry import cylinder, geom, numbers, rounded_box


def _alignment_texture():
    """An explicitly labelled setup target, not a simulated live camera reading."""
    image = Image.new("RGB", (960, 540), (18, 25, 30))
    draw = ImageDraw.Draw(image)
    large, small = ImageFont.load_default(size=27), ImageFont.load_default(size=18)
    draw.text((32, 25), "MICROSCOPY  /  OPTICAL ALIGNMENT", font=large, fill=(218, 230, 235))
    draw.line((32, 72, 928, 72), fill=(54, 72, 80), width=2)
    for x in range(72, 665, 37):
        draw.line((x, 102, x, 435), fill=(38, 51, 59))
    for y in range(102, 436, 37):
        draw.line((72, y, 664, y), fill=(38, 51, 59))
    for radius in (36, 74, 111):
        draw.ellipse((368-radius, 269-radius, 368+radius, 269+radius), outline=(121, 155, 165), width=2)
    draw.line((344, 269, 392, 269), fill=(187, 218, 218), width=2)
    draw.line((368, 245, 368, 293), fill=(187, 218, 218), width=2)
    for y, title, subtitle in ((125, "ILLUMINATION", "Transmitted light"),
                               (229, "SAMPLE PLANE", "Stage alignment"),
                               (333, "MICROTOOLS", "Approach clearance")):
        draw.text((716, y), title, font=small, fill=(137, 185, 187))
        draw.text((716, y+29), subtitle, font=small, fill=(194, 204, 210))
    draw.text((32, 487), "SETUP TARGET", font=small, fill=(145, 178, 182))
    draw.text((605, 487), "Use microscope view for live imaging", font=small, fill=(146, 156, 167))
    # Content-addressing and atomic replacement also support concurrent workers.
    digest = hashlib.sha256(image.tobytes()).hexdigest()[:16]
    directory = Path(__file__).resolve().parents[2]/"temp"/"microscopy_assets"/"textures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"alignment-{digest}.png"
    if not path.exists():
        descriptor, temporary = tempfile.mkstemp(suffix=".png", dir=directory)
        os.close(descriptor)
        try:
            image.save(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


def _round_existing(asset, world, name, radius):
    node = world.find(f"geom[@name='{name}']")
    if node is None:
        return
    attributes = dict(node.attrib)
    halfsize = [float(value) for value in attributes.pop("size").split()]
    position = [float(value) for value in attributes.pop("pos", "0 0 0").split()]
    for key in ("name", "type"):
        attributes.pop(key)
    world.remove(node)
    rounded_box(asset, world, name, halfsize, position, radius=radius, **attributes)


def finish_accessories(root):
    """Add massless surface details to the existing fixed console and monitor."""
    asset, world = root.find("asset"), root.find("worldbody")
    for name, radius in (("pressure_controller", .006), ("monitor", .006)):
        _round_existing(asset, world, name, radius)
    console = world.find("geom[@name='pressure_controller']")
    if console is not None:
        cx, cy, cz = map(float, console.get("pos").split())
        rounded_box(asset, world, "console_base_seam", (.0683, .0553, .002),
                    (cx, cy, cz-.032), radius=.001, material="graphite", mass="0")
        for index, (dx, dy) in enumerate(((-.05, -.037), (.05, -.037), (-.05, .037), (.05, .037))):
            geom(world, f"console_foot_{index}", "cylinder", (.006, .0025),
                 (cx+dx, cy+dy, cz-.0425), "rubber", mass="0")
        for index, x in enumerate((cx-.058, cx+.058)):
            cylinder(world, f"console_panel_screw_{index}", (x, cy-.0555, cz+.027),
                     (x, cy-.0561, cz+.027), .0015, "metal", mass="0")
        for tool in ("holder", "injector"):
            display = world.find(f"geom[@name='pressure_display_{tool}']")
            if display is None:
                continue
            x, y, z = map(float, display.get("pos").split())
            geom(world, f"pressure_indicator_{tool}", "box", (.010, .0001, .0006),
                 (x, y-.0009, z-.003), "cyan", mass="0")
    for node in list(world.findall("geom")):
        if node.get("name", "").startswith(("display_bar_", "display_line_")):
            world.remove(node)
    panel = world.find("geom[@name='monitor_panel']")
    if panel is None:
        return
    x, y, z = map(float, panel.get("pos").split())
    hx, hy, hz = map(float, panel.get("size").split())
    ET.SubElement(asset, "texture", name="alignment_display_texture", type="2d",
                  file=str(_alignment_texture()))
    ET.SubElement(asset, "material", name="alignment_display", texture="alignment_display_texture",
                  rgba="1 1 1 1", emission=".35", specular=".08", shininess=".15")
    # A very thin solid avoids degenerate mesh hulls while retaining explicit UVs.
    vertices = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz),
                (-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
    ET.SubElement(asset, "mesh", name="alignment_panel_mesh", vertex=numbers([v for p in vertices for v in p]),
                  face=" ".join(str(v) for f in faces for v in f),
                  texcoord="0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0")
    world.remove(panel)
    ET.SubElement(world, "geom", name="monitor_panel", type="mesh", mesh="alignment_panel_mesh",
                  pos=numbers((x, y, z)), material="alignment_display", mass="0")
