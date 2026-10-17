"""Neutral lab surfaces and visual surroundings; no instrument dynamics."""

import xml.etree.ElementTree as ET

from microscopy.workstation_finish import finish_accessories


def add_lab_appearance(root):
    asset, world = root.find("asset"), root.find("worldbody")
    for name, colour, specular, shininess in (
        ("ivory", ".87 .865 .83 1", .3, .32),
        ("graphite", ".075 .082 .086 1", .3, .3),
        ("metal", ".63 .65 .67 1", .6, .55),
        ("glass", ".91 .94 .96 .24", .95, .96),
    ):
        node = asset.find(f"material[@name='{name}']")
        node.set("rgba", colour)
        node.set("specular", str(specular))
        node.set("shininess", str(shininess))
    for name, colour in (("lab_floor", ".48 .49 .49 1"),
                          ("lab_wall", ".79 .78 .75 1"),
                          ("breadboard", ".49 .51 .52 1")):
        ET.SubElement(asset, "material", name=name, rgba=colour, specular=".08", shininess=".15")
    asset.find("texture[@type='skybox']").set("rgb1", ".66 .68 .69")
    asset.find("texture[@type='skybox']").set("rgb2", ".36 .39 .4")
    world.find("geom[@name='floor']").set("material", "lab_floor")
    world.find("geom[@name='bench']").set("material", "breadboard")
    ET.SubElement(world, "geom", name="lab_back_wall", type="box", size="1.6 .02 1.1",
                  pos="0 1.05 1.1", material="lab_wall", mass="0", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="lab_wall_trim", type="box", size="1.6 .026 .04",
                  pos="0 1.02 .06", material="graphite", mass="0", contype="0", conaffinity="0")
    root.find("visual/headlight").set("ambient", ".36 .36 .36")
    root.find("visual/headlight").set("diffuse", ".46 .46 .46")
    lights = world.findall("light")
    lights[0].set("diffuse", ".84 .82 .79")
    lights[1].set("diffuse", ".65 .67 .69")
    finish_accessories(root)
