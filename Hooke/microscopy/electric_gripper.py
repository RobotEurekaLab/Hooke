"""Original estimated motor housing and guides for the calibration-bead gripper.

Appearance surfaces have no added collision or mass. The existing two jaw
slides supply the physical actuation; this is not vendor CAD or electronics.
"""

from microscopy.geometry import cylinder, geom


def housing(parent):
    geom(parent, "gripper_reference_roof", "box", (.006, .004, .0005), (.011, 0, .0065), "graphite", mass="0")
    geom(parent, "gripper_reference_back", "box", (.001, .004, .002), (.016, 0, .004), "metal", mass="0")
    for sign, side in ((1, "a"), (-1, "b")):
        geom(parent, "gripper_reference_side_" + side, "box", (.005, .0004, .002),
             (.011, sign * .0036, .004), "metal", mass="0")
        geom(parent, "gripper_reference_motor_" + side, "box", (.003, .0015, .0015),
             (.020, sign * .0022, .005), "graphite", mass="0")
    for index, x in enumerate((.008, .011)):
        cylinder(parent, f"gripper_reference_guide_{index}", (x, -.0035, .002),
                 (x, .0035, .002), .00025, "metal", mass="0")


def carriage(jaw, side, sign):
    geom(jaw, "jaw_" + side + "_reference_carriage", "box", (.0015, .00045, .0007),
         (.009, sign * .001, .002), "metal", mass="0")
