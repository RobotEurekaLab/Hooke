"""Authored open spindle support for the representative SHRIMP sample stage.

The mounting geometry repairs a gap in the surrogate's load path. It is not an
OEM bearing design, a vacuum seal, or a bearing friction/compliance model.
"""

from real_labs.mounting import annular_mount


def transfer_deck(b, parent):
    """Keep the original deck envelope with a central 20 mm square opening."""
    for key, position, half in (
        ("left", (-0.043, 0, 0), (0.033, 0.06, 0.012)),
        ("right", (0.043, 0, 0), (0.033, 0.06, 0.012)),
        ("front", (0, -0.035, 0), (0.010, 0.025, 0.012)),
        ("back", (0, 0.035, 0), (0.010, 0.025, 0.012)),
    ):
        b.box(
            parent,
            position,
            half,
            "dark",
            collision=True,
            name=f"{b.name}__stage_deck_{key}",
        )
    # The flange rests on the deck and carries the puck's lower face. The
    # sleeve descends into the genuine deck opening; its bore clears the shaft.
    for key, center, outer, half in (
        ("sleeve", (0, 0, 0.009), 0.0095, 0.004),
        ("thrust_flange", (0, 0, 0.0125), 0.016, 0.0005),
    ):
        annular_mount(
            b,
            parent,
            key,
            center,
            0.0053,
            outer,
            half,
            axis="z",
            proxies=True,
            namespace="stage",
            material="bright",
        )


def puck_spindle(b, parent):
    """Join the unchanged rotating puck to a shaft inside the open sleeve."""
    b.cylinder(
        parent,
        (0, 0, -0.013),
        0.005,
        0.004,
        "bright",
        collision=True,
        name=f"{b.name}__stage_spindle",
    )
