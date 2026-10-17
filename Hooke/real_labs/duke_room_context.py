"""Photo-supported open shelving in the estimated Duke test-frame bay."""

from real_labs.architecture import box


def add_context(world, definition):
    timber = (0.63, 0.53, 0.29, 1)
    for index, x in enumerate((-1.20, -0.40, 0.40)):
        prefix = f"duke_open_shelf_{index}_"
        box(world, prefix + "back", (0.38, 0.015, 0.40), (x, 2.195, 1.82), timber)
        for side in (-1, 1):
            box(
                world,
                prefix + f"side_{side}",
                (0.016, 0.24, 0.40),
                (x + side * 0.364, 1.97, 1.82),
                timber,
            )
        for height in (1.435, 1.70, 1.96, 2.205):
            box(
                world,
                prefix + f"board_{height}",
                (0.38, 0.24, 0.015),
                (x, 1.97, height),
                timber,
            )
        for height in (1.50, 2.13):
            box(
                world,
                prefix + f"anchor_{height}",
                (0.30, 0.060, 0.018),
                (x, 2.245, height),
                (0.32, 0.33, 0.31, 1),
            )
        # A few thin source-visible storage trays, not invented experimental
        # equipment. Each rests on an actual shelf surface.
        if index != 1:
            for layer in range(2):
                box(
                    world,
                    prefix + f"tray_{layer}",
                    (0.13, 0.12, 0.008),
                    (x + 0.08, 1.92, 1.983 + 0.016 * layer),
                    (0.30, 0.32, 0.32, 1),
                )
