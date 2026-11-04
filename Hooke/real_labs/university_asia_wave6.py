"""Yonsei MICS close-up RF probe station with inert mechanical inspection axes."""

from real_labs.architecture import box


SOURCE = "https://mics.yonsei.ac.kr/equipment/"


def _rf_station(b, base, params):
    from real_labs.yonsei_probe_geometry import build_probe_station

    build_probe_station(b, base, params)


def _features(world, definition):
    # Close-up source provides no room: retain an explicitly estimated compact electronics bay.
    box(
        world,
        "mics_bench_back_strip",
        (0.80, 0.02, 0.025),
        (0, 1.35, 1.08),
        (0.62, 0.64, 0.63, 1),
        collision=False,
    )
    for index in range(3):
        box(
            world,
            f"mics_blank_outlet_{index}",
            (0.035, 0.009, 0.025),
            (-0.55 + index * 0.5, 1.325, 1.08),
            (0.82, 0.81, 0.76, 1),
            collision=False,
        )


BUILDERS = {"mics_rf_probe_workspace": _rf_station}
SOURCES = {
    "mics_rf_probe_workspace": dict(
        reference="Yonsei MICS official installed RF probe close-up",
        url=SOURCE,
        dimensions_m=[0.44, 0.36, 0.235],
        dimension_basis="Station dimensions estimated, including the original 10 mm lift range. Photo supports round chuck, tapered opposing heads and rear ribbon; hidden positioners/bench are authored. The visible 40A-GSG-100-DP label and manufacturer nomenclature support 100 micrometre tip spacing only.",
    )
}
SAMPLE_INTERFACES = {
    "mics_rf_probe_workspace": (
        "rf_coupon",
        (0.0048, 0.0040, 0.0004),
        "clamped",
        "Visible inert electronic coupon on a 20 x 16 mm carrier",
    )
}
FEATURES = {"yonsei_mics_rf_probe": _features}
