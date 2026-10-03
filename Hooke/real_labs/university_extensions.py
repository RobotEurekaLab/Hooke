"""Explicit laboratory instrument and furnishing extensions.

Each module owns its instruments, provenance and reference-specific room
features. Scene catalogues are independent JSON so listing them stays cheap.
"""

from functools import lru_cache
from importlib import import_module

MODULES = (
    "university_vacuum",
    "university_probe",
    "university_facilities",
    "university_precision",
    "university_asia_wave1",
    "university_europe_wave1",
    "university_americas_wave1",
    "university_asia_wave2",
    "university_europe_wave2",
    "university_americas_wave2",
    "university_oceania_wave1",
    "university_europe_wave3",
    "university_americas_wave3",
    "university_europe_wave4",
    "university_americas_wave4",
    "university_oceania_wave2",
    "university_americas_wave5",
    "university_asia_wave3",
    "university_europe_wave5",
    "university_asia_wave4",
    "university_americas_wave6",
    "university_asia_wave5",
    "university_engineering_wave1",
    "university_asia_wave6",
    "university_europe_wave6",
    "university_crossregion_wave1",
    "university_engineering_wave2",
    "university_robotics_wave1",
    "university_crossregion_wave2",
    "university_facility_extra_wave1",
    "university_europe_wave7",
    "university_facility_extra_wave2",
)


@lru_cache(maxsize=1)
def instrument_specs():
    result = {}
    for name in MODULES:
        module = import_module("real_labs." + name)
        builders = module.BUILDERS
        if set(builders) != set(module.SOURCES) or set(builders) != set(
            module.SAMPLE_INTERFACES
        ):
            raise ValueError(f"Incomplete instrument extension metadata: {name}")
        for kind, builder in builders.items():
            if kind in result:
                raise ValueError(f"Duplicate instrument extension: {kind}")
            result[kind] = (
                builder,
                module.SOURCES[kind],
                module.SAMPLE_INTERFACES[kind],
            )
    return result


def add_features(world, definition):
    for name in MODULES:
        features = import_module("real_labs." + name).FEATURES
        if definition["id"] in features:
            features[definition["id"]](world, definition)
