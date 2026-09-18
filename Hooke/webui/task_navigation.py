"""Presentation and launch links for the scene picker's task catalogue.

Keep these UI groups separate from the task taxonomy used by controllers
and protocol composition. Resolving an entry never loads a simulator.
"""

from urllib.parse import urlencode

from archetypes.task_catalog import CatalogEntry


WORLD_LABELS = {
    "lunar": "Lunar Experiments",
    "martian": "Martian Experiments",
    "orbital": "Space Station Experiments",
}
MICROSCOPY_LABELS = {
    "push": "Cell Pushing with a Blunt Probe",
    "pick_place": "Cell Grasp and Transfer",
    "injection": "Volume-Controlled Cell Injection · 0.5 pL",
    "cell_injection": "Adherent Cell Injection (phantom) · 0.5 pL",
    "suction_injection": "Suspended Cell Holding and Injection (phantom) · 0.5 pL",
}
SPACE_LABELS = {
    "surface_sampling": "Rover Rock Sampling and Return",
    "humanoid_rover": "Humanoid and Rover Cooperation",
    "sample_transfer": "Sample Loading and Retrieval",
    "mass_measurement": "Mass Measurement and Calibration",
    "spectral_measurement": "Synthetic Reflectance Spectrum Measurement",
    "assets_workstation": "Open Asset Scene (display only)",
    "workstation": "Environment Workstation (display only)",
}


def task_navigation(entry: CatalogEntry) -> dict:
    """Describe a task without changing its identifier or execution rules."""
    navigation = {
        "category": entry.category,
        "category_label": entry.category.replace("_", " ").title(),
        "task_label": entry.name,
        "url": None,
    }
    if entry.category == "microscopy":
        operation = entry.name.removeprefix("microscopy_")
        navigation.update(
            category_label="Microscopy Lab",
            task_label=MICROSCOPY_LABELS.get(operation, entry.name),
        )
        if operation in MICROSCOPY_LABELS:
            navigation["url"] = "/microscopy?" + urlencode({"experiment": operation})
    elif entry.category == "space":
        parts = entry.name.split("_", 2)
        if len(parts) != 3:
            return navigation
        prefix, world, operation = parts
        if prefix == "space" and world in WORLD_LABELS:
            navigation.update(
                category=world,
                category_label=WORLD_LABELS[world],
                task_label=SPACE_LABELS.get(operation, entry.name),
                url="/backends?" + urlencode({"task": entry.name}),
            )
            if entry.completion_rule == "surface_sampling" and world != "orbital":
                parameters = {"world": world}
                if operation == "humanoid_rover":
                    parameters["scenario"] = "team"
                navigation["url"] = "/surface-missions?" + urlencode(parameters)
    return navigation
