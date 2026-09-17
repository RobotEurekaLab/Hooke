"""Reference environments; pressure is not a gas dynamics implementation."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Environment:
    pressure_pa: float
    boundary_temperature_k: float
    medium: str

    def __post_init__(self):
        if not math.isfinite(self.pressure_pa) or self.pressure_pa < 0:
            raise ValueError("Pressure must be finite and nonnegative")
        if (
            not math.isfinite(self.boundary_temperature_k)
            or self.boundary_temperature_k <= 0
        ):
            raise ValueError("Temperature must be finite and positive")


@dataclass(frozen=True)
class WorldProfile:
    name: str
    label: str
    gravity_m_s2: float
    workspace: Environment
    exterior: Environment
    solar_flux_w_m2: float
    sources: tuple[str, ...]

    def __post_init__(self):
        for value in (self.gravity_m_s2, self.solar_flux_w_m2):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Gravity and solar flux must be finite and nonnegative"
                )

    @property
    def task_name(self):
        return f"space_{self.name}_workstation"

    def report(self):
        return {
            **asdict(self),
            "gravity_vector_m_s2": [0.0, 0.0, -self.gravity_m_s2],
            "gravity_model": (
                "local_free_fall_frame"
                if self.name == "orbital"
                else "uniform_local_surface"
            ),
            "protected_module": {
                "reference_pressure_pa": 101325.0,
                "reference_temperature_k": 293.15,
                "status": "design_requirement_not_simulated_gas_or_thermal_control",
            },
            "implemented": [
                "rigid_body_gravity",
                "collision",
                "fixed_instrument_mounts",
                "radiative_thermal_witness",
            ],
            "disabled": [
                "gas_dynamics",
                "gas_drag",
                "buoyancy",
                "convection",
                "pressure_loads",
                "leaks",
                "boiling",
                "sublimation",
                "dust_transport",
                "electrostatics",
                "ionizing_radiation_damage",
                "material_aging",
                "orbital_tides",
                "station_recoil",
            ],
            "temperature_scope": "declared_scenario_boundary_not_global_air_or_regolith_temperature",
            "scientific_process_validated": False,
        }


WORLDS = {
    "orbital": WorldProfile(
        "orbital",
        "空间站实验舱",
        0.0,
        Environment(101325.0, 293.15, "pressurized_air"),
        Environment(0.0, 3.0, "ideal_vacuum"),
        0.0,
        (
            "https://www.nasa.gov/wp-content/uploads/2023/08/np-2022-06-009-jsc-iss-tech-demo-mini-book-proof-3.pdf",
        ),
    ),
    "lunar": WorldProfile(
        "lunar",
        "月球表面实验工作区",
        1.62,
        Environment(0.0, 250.0, "ideal_vacuum"),
        Environment(0.0, 250.0, "ideal_vacuum"),
        1361.0,
        ("https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html",),
    ),
    "martian": WorldProfile(
        "martian",
        "火星表面实验工作区",
        3.73,
        Environment(636.0, 214.0, "thin_co2_atmosphere"),
        Environment(636.0, 214.0, "thin_co2_atmosphere"),
        586.2,
        ("https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html",),
    ),
}


def exposure_issues(
    world, *, sealed, retained, zone="workspace", requires_liquid=False
):
    """Design screening, independent of any gas/phase-change solver."""
    if zone not in ("workspace", "exterior"):
        raise ValueError("Unknown exposure zone")
    environment = getattr(world, zone)
    issues = []
    if world.gravity_m_s2 == 0 and not retained:
        issues.append("unrestrained_sample_in_microgravity")
    if not sealed and environment.pressure_pa < 10000:
        issues.append("unsealed_sample_in_low_pressure_environment")
    if requires_liquid:
        issues.append("liquid_phase_stability_and_containment_not_qualified")
    return issues
