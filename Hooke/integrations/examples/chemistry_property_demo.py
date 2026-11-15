"""Worked example: ALCHEMI geometry relaxation + MD on one real molecule.

This is deliberately NOT wired into Hooke's task catalog. None of the
catalog's chemistry tasks (filter_cartridge_housing, hplc_injector_plunger,
...) currently carry molecular-level reagent data -- they're about
manipulating instrument hardware (a cartridge housing, an autosampler
plunger), not about the chemistry inside the glassware. Attaching a real
task to this would mean first deciding what molecular structure each task's
reagent should carry, which is a task-catalog schema decision, not
something a client library can settle on its own. See ../README.md.

What this script actually demonstrates, end to end, against two live
ALCHEMI NIM containers (BGR and BMD -- see integrations/alchemi.py for how
to point ALCHEMI_BGR_URL / ALCHEMI_BMD_URL elsewhere, they default to
http://localhost:8000 and :8001):

  1. Start from a simple, hand-built geometry for an isolated water
     molecule -- experimental gas-phase O-H bond length 0.9584 A and H-O-H
     angle 104.45 deg (NIST CCCBDB, https://cccbdb.nist.gov/exp2x.asp).
     Water stands in here for "some reagent a Hooke chemistry task
     touches" -- it's close to universal across this category's fume-hood,
     filter-wash and HPLC-mobile-phase tasks, and it keeps this example's
     only hand-specified numbers to the two most textbook-certain ones
     for any molecule. It is not read from the task catalog.
  2. Relax it with BGR to its nearest energy minimum. BGR's whole job is
     to take a reasonable starting guess and relax it -- the input above
     does not need to already be at the minimum.
  3. Run a short BMD trajectory from the relaxed structure at room
     temperature and report the resulting energy.

Run the containers with ALCHEMI_NIM_MODEL_TYPE=aimnet2: of the three MLIP
families NVIDIA documents for these NIMs (MACE, TensorNet, AIMNet2), only
AIMNet2 is documented as supporting non-periodic (isolated-molecule)
systems -- MACE and TensorNet are periodic-systems-only.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrations.alchemi import (  # noqa: E402
    AlchemiConfig,
    AtomicStructure,
    MDConfig,
    relax_geometry,
    run_molecular_dynamics,
)

WATER = AtomicStructure(
    numbers=[8, 1, 1],
    coord=[
        0.0, 0.0, 0.0,
        0.757541, 0.58708, 0.0,
        -0.757541, 0.58708, 0.0,
    ],
    pbc=[False, False, False],
    structure_id="water_demo",
)


def main() -> None:
    config = AlchemiConfig()

    relaxed = relax_geometry(config, [WATER])
    structure = relaxed.structures[0]
    print(
        f"BGR: converged={structure.converged} after {structure.optimizer_nsteps} steps, "
        f"relaxed energy={structure.energy:.6f} (model-defined units, typically eV for these MLIPs)"
    )

    relaxed_atoms = AtomicStructure(
        numbers=structure.numbers,
        coord=structure.coord,
        pbc=[False, False, False],
        structure_id="water_demo_relaxed",
    )
    trajectory = run_molecular_dynamics(config, relaxed_atoms, MDConfig(temperature_k=298.15, md_time_ps=1.0))
    last = trajectory.trajectory[-1]
    print(
        f"BMD: {len(trajectory.trajectory)} saved frames over {last.md_time:.3f} ps at 298.15 K, "
        f"final energy={trajectory.final_energy:.6f} (same units as above)"
    )


if __name__ == "__main__":
    main()
