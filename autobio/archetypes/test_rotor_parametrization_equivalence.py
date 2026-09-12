"""Regression test for the `InsertCentrifuge5430` rotor-capacity parametrization.

`instrument.py`'s `Centrifuge_Eppendorf_5430._reload` used to hardcode
`range(30)` for slot-site discovery, and `load_centrifuge_5430.py` hardcoded
`% 30` (and an implicit `+15` "opposite slot" offset) in `reset()`. Both were
generalized to use `self.instrument.num_slots` (auto-detected from the
loaded MJCF) so that `InsertCentrifuge5430.for_variant(...)` can point the
same task at a different rotor-capacity scene (see
`archetypes/rotor_variants.py`).

`fixtures/insert_centrifuge_5430_reference.npz` was generated once from the
pre-parametrization code (via commit HEAD at the time, before the num_slots
generalization) against the original 30-slot scene. This test re-runs the
same seeds through the *current* code (still against the original,
unparametrized scene -- this is a regression check that generalizing the
30-slot case didn't change it, not a check of the new variants themselves;
see `archetypes/validate_rotor_variants.py` for validating the variants).

Invoke directly:

    python -m archetypes.test_rotor_parametrization_equivalence
"""
from pathlib import Path

import numpy as np

from load_centrifuge_5430 import InsertCentrifuge5430

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "insert_centrifuge_5430_reference.npz"


if __name__ == "__main__":
    fixture = np.load(FIXTURE_PATH)
    all_ok = True
    for seed in [0, 1, 2]:
        expert = InsertCentrifuge5430.Expert(InsertCentrifuge5430.load())
        expert.reset(seed)
        expert.execute()

        ref_qpos = fixture[f"insert_centrifuge_5430_seed{seed}_qpos"]
        ref_time = fixture[f"insert_centrifuge_5430_seed{seed}_time"].item()
        ref_check = bool(fixture[f"insert_centrifuge_5430_seed{seed}_check"])

        qpos_ok = np.array_equal(ref_qpos, expert.data.qpos)
        time_ok = ref_time == expert.data.time
        check_ok = ref_check == expert.check()
        status = "OK" if (qpos_ok and time_ok and check_ok) else "MISMATCH"
        print(f"seed={seed}: {status} (qpos_equal={qpos_ok}, time_equal={time_ok}, "
              f"check: {ref_check} vs {expert.check()})")
        all_ok &= (qpos_ok and time_ok and check_ok)

    assert all_ok, "InsertCentrifuge5430 diverged from the pre-parametrization reference"
    print("\nAll equivalence checks passed.")
