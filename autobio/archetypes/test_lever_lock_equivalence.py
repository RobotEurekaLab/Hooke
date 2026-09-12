"""Regression test for the lever-lock centrifuge archetype refactor.

`fixtures/lever_lock_reference.npz` was generated once from the *original*,
hand-written `mani_centrifuge_5430.py` / `mani_centrifuge_5910.py`
implementations (as they existed at commit e1c0664, before they were
rewritten into thin `archetypes.lever_lock_centrifuge` instantiations) --
see `private/technical-log.md` for exactly how. This test re-runs the same
seeds through the *current* (now generic-archetype-backed)
`mani_centrifuge_5430.py` / `mani_centrifuge_5910.py` and asserts the
resulting MuJoCo state (qpos, qvel, time) is still bit-identical, so this
file is a durable guard against future edits to the shared archetype code
silently changing behavior for either instrument.

Not run via pytest/CI (no such harness exists in this repo yet) -- invoke
directly:

    python -m archetypes.test_lever_lock_equivalence
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "lever_lock_reference.npz"
_TMP_LOG_ROOT = Path(tempfile.mkdtemp(prefix="lever_lock_equivalence_"))


def run_current(task_cls_path: str, class_name: str, tag: str, seed: int):
    module = __import__(task_cls_path, fromlist=[class_name])
    task_cls = getattr(module, class_name)
    expert = task_cls.Expert(task_cls.load())
    expert.reset(seed=seed)
    expert.set_serializer(log_root=_TMP_LOG_ROOT / tag, log_name=str(seed))
    expert.execute()
    return expert.data.qpos.copy(), expert.data.qvel.copy(), expert.data.time


def check(tag, task_cls_path, class_name, fixture, seeds):
    print(f"=== {tag} ===")
    for seed in seeds:
        cur_qpos, cur_qvel, cur_time = run_current(task_cls_path, class_name, tag, seed)
        ref_qpos = fixture[f"{tag}_seed{seed}_qpos"]
        ref_qvel = fixture[f"{tag}_seed{seed}_qvel"]
        ref_time = fixture[f"{tag}_seed{seed}_time"].item()
        qpos_ok = np.array_equal(ref_qpos, cur_qpos)
        qvel_ok = np.array_equal(ref_qvel, cur_qvel)
        time_ok = ref_time == cur_time
        status = "OK" if (qpos_ok and qvel_ok and time_ok) else "MISMATCH"
        max_qpos_diff = np.max(np.abs(ref_qpos - cur_qpos))
        print(f"  seed={seed}: {status}  (max|dqpos|={max_qpos_diff:.3e}, "
              f"qvel_equal={qvel_ok}, time_equal={time_ok} [{ref_time} vs {cur_time}])")
        assert qpos_ok and qvel_ok and time_ok, f"{tag} seed={seed} diverged from the reference fixture"


if __name__ == "__main__":
    fixture = np.load(FIXTURE_PATH)
    check("centrifuge_5430", "mani_centrifuge_5430", "Centrifuge5430Manipulate", fixture, seeds=[0, 1, 2])
    check("centrifuge_5910", "mani_centrifuge_5910", "Centrifuge5910Manipulate", fixture, seeds=[0, 1, 2])
    print("\nAll equivalence checks passed.")
    shutil.rmtree(_TMP_LOG_ROOT, ignore_errors=True)
