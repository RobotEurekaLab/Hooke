"""Centrifuge Eppendorf 5910: close-lid task.

This used to be a ~260-line file with its own copy of `UR5eArm` and the
lever/lock interaction logic. It's now a thin instantiation of the generic
`archetypes.lever_lock_centrifuge` archetype -- see that module and
`archetypes/centrifuge_specs.py` for the actual logic/numbers, and
`private/technical-log.md` for the equivalence check run when this file was
migrated (bit-identical qpos/qvel/time vs. the original implementation,
seeds 0-2).
"""
from archetypes.lever_lock_centrifuge import make_task_classes
from archetypes.centrifuge_specs import CENTRIFUGE_5910_SPEC

Centrifuge5910Manipulate, Centrifuge5910ManipulateExpert = make_task_classes(CENTRIFUGE_5910_SPEC)

if __name__ == "__main__":
    from tqdm import trange
    spec = Centrifuge5910Manipulate.load()
    expert = Centrifuge5910Manipulate.Expert(spec)
    for i in trange(1):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
