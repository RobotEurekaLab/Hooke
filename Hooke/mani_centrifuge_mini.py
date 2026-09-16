"""Tiangen mini lid controller using the shared instrument geometry and motion."""
from archetypes.lever_lock_centrifuge import make_task_classes
from archetypes.centrifuge_specs import CENTRIFUGE_MINI_SPEC

CentrifugeMiniManipulate, CentrifugeMiniManipulateExpert = make_task_classes(CENTRIFUGE_MINI_SPEC)

if __name__ == "__main__":
    expert = CentrifugeMiniManipulateExpert(CentrifugeMiniManipulate.load())
    expert.reset(0)
    expert.set_serializer()
    expert.execute()
