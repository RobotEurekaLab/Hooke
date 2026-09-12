from dataclasses import dataclass

import numpy as np
import mujoco

@dataclass
class JointEquality:
    id: int
    name: str
    active: bool
    joint1: int
    joint2: int | None
    polycoef: np.ndarray

    def compute_joint1(self, joint2):
        return (
            self.polycoef[0] +
            self.polycoef[1] * joint2 +
            self.polycoef[2] * joint2**2 +
            self.polycoef[3] * joint2**3 +
            self.polycoef[4] * joint2**4
        )

def build_equality(model: mujoco.MjModel, i: int) -> JointEquality:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
    eq_type = model.eq_type[i]
    if eq_type != mujoco.mjtEq.mjEQ_JOINT:
        # print(f"Skipping non-joint equality constraint {i}")
        return
    active = bool(model.eq_active0[i])
    joint1 = model.eq_obj1id[i].item()
    joint2 = model.eq_obj2id[i].item()
    if joint2 == -1:
        joint2 = None
    # NOTE: used to assert joint1 > joint2 here. MuJoCo's own <equality><joint
    # joint1=".." joint2=".."/> semantics already fix which side is dependent
    # (joint1 = polynomial(joint2), see compute_joint1 below) independent of
    # their numeric ids -- this assertion was checking an incidental
    # property of ur5e_gripper.xml's own joint declaration order (the only
    # file this had ever processed), not a real requirement. It broke on
    # the first other robot whose gripper-mimic <equality> happened to
    # declare joint1/joint2 the other way around (Franka Panda, UFACTORY
    # xArm7 -- both vendored from MuJoCo Menagerie); nothing downstream in
    # enforce_equality()/expand_qpos() actually depends on this ordering.
    polycoef = model.eq_data[i][:5]
    return JointEquality(i, name, active, joint1, joint2, polycoef)
