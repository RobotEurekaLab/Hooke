"""Evaluate source passive laws, without advancing MuJoCo physics.

Native drives already apply scalar joint springs/damping and gravity
compensation. The residual retains tendon elasticity, fluid forces and the
original passive plugin laws (including detents).
"""
import numpy as np
import mujoco


def update_kinematics(model,data):
    """Refresh source FK and passive laws without a second collision solve."""
    if model.nflex:raise NotImplementedError('Flexible bodies need a dedicated native adapter')
    data.ncon=0;data.nefc=0
    # mj_rnePostConstraint walks `ne`, independently of `nefc`. Native
    # contacts can recycle the arena, so stale equality rows are unsafe.
    data.ne=0;data.nf=0;data.nl=0
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_camlight(model,data)
    mujoco.mj_tendon(model,data)
    mujoco.mj_transmission(model,data)
    mujoco.mj_fwdVelocity(model,data)
    mujoco.mj_sensorPos(model,data)
    mujoco.mj_sensorVel(model,data)


def passive_residual(model, data):
    force=data.qfrc_passive.copy()-data.qfrc_gravcomp
    for j,kind in enumerate(model.jnt_type):
        if kind not in (2,3):continue
        qa=model.jnt_qposadr[j];va=model.jnt_dofadr[j]
        force[va]+=model.jnt_stiffness[j]*(data.qpos[qa]-model.qpos_spring[qa])
        force[va]+=model.dof_damping[va]*data.qvel[va]
    if not np.isfinite(force).all():raise ValueError('Non-finite source passive force')
    return force
