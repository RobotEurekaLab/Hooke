from dataclasses import dataclass
import numpy as np
from grasp.hierarchy import build_hierarchy
from grasp.quat import quatapply, quatinv, quatcompose
import mujoco
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')
import jax.numpy as jnp
from scipy.optimize import minimize

@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray

def mul_pose(p1: Pose, p2: Pose) -> Pose:
    res_pos, res_quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(
        res_pos, res_quat,
        p1.pos, p1.quat,
        p2.pos, p2.quat
    )
    return Pose(res_pos, res_quat)

def neg_pose(p: Pose) -> Pose:
    res_pos, res_quat = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(res_pos, res_quat, p.pos, p.quat)
    return Pose(res_pos, res_quat)

def align_axes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation between axes, retaining unconstrained axial twist."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    for axis in (source, target):
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError('Axis must be a finite, nonzero 3D vector')
    source, target = source/np.linalg.norm(source), target/np.linalg.norm(target)
    scalar = 1.+float(np.clip(np.dot(source, target), -1., 1.))
    if scalar < 1e-10:
        basis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, basis)
        return np.r_[0., axis/np.linalg.norm(axis)]
    quat = np.r_[scalar, np.cross(source, target)]
    return quat/np.linalg.norm(quat)

def slerp(q0: np.ndarray, q1: np.ndarray, amount=0.5):
    """Spherical Linear Interpolation between quaternions.
    Implemented as described in https://en.wikipedia.org/wiki/Slerp

    Find a valid quaternion rotation at a specified distance along the
    minor arc of a great circle passing through any two existing quaternion
    endpoints lying on the unit radius hypersphere.

    This is a class method and is called as a method of the class itself rather than on a particular instance.

    Params:
        q0: first endpoint rotation as a Quaternion object
        q1: second endpoint rotation as a Quaternion object
        amount: interpolation parameter between 0 and 1. This describes the linear placement position of
            the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
            Defaults to the midpoint (0.5).

    Returns:
        A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.

    Note:
        This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere).
            Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.
    """
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1

    amount = np.clip(amount, 0, 1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        qr /= np.linalg.norm(qr)
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr /= np.linalg.norm(qr)
    return qr


# How close an IK solve's residual (squared position error in m^2 plus a
# quaternion-alignment term) must get before being accepted as "reached".
# Was hardcoded to 1e-6 -- fine for UR5e's original recipes (which always
# converged far below it) but too strict once other targets entered the
# picture: xArm7/Kinova/Lite6 waypoints plateau around 1e-2 (a genuine
# reachability issue worth its own calibration, see private/TODO.md), but
# several UR5e targets in the reagent-bottle pickup task (a different
# region of its workspace/orientation than the original recipes were
# tuned for -- a horizontal side-grasp, not the original downward-reach
# ones) kept landing at 2e-4 to 1.4e-3: correct in every practical sense
# (a few mm/degrees of residual) but rejected by the old bound. 2e-3
# comfortably covers those cases while staying one to two orders of
# magnitude tighter than the genuine near-misses above, so it doesn't
# quietly accept those too.
_CONVERGENCE_TOL = 2e-3


class IK:
    def __init__(self, dof: int, model, data, root: str, site: str):
        self.dof = dof
        self.model = model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        hierarchy = build_hierarchy(model, root)
        self.hierarchy = hierarchy
        site = hierarchy.site_name2id[site]
        self.site = site
        self.initial_qpos = np.zeros(dof)

        # if pose is None:
        #     pos = np.zeros(3)
        #     quat = np.array([1.0, 0., 0., 0.])
        # else:
        #     pos = pose.pos
        #     quat = pose.quat
        base = model.body_parentid[root]
        self.pos = data.xpos[base]
        self.quat = data.xquat[base]

        def objective(qpos: jax.Array, target_pos: jax.Array, target_quat: jax.Array, *, target_site: int):
            assert len(qpos) == self.dof
            _, _, site_poses = hierarchy.resolve_pose(qpos)
            site_pose = site_poses[target_site]
            site_pos = site_pose.pos
            site_quat = site_pose.quat
            pos_error = jnp.sum((site_pos - target_pos) ** 2)
            quat_error = 1 - (site_quat @ target_quat) ** 2
            return pos_error + quat_error
        
        def gradient(qpos: jax.Array, target_pos: jax.Array, target_quat: jax.Array, *, target_site: int):
            grad_fn = jax.grad(objective, argnums=0)
            return grad_fn(qpos, target_pos, target_quat, target_site=target_site)
        objective = jax.jit(objective, static_argnames=('target_site',))
        gradient = jax.jit(gradient, static_argnames=('target_site',))
        def objective_np(qpos: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray):
            return objective(qpos, target_pos, target_quat, target_site=site).item()
        def gradient_np(qpos: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray):
            return jax.device_get(
                gradient(qpos, target_pos, target_quat, target_site=site)
            ).astype(np.float64)
        self.objective_np = objective_np
        self.gradient_np = gradient_np
    
    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        initial_qpos = self.initial_qpos
        target_pos = quatapply(quatinv(self.quat), target_pos - self.pos)
        target_quat = quatcompose(quatinv(self.quat), target_quat)

        # Was hardcoded to [:6], silently correct only for a 6-DOF arm (UR5e,
        # the only one this ever drove until now) -- self.dof already carries
        # the real chain length, so generalizing this is what actually makes
        # IK usable for a 7-DOF arm like the Panda/xArm7.
        bound = self.hierarchy.bounds[:self.dof]

        # Single-start L-BFGS-B warm-started from the previous solve. That's
        # enough for UR5e's recipes (never observed to fail below), but a
        # 7-DOF arm with different link geometry can warm-start into a local
        # basin that stalls just above the threshold (e.g. xArm7 at a
        # waypoint tuned for UR5e's dimensions: sln.fun ~1.8e-4, "success"
        # but not actually converged). Retrying from fresh, randomized
        # starting points is a generic fix for that -- not a per-robot
        # hack -- and only ever triggers on the failure path, so it can't
        # change any already-passing (e.g. UR5e) result.
        sln = minimize(
            fun=self.objective_np,
            x0=initial_qpos,
            args=(target_pos, target_quat),
            jac=self.gradient_np,
            bounds=bound,
        )
        if not sln.success or sln.fun >= _CONVERGENCE_TOL:
            rng = np.random.default_rng(0)
            lo = np.array([b[0] for b in bound])
            hi = np.array([b[1] for b in bound])
            for _ in range(20):
                x0 = rng.uniform(lo, hi)
                retry = minimize(
                    fun=self.objective_np,
                    x0=x0,
                    args=(target_pos, target_quat),
                    jac=self.gradient_np,
                    bounds=bound,
                )
                if retry.success and retry.fun < _CONVERGENCE_TOL:
                    sln = retry
                    break
                if retry.success and retry.fun < sln.fun:
                    sln = retry
        assert sln.success, f"IK failed: {sln.message} @ {target_pos}, {target_quat}"
        assert sln.fun < _CONVERGENCE_TOL, f"Near unreachable: {sln.fun} @ {target_pos.tolist()}, {target_quat.tolist()}, {initial_qpos.tolist()}, {sln.x.tolist()}"
        self.initial_qpos = sln.x
        return sln.x

class AlohaAnalyticalIK:
    def __init__(self, pose: Pose = None):
        if pose is None:
            pos = np.zeros(3)
            quat = np.array([1.0, 0., 0., 0.])
        else:
            pos = pose.pos
            quat = pose.quat
        assert pos.shape == (3,) and quat.shape == (4,), "pos and quat must be 3D and 4D vectors respectively"
        self.pos = pos
        self.quat = quat
        self.initial_qpos = np.zeros(6)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        from aloha_analytical_ik import aloha_analytical_ik

        target_pos = quatapply(quatinv(self.quat), target_pos - self.pos)
        target_quat = quatcompose(quatinv(self.quat), target_quat)

        def format_error(e):
            return f"{e}: {target_pos.tolist()}, {target_quat.tolist()}, {self.initial_qpos.tolist()}"

        try:
            solutions = aloha_analytical_ik(target_pos, target_quat)
        except NotImplementedError:
            raise ValueError(format_error("Edge case detected"))

        if solutions is None or len(solutions) == 0:
            raise ValueError(format_error("No valid IK solutions found"))

        # Solution selection
        dists = np.linalg.norm(solutions - self.initial_qpos, axis=1)
        solution = solutions[np.argmin(dists)]

        if solution[4] == 0.0:
            print(format_error("Gimbal lock detected"))

        self.initial_qpos = solution
        return solution


class FK:
    # General forward kinematics
    def __init__(self, dof: int, model, data, root: str, site: str):
        self.dof = dof
        self.model = model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        base = model.body_parentid[root]
        hierarchy = build_hierarchy(model, root)
        self.hierarchy = hierarchy
        site = hierarchy.site_name2id[site]
        self.site = site
        self.basepose = Pose(
            pos=data.xpos[base],
            quat=data.xquat[base]
        )
    
    def forward(self, qpos: np.ndarray) -> Pose:
        assert len(qpos) == self.dof
        _, _, site_poses = self.hierarchy.resolve_pose(qpos)
        site_pose = site_poses[self.site]
        res_pos = np.zeros(3)
        res_quat = np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            self.basepose.pos, self.basepose.quat,
            site_pose.pos, site_pose.quat
        )
        return Pose(res_pos, res_quat)
