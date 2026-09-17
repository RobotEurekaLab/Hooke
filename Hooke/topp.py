from typing import Callable, List
from dataclasses import dataclass
import numpy as np
import toppra as ta
from kinematics import Pose

@dataclass
class HoldTrajectory:
    """A zero-duration Cartesian request still supplies a valid hold target."""
    qpos: np.ndarray
    duration: float = 0.0

    @property
    def dof(self):
        return self.qpos.size

    @property
    def path_interval(self):
        return np.array([0., 0.])

    def eval(self, t):
        return np.broadcast_to(self.qpos, np.shape(t) + self.qpos.shape).copy()

    def evald(self, t):
        return np.zeros(np.shape(t) + self.qpos.shape)

    def evaldd(self, t):
        return np.zeros(np.shape(t) + self.qpos.shape)


class Topp:
    # re-parameterize trajectory using TOPP-RA

    def __init__(self, dof: int, qc_vel: float, qc_acc: float, ik: Callable):
        self.dof = dof
        self.qc_vel = [(-qc_vel, qc_vel)] * self.dof
        self.qc_vel = ta.constraint.JointVelocityConstraint(np.array(self.qc_vel))
        self.qc_acc = [(-qc_acc, qc_acc)] * self.dof
        self.qc_acc = ta.constraint.JointAccelerationConstraint(np.array(self.qc_acc))
        self.ik: Callable = ik
    
    def jnt_traj(self, pose_path: List[Pose],):
        assert self.ik is not None, "IK solver not set"
        if not pose_path:
            raise ValueError("Cannot plan an empty pose path")
        first = pose_path[0]
        same_pose = all(
            np.allclose(pose.pos, first.pos, rtol=0, atol=1e-12)
            and (np.allclose(pose.quat, first.quat, rtol=0, atol=1e-12)
                 or np.allclose(pose.quat, -np.asarray(first.quat), rtol=0, atol=1e-12))
            for pose in pose_path)
        if same_pose:
            # Repeated IK solves of an identical target can drift slightly,
            # producing an ill-conditioned spline that TOPPRA cannot time.
            # Solve once and let the caller's normal settling interval hold it.
            return HoldTrajectory(np.asarray(self.ik(first.pos, first.quat), dtype=float).copy())
        jnts = [self.ik(pose.pos, pose.quat) for pose in pose_path]
        return self._retime(jnts)

    def joint_traj(self, joint_path):
        """Time joint waypoints with the same velocity/acceleration constraints."""
        jnts = np.asarray(joint_path, dtype=float)
        if jnts.ndim != 2 or not len(jnts) or jnts.shape[1] != self.qc_vel.dof or not np.isfinite(jnts).all():
            raise ValueError('Joint path must contain finite waypoints of the configured size')
        if np.allclose(jnts, jnts[0], rtol=0., atol=1e-12):
            return HoldTrajectory(jnts[0].copy())
        return self._retime(jnts, parametrizer='ParametrizeConstAccel')

    def _retime(self, jnts, parametrizer='ParametrizeSpline'):
        ss = np.linspace(0, 1, len(jnts))
        path = ta.SplineInterpolator(ss, jnts)
        instance = ta.algorithm.TOPPRA([self.qc_vel, self.qc_acc], path, parametrizer=parametrizer)
        trajectory = instance.compute_trajectory(0, 0)
        if trajectory is None:
            raise ValueError('Joint path could not be timed within the motion constraints')
        return trajectory

    @staticmethod
    def query(traj: ta.interpolator.AbstractGeometricPath, t: float):
        t = np.clip(t, 0, traj.duration)
        return traj.eval(t)
