"""Dual-arm vortex recipe using the shared instrument and actual contact state."""
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
import numpy as np
from kinematics import IK, Pose, slerp, AlohaAnalyticalIK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import VortexMixerGenie2
from liquid import ContainerSystem, ContainerCoordinator
from expert_common import set_gravcomp
from contact_state import body_geoms, touching

def quat_inverse(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def get_rotate_quat(q1, q2):
    q1_inv = quat_inverse(q1)
    qx = np.zeros(4)
    mujoco.mju_mulQuat(qx, q2, q1_inv)
    return qx

def rotate_quaternion_around_axis(base_quat, axis, angle):
    quat_rel = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat_rel, axis, angle)
    rotated_quat = np.zeros(4)
    mujoco.mju_mulQuat(rotated_quat, quat_rel, base_quat)
    return rotated_quat

class AlohaArm:
    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}left/waist'
        self.act_name = f'{prefix}left/waist'
        self.site_name = f'{prefix}left/gripper'
        self.base_name = f'{prefix}left/base_link'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_jnt_adr = model.joint(f'{prefix}left/right_finger').qposadr.item()
        self.gripper_id = model.actuator(f'{prefix}left/gripper').id
        self.nv = model.nv
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) +  [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span)+ [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        parentid = self.model.body(self.base_name).parentid.item()
        pose = Pose(data.xpos[parentid], data.xquat[parentid])
        self.ik = AlohaAnalyticalIK(pose=pose)

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def get_jac(self, data: mujoco.MjData) -> np.ndarray:
        jac = np.zeros((6, self.nv))
        mujoco.mj_jacSite(self.model, data, jac[:3], jac[3:], self.site_id)
        jac = jac[:, self.jnt_adr: self.jnt_adr + self.dof]
        return jac

    def qpos_perturb(self):
        lows = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1)
        highs = (0.2, 0.3, 0.1, 0.05, 0.3, 0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation

class CentrifugeTube:
    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str):
        self.model = model
        self.cap_id = model.body(f'{cap_prefix}centrifuge_15ml_cap').id
        self.cap_jnt_adr = model.joint(f'centrifuge_15ml_cap').qposadr.item()
        self.body_id = model.body(f'{body_prefix}centrifuge_15ml_body').id
        self.body_jnt_adr = model.joint(f'centrifuge_15ml_body').qposadr.item()
        self.cap_pos_span = range(self.cap_jnt_adr, self.cap_jnt_adr + 3)
        self.body_pos_span = range(self.body_jnt_adr, self.body_jnt_adr + 3)

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])

    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])

    def get_end_effector_pose(self, data: mujoco.MjData, random: bool = False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.005])
        pos = cap_pos + bias
        quat = np.zeros(4)
        quat_rel = np.zeros(4)
        axis = np.array([0.0, 1.0, 0.0])
        mujoco.mju_axisAngle2Quat(quat_rel, axis, np.pi / 8)
        mujoco.mju_mulQuat(quat, quat_rel, body_quat)
        return Pose(pos, quat)

class VortexMixerManipulate(Task):
    default_scene = SCENE_ROOT / "vortex_mixer.xml"
    default_task = "vortex_mixer"

    time_limit = 30.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body1 = spec.body('1/aloha:left/base_link')
        body2 = spec.body('2/aloha:left/base_link')
        set_gravcomp(body1)
        set_gravcomp(body2)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = VortexMixerGenie2("/vortex_mixer_genie_2:")
        cs = ContainerSystem("1/centrifuge_15ml_body-visual")
        cc = ContainerCoordinator()
        manager = Manager.from_spec(spec, [self.instrument, cs, cc])
        super().__init__(manager)
        self.arm1 = AlohaArm(self.model, '1/aloha:')
        self.arm2 = AlohaArm(self.model, '2/aloha:')
        self.object = CentrifugeTube(self.model, '2/', '1/')
        self.tube_geoms = body_geoms(self.model, self.object.body_id)
        joint = self.model.joint('/vortex_mixer_genie_2:platform/pivot')
        self.platform_velocity_adr = int(joint.dofadr[0])
        self.aloha2_withdraw_pose = Pose(pos=(0.0, 0.0, 0.0), quat=np.array([1.0, 0.0, 0.0, 0.0]))
        self.model.key_qpos[0] += self.model.key_qpos[1]
        self.model.key_ctrl[0] += self.model.key_ctrl[1]

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize tube position
        slot_origin_site_id = self.model.site('origin1').id
        slot_origin_pos = self.data.site_xpos[slot_origin_site_id]
        slot_origin_frame = self.data.site_xmat[slot_origin_site_id].reshape(3, 3)
        row_direction = slot_origin_frame[:, 0]
        col_direction = slot_origin_frame[:, 1]
        row = np.random.randint(2)
        col = np.random.randint(5)
        bias = 0.036 * row * row_direction + 0.036 * col * col_direction
        tube_pos = slot_origin_pos + bias
        self.data.qpos[self.object.body_pos_span] = tube_pos
        self.data.qpos[self.object.cap_pos_span] = tube_pos
        mujoco.mj_kinematics(self.model, self.data)

        self.task_info = {
            'prefix': 'dual-Aloha arms mixing the centrifuge tube on the vortex mixer: one operates test tube, the other operates the vortex mixer',
            'state_indices': self.arm1.state_indices + self.arm2.state_indices,
            'action_indices': self.arm1.action_indices + self.arm2.action_indices,
            'seed': seed,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '1/aloha:wrist_cam_left',
                'wrist_image_2': '2/aloha:wrist_cam_left'
            }
        }

        return self.task_info

    def check(self):
        # Placeholder for task completion logic
        return False

class VortexMixerManipulateExpert(VortexMixerManipulate, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm1.register_ik(self.data)
        self.arm2.register_ik(self.data)
        self.planner1 = Topp(
            dof=self.arm1.dof,
            qc_vel=0.8,
            qc_acc=0.8,
            ik=self.arm1.ik.solve
        )
        self.planner2 = Topp(
            dof=self.arm2.dof,
            qc_vel=0.8,
            qc_acc=0.8,
            ik=self.arm2.ik.solve
        )

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def _arm_and_planner(self, arm_id: int):
        if arm_id == 1:
            return self.arm1, self.planner1
        if arm_id == 2:
            return self.arm2, self.planner2
        raise ValueError(f'Unknown arm {arm_id}; expected 1 or 2')

    def wait(self, seconds: float):
        for _ in range(round(seconds / self.dt)):
            self.step_and_log({})

    def path_follow(self, path: list[Pose], arm_id: int):
        arm, planner = self._arm_and_planner(arm_id)
        trajectory = planner.jnt_traj(path)
        for step in range(int((trajectory.duration + .2) / self.dt)):
            if step % self.period == 0:
                self.data.ctrl[arm.act_span] = planner.query(trajectory, step * self.dt)
            self.step_and_log({})

    def gripper_control(self, value: float, arm_id: int):
        arm, _ = self._arm_and_planner(arm_id)
        self.data.ctrl[arm.gripper_id] = value
        self.wait(.4)

    def mix(self, seconds: float, timeout: float = 5.):
        """Require uninterrupted tube contact with the actually moving platform."""
        elapsed = continuous = 0.
        while elapsed < seconds + timeout:
            self.step_and_log({})
            active = (abs(self.data.qvel[self.platform_velocity_adr]) >= 1.
                      and touching(self.data, self.tube_geoms,
                                   self.instrument.platform_geom_id_range))
            continuous = continuous + self.dt if active else 0.
            elapsed += self.dt
            if continuous >= seconds:
                return
        raise RuntimeError('Vortex mixing failed: sustained contact with the moving platform was not reached')

    def pick_up_tube(self):
        self.gripper_control(0.030, 1)
        eef_pose = self.object.get_end_effector_pose(self.data)
        eef_pose.quat = rotate_quaternion_around_axis(eef_pose.quat, np.array([0.0, 0.0, 1.0]), np.pi * -3 / 8)
        eef_pre_pose = Pose(pos=eef_pose.pos + (0.0, 0.0, 0.04), quat=eef_pose.quat)
        cur_pose = self.arm1.get_site_pose(self.data)
        path = self.interpolate(cur_pose, eef_pre_pose, 5)
        path_ = self.interpolate(eef_pre_pose, eef_pose, 10)
        path.extend(path_[1:])
        self.path_follow(path, 1)
        self.gripper_control(0.01, 1)
        cur_pose = self.arm1.get_site_pose(self.data)
        terminal_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.2), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, terminal_pose, 20)
        self.path_follow(path, 1)

    def place_on_platform(self):
        site_name = "/vortex_mixer_genie_2:platform-function"
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Missing vortex interaction site: {site_name}")
        pos_cen_devia = self.object.get_cap_pose(self.data).pos - self.object.get_body_pose(self.data).pos
        quat1 = self.arm1.get_site_pose(self.data).quat.copy()
        quat1 = rotate_quaternion_around_axis(quat1, np.array([0.0, 0.0, 1.0]), np.pi / 6 * (-1.0))
        end_pose = Pose(pos=self.data.site_xpos[site_id] + pos_cen_devia - (0.0, 0.0, 0.005), quat=quat1.copy())
        mid_end_pose = Pose(pos=self.data.site_xpos[site_id] + pos_cen_devia + (0., 0., 0.05), quat=quat1.copy())
        cur_pose = self.arm1.get_site_pose(self.data)
        path = self.interpolate(cur_pose, mid_end_pose, 5)
        path_ = self.interpolate(mid_end_pose, end_pose, 10)
        path.extend(path_[1:])
        self.path_follow(path, 1)

    def _switch_frame(self):
        site = self.model.site('/vortex_mixer_genie_2:switch-function').id
        rotation = self.data.site_xmat[site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        quat = rotate_quaternion_around_axis(quat, rotation[:, 2], np.pi / 2)
        quat = rotate_quaternion_around_axis(quat, rotation[:, 1], np.pi)
        return self.data.site_xpos[site].copy(), rotation[:, 1], rotation[:, 2], quat

    def _move_via(self, arm_id: int, poses: list[Pose]):
        arm, _ = self._arm_and_planner(arm_id)
        path = [arm.get_site_pose(self.data)]
        for pose in poses:
            path.extend(self.interpolate(path[-1], pose, 10)[1:])
        self.path_follow(path, arm_id)

    def turn_on(self):
        position, y_axis, z_axis, quat = self._switch_frame()
        self.aloha2_withdraw_pose = Pose(position - y_axis * .06, quat)
        self._move_via(2, [
            Pose(position - y_axis * .05 + z_axis * .02, quat),
            Pose(position - y_axis * .01 + z_axis * .02, quat),
            Pose(position - y_axis * .01, quat),
            self.aloha2_withdraw_pose,
        ])
        self.gripper_control(.030, 2)

    def set_gear(self, gear: int = 1):
        self.gripper_control(0.030, 2)
        site_name = "/vortex_mixer_genie_2:knob-function"
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Missing vortex interaction site: {site_name}")
        quat1 = np.zeros(4)
        mujoco.mju_mat2Quat(quat1, self.data.site_xmat[site_id])
        site_rot_mat = self.data.site_xmat[site_id].reshape(3, 3)
        axis_choose = np.array([0.0, 1.0, 0.0])
        axis_local = np.dot(site_rot_mat, axis_choose)
        quat1 = rotate_quaternion_around_axis(quat1, axis_local, np.pi / 2)
        axis_choose = np.array([0.0, 0.0, 1.0])
        axis_z = np.dot(site_rot_mat, axis_choose)
        cur_pose = self.arm2.get_site_pose(self.data)
        mid_end_pose1 = Pose(pos=self.data.site_xpos[site_id] + axis_z * 0.03, quat=quat1.copy())
        mid_end_pose2 = Pose(pos=self.data.site_xpos[site_id] + axis_z * 0.013, quat=quat1.copy())
        path1 = self.interpolate(cur_pose, mid_end_pose1, 10)
        path1_ = self.interpolate(mid_end_pose1, mid_end_pose2, 10)
        path1.extend(path1_[1:])
        self.path_follow(path1, 2)
        self.gripper_control(0.008, 2)
        if gear < 0 or gear > 10:
            raise ValueError("Gear must be between 0 and 10")
        else:
            angle = gear * np.pi * 2 / 14
            wrist_id = self.model.actuator(f'2/aloha:left/wrist_rotate').id
            for n in range(50):
                self.data.ctrl[wrist_id] += angle / 50
                for _ in range(20):
                    self.step_and_log({})
        self.gripper_control(0.02, 2)
        cur_pose = self.arm2.get_site_pose(self.data)
        path2 = self.interpolate(cur_pose, self.aloha2_withdraw_pose, 10)
        self.path_follow(path2, 2)

    def turn_off(self):
        self.gripper_control(.005, 2)
        position, y_axis, z_axis, quat = self._switch_frame()
        self.aloha2_withdraw_pose = Pose(position - y_axis * .06, quat)
        self._move_via(2, [
            Pose(position - y_axis * .01, quat),
            Pose(position - y_axis * .01 + z_axis * .02, quat),
            self.aloha2_withdraw_pose,
        ])
        self.gripper_control(.030, 2)

    def return_tube(self):
        site_name = "/vortex_mixer_genie_2:platform-function"
        site_name1 = "origin1"
        site_name2 = "centrifuge_body"
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        hole_id1 = mujoco.mj_name2id(self.data.model, mujoco.mjtObj.mjOBJ_SITE, site_name1)
        cen_id = mujoco.mj_name2id(self.data.model, mujoco.mjtObj.mjOBJ_SITE, site_name2)
        if site_id == -1 or hole_id1 == -1 or cen_id == -1:
            raise ValueError('Missing vortex return reference site')
        cur_pose = self.arm1.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
        path1 = self.interpolate(cur_pose, end_pose, 5)
        self.path_follow(path1, 1)
        hole_quat = np.zeros(4)
        cen_quat = np.zeros(4)
        bias_quat = np.zeros(4)
        compensate_cur_quat = np.zeros(4)
        for _ in range(4):
            cur_pose = self.arm1.get_site_pose(self.data)
            mujoco.mju_mat2Quat(hole_quat, self.data.site_xmat[hole_id1])
            mujoco.mju_mat2Quat(cen_quat, self.data.site_xmat[cen_id])
            bias_quat = get_rotate_quat(cen_quat, hole_quat)
            mujoco.mju_mulQuat(compensate_cur_quat, bias_quat, cur_pose.quat)
            bias_pos = np.zeros(3)
            bias_pos = self.data.site_xpos[hole_id1] - self.data.site_xpos[cen_id]
            if _ == 0:
                bias_pos[2] += 0.18
            elif _ == 1:
                bias_pos[2] += 0.1
            elif _ == 2:
                bias_pos[2] += 0.06
            elif _ == 3:
                bias_pos[:2] = 0.
                bias_pos[2] += 0.
            compensate_cur_pos = cur_pose.pos + bias_pos
            compensate_cur_pose = Pose(pos=compensate_cur_pos, quat=compensate_cur_quat)
            path2 = self.interpolate(cur_pose, compensate_cur_pose, 10)
            self.path_follow(path2, 1)
        self.gripper_control(0.030, 1)
        cur_pose = self.arm1.get_site_pose(self.data)
        New_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.15), quat=cur_pose.quat)
        path2 = self.interpolate(cur_pose, New_pose, 10)
        self.path_follow(path2, 1)


    def execute(self, gear: int = 3, mix_seconds: float = 1.):
        if not isinstance(gear, int) or not 1 <= gear <= 10:
            raise ValueError("Mixing gear must be an integer from 1 to 10")
        if not np.isfinite(mix_seconds) or mix_seconds < .5:
            raise ValueError("Mixing duration must be finite and at least 0.5 seconds")
        self.arm1.ik.initial_qpos = self.data.qpos[self.arm1.jnt_span]
        self.arm2.ik.initial_qpos = self.data.qpos[self.arm2.jnt_span]
        self.pick_up_tube()
        self.place_on_platform()
        self.turn_on()
        self.set_gear(gear)
        self.mix(mix_seconds)
        self.turn_off()
        self.return_tube()
        self.finish()

VortexMixerManipulate.Expert = VortexMixerManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = VortexMixerManipulate.load()
    expert = VortexMixerManipulate.Expert(spec)
    for i in trange(3):
        expert.reset(i)
        expert.set_serializer()
        expert.execute(3)
