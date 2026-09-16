import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
import numpy as np
from kinematics import IK, Pose, slerp
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from liquid import ContainerSystem, ContainerCoordinator
from pipetting import PipetteTransferSystem
from process_progress import ProcessProgressSystem

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

class UR5eArm:

    # 6-DOF arm

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}shoulder_pan'
        self.act_name = f'{prefix}shoulder_pan'
        self.base_name = f'{prefix}base'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        if prefix == "2/ur:":
            self.site_name = f'{prefix}2f85:pinch'
            self.gripper_id = model.actuator(f'{prefix}2f85:fingers_actuator').id
            self.gripper_jnt_adr = model.joint(f'{prefix}2f85:right_driver_joint').qposadr.item()
            self.site_id = model.site(self.site_name).id
            self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
            self.action_indices = list(self.act_span) + [self.gripper_id]
        else:
            self.site_name = 'tl/tip_site'
            self.site_id = model.site(self.site_name).id
            self.hand_jnt_adr = model.joint('1/ur:dh:rh_thj1').qposadr.item()
            self.hand_id = model.actuator('1/ur:dh:rh_thj1').id
            self.thj3_id = model.actuator('1/ur:dh:rh_thj3').id
            self.thj3_qposadr = model.joint('1/ur:dh:rh_thj3').qposadr.item()
            self.state_indices = list(self.jnt_span) + [self.thj3_qposadr]
            self.action_indices = list(self.act_span) + [self.thj3_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)
    
    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self):
        lows = (-0.01, 0.0, -0.1, -0.02, 0.0, -0.1)
        highs = (0.01, 0.1, 0.0, 0.02, 0.1,  0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation
    
class CentrifugeTube:

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.body_id_50 = model.body('5/centrifuge_50ml_screw_body').id
        self.body_jnt_adr_50 = model.joint('centrifuge_50ml_body').qposadr.item()
        self.body_id = self.body_id_50
        self.body_jnt_adr = self.body_jnt_adr_50
        self.body_pos_span = range(self.body_jnt_adr, self.body_jnt_adr + 3)
        self.tube_params = {
            '50ml': {
                'rows': 5,
                'columns': 2,
                'x0': -0.072,
                'y0': -0.018,
                'delta_x': 0.036,
                'delta_y': 0.036
            }
        }
        self.x = 0
        self.y = 0

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def randomposition(self, data: mujoco.MjData):
        params = self.tube_params['50ml']
        row_idx = np.random.randint(0, params['rows'])
        col_idx = np.random.randint(0, params['columns'])
        x = params['x0'] + row_idx * params['delta_x']
        y = params['y0'] + col_idx * params['delta_y']
        return [x, y]

    def get_eefpose_lever(self, sitepose: Pose, mode: str='tube') -> Pose:
        rel_quat = np.array([-0.379914933420129, -0.59635140006596, -0.59635140006596, 0.379914933420129])
        match mode:
            case 'tube':
                rel_pos = np.array([0.0, 0.0, 0.08])
            case _:
                raise ValueError(f"Unknown approach mode: {mode}")
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)

    @staticmethod
    def randomsphere(x, y, z, d):
        radius = d / 2.0
        while True:
            x_dir, y_dir, z_dir = np.random.normal(0, 1, size=3)
            norm = np.linalg.norm([x_dir, y_dir, z_dir])
            if norm > 0:
                break
        x_unit = x_dir / norm
        y_unit = y_dir / norm
        z_unit = z_dir / norm
        u = np.random.uniform(0, 1)
        d = radius * (u ** (1/3))
        x = x + x_unit * d
        y = y + y_unit * d
        z = z + z_unit * d
        return np.array([x, y, z])      

class Pipette(Task):
    default_scene = SCENE_ROOT / "mani_pipette.xml"
    default_task = "pipette"
    target_reservoir = 'tip'

    time_limit = 30.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        pipette_spec = mujoco.MjSpec.from_file("model/object/pipette.gen.xml")
        body_name1 = f'1/ur:world'
        body_name2 = f'2/ur:world'
        body1 = spec.body(body_name1)
        body2 = spec.body(body_name2)
        pipette_body = pipette_spec.body("pipette")
        hand_body = spec.body("1/ur:dh:rh_base")
        pipette_body.pos = [0, 0, 0.08]
        pipette_body.quat = [0, 0, 0, 1]
        frame = hand_body.add_frame(pos=[0.05022559, -0.18, 0.16], quat=[0., 0., 0.70710678, 0.70710678])
        frame.attach_body(pipette_body, "tl/", "")
        set_gravcomp(body1)
        set_gravcomp(body2)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.container = ContainerSystem("5/centrifuge_50ml_screw_body-visual")
        self.destinations = self.make_destinations()
        self.liquid_transfer = PipetteTransferSystem(source=self.container, tip_site='tl/tip_site',
                                                     plunger_joint='tl/pipette_button', tip_capacity_m3=200e-9,
                                                     destinations=self.destinations,
                                                     target_reservoir=self.target_reservoir)
        cc = ContainerCoordinator()
        self.progress = ProcessProgressSystem()
        manager = Manager.from_spec(spec, [self.container, *self.destinations.values(),
                                           self.liquid_transfer, cc, self.progress])
        super().__init__(manager)
        self.arm1 = UR5eArm(self.model, '1/ur:')
        self.arm2 = UR5eArm(self.model, '2/ur:')
        self.object = CentrifugeTube(self.model)

        self.model.key_qpos[0, self.arm1.hand_jnt_adr: self.arm1.hand_jnt_adr + 19] = [0.        , 0.        , 0.        , 0.        , 0.        ,
                1.3       , 0.80592188, 0.80592188, 1.3       , 0.87486328,
                0.87486328, 0.        , 1.3       , 0.80580566, 0.80580566,
                0.        , 1.3        , 0.80        , 0.80        ]
        self.model.key_ctrl[0, self.arm1.hand_id: self.arm1.hand_id + 12] = [0.        , 0.        , 0.        , 0.        , 1.3       ,
                1.00592188, 1.3       , 1.07486328, 1.3       , 1.00580566,
                1.3        , 1.0        ]
        self.model.key_qpos[0,self.arm1.hand_jnt_adr: self.arm1.hand_jnt_adr + 19] += self.model.key_qpos[1,self.arm1.hand_jnt_adr: self.arm1.hand_jnt_adr + 19]
        self.model.key_ctrl[0,self.arm1.hand_id: self.arm1.hand_id + 12] += self.model.key_ctrl[1,self.arm1.hand_id: self.arm1.hand_id + 12]
        self.model.key_qpos[0, 28:34] = [0.         ,-1.5708, 1.5708 ,    -1.5708,     -1.5708,     -1.5708]
        self.model.key_ctrl[0, 18:24] = [0.         ,-1.5708      ,1.5708     ,-1.5708     ,-1.5708     ,-1.5708]
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "2/ur:wrist_cam")
        self.model.cam_quat[cam_id] = [-0.395128979076749, -0.586406935472579, -0.395128979076749, 0.586406935472579]
        self.model.cam_pos[cam_id][0] -= 0.18
        self.progress.bind(self)

    def make_destinations(self):
        return {}

    def compute_liquid_height(self, data: mujoco.MjData):
        container = self.container.container
        container_pos = container.position
        container_mat = container.rotation_matrix
        body_pos = data.xpos[self.object.body_id]
        body_pos_in_container = container_mat.T @ (body_pos - container_pos)

        return container.liquid.surface.distance - body_pos_in_container @ container.liquid.surface_normal

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)

        initial_volume = np.random.uniform(15e-6, 45e-6)  # 15mL to 45mL
        self.container.initial_volume = initial_volume

        self.manager.reset(keyframe=0)


        # Randomize the arm joint positions
        perturbation = self.arm1.qpos_perturb()
        self.data.qpos[self.arm1.jnt_span] += perturbation
        self.data.ctrl[self.arm1.act_span] += perturbation
        perturbation = self.arm2.qpos_perturb()
        self.data.qpos[self.arm2.jnt_span] += perturbation
        self.data.ctrl[self.arm2.act_span] += perturbation

        # Randomize the tube position
        self.data.qpos[self.object.body_jnt_adr:self.object.body_jnt_adr+2] = self.object.randomposition(self.data)

        self.task_info = {
            'prefix': 'dual-UR5e pipetting: one arm lifts centrifuge tube, the other aligns pipette tip and aspirates liquid',
            'state_indices': self.arm1.state_indices + self.arm2.state_indices,
            'action_indices': self.arm1.action_indices + self.arm2.action_indices,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '1/ur:wrist_cam',
                'wrist_image_2': '2/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        return self.progress.success

class PipetteRecipe:
    """Robot motion and actual thumb actuation shared by pipetting tasks."""
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm1.register_ik(self.data)
        self.arm2.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm1.dof,
            qc_vel=2.0,
            qc_acc=1.5,
            ik=self.arm1.ik.solve
        )
        self.withdrawal_planner = Topp(
            dof=self.arm1.dof,
            qc_vel=0.5,
            qc_acc=0.5,
            ik=self.arm1.ik.solve
        )

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def path_follow(self, path: list[Pose], arm: UR5eArm, planner: Topp | None = None):
        planner = self.planner if planner is None else planner
        planner.ik = arm.ik.solve
        trajectory = planner.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = planner.query(trajectory, t)
                self.data.ctrl[arm.act_span] = ctrl
            self.step_and_log({})

    def move_to(self, pose: Pose, arm: UR5eArm, num_steps: int=2, planner: Topp | None = None):
        cur_pos = arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path, arm, planner)

    def gripper_control(self, value: float, arm: UR5eArm):
        self.data.ctrl[arm.gripper_id] = value
        for _ in range(300):
            self.step_and_log({})

    def pipette_ctrl(self, mode: str):
        match mode:
            case 'push':
                self.data.ctrl[self.arm1.thj3_id] = 0.8
                for _ in range(150):
                    self.step_and_log({})
            case 'pull':
                self.data.ctrl[self.arm1.thj3_id] = 0.8
                for step in range(1600):
                    progress = step / 1599
                    # Clear the spring-loaded button before lifting the tool.
                    self.data.ctrl[self.arm1.thj3_id] = 0.8 * (1-progress) + 0.2 * progress
                    self.step_and_log({})


    def aspirate(self):
        self.arm1.ik.initial_qpos = self.data.qpos[self.arm1.jnt_span]
        self.arm2.ik.initial_qpos = self.data.qpos[self.arm2.jnt_span]
        self.step_and_log({})
        tube_pos = self.object.get_body_pose(self.data)
        eef_pose = self.object.get_eefpose_lever(tube_pos, mode='tube')
        self.source_grasp_pose = eef_pose
        target_quat1 = np.array([0, 0., 0., 1])
        tube_pos_random_1 = Pose(pos=self.object.randomsphere(0.0, 0.0, 1.15, 0.02), quat=target_quat1)
        cur_pose_1 = self.arm1.get_site_pose(self.data)
        cur_pose_2 = self.arm2.get_site_pose(self.data)
        path_2_1 = self.interpolate(cur_pose_2, eef_pose, 5)

        self.path_follow(path_2_1, self.arm2)
        self.gripper_control(225, self.arm2)
        height = np.array([0.0 ,0.0 ,0.1])
        lift_pose = Pose(pos=self.arm2.get_site_pose(self.data).pos + (0, 0, 0.08), quat=eef_pose.quat)
        tube_pos_random_2 = Pose(pos=tube_pos_random_1.pos - height, quat=eef_pose.quat)
        path_2_2 = self.interpolate(self.arm2.get_site_pose(self.data), lift_pose, 5)
        path_2_3 = self.interpolate(lift_pose, tube_pos_random_2, 5)
        path_2_2.extend(path_2_3[1:])
        self.path_follow(path_2_2, self.arm2)
        path_1_1 = self.interpolate(cur_pose_1, tube_pos_random_1, 5)
        self.path_follow(path_1_1, self.arm1)

        self.pipette_ctrl(mode='push')
        desired_height = self.compute_liquid_height(self.data) - 0.01
        descent_height = np.array([0.0 ,0.0 ,desired_height])
        descent_pose = Pose(pos=self.object.get_body_pose(self.data).pos + descent_height, quat=target_quat1)
        self.move_to(descent_pose, self.arm1, 5)
        self.pipette_ctrl(mode='pull')
        final_pose = Pose(pos=self.arm1.get_site_pose(self.data).pos + height, quat=target_quat1)
        self.move_to(final_pose, self.arm1, 5, self.withdrawal_planner)

    def execute(self):
        self.aspirate()
        self.finish()


class PipetteExpert(PipetteRecipe, Pipette, Expert):
    pass


class PipetteTransfer(Pipette):
    """Reuse the pipette, tube and rack assets for source-to-destination transfer."""

    default_task = 'pipette_transfer'
    target_reservoir = 'destination'
    time_limit = 45.

    @classmethod
    def prepare(cls, spec):
        spec = super().prepare(spec)
        tube_spec = mujoco.MjSpec.from_file(str(SCENE_ROOT.parent/'object/centrifuge_50ml_screw.xml'))
        rack_spec = mujoco.MjSpec.from_file(str(SCENE_ROOT.parent/'object/centrifuge_10slot.gen.xml'))
        rack = spec.worldbody.add_body(name='pipette_destination_rack', pos=[-.2, -.25, .854], quat=[1, 0, 0, 1])
        rack.add_frame().attach_body(rack_spec.body('centrifuge_10slot'), '7/', '')
        receiver = spec.worldbody.add_body(name='pipette_destination', pos=[-.236, -.268, .829])
        receiver.add_joint(name='pipette_destination', type=mujoco.mjtJoint.mjJNT_FREE)
        receiver.add_frame().attach_body(tube_spec.body('centrifuge_50ml_screw_body'), '6/', '')
        return spec

    def make_destinations(self):
        return {'destination': ContainerSystem('6/centrifuge_50ml_screw_body-visual', initial_volume=0.)}

    def reset(self, seed=None):
        info = super().reset(seed)
        mujoco.mj_forward(self.model, self.data)
        self.source_return_position = self.data.xpos[self.object.body_id].copy()
        self.destination_position = self.data.xpos[self.model.body('6/centrifuge_50ml_screw_body').id].copy()
        info['prefix'] = 'dual-UR5e pipetting: aspirate 200 uL, transfer and dispense into an empty tube, withdraw and return the source'
        return info


class PipetteTransferExpert(PipetteRecipe, PipetteTransfer, Expert):
    def execute(self):
        self.phase_history = []

        def phase(name, action):
            start = float(self.data.time)
            action()
            self.phase_history.append(dict(phase=name, start_s=start, end_s=float(self.data.time)))

        phase('aspirate', self.aspirate)
        receiver_id = self.model.body('6/centrifuge_50ml_screw_body').id
        position = self.data.xpos[receiver_id].copy()
        quat = self.arm1.get_site_pose(self.data).quat.copy()
        hover = Pose(position + (0., 0., .32), quat)
        dispense = Pose(position + (0., 0., .09), quat)
        phase('transfer', lambda: self.move_to(hover, self.arm1, 5, self.withdrawal_planner))
        phase('enter_destination', lambda: self.move_to(dispense, self.arm1, 5, self.withdrawal_planner))
        phase('dispense', lambda: self.pipette_ctrl('push'))
        # Keep the button pressed until the tip leaves the receiver. Releasing
        # under its new liquid surface would aspirate the delivered sample.
        phase('withdraw_destination', lambda: self.move_to(hover, self.arm1, 5, self.withdrawal_planner))
        phase('release_in_air', lambda: self.pipette_ctrl('pull'))

        def return_source():
            raised = Pose(self.source_grasp_pose.pos + (0., 0., .08), self.source_grasp_pose.quat)
            self.move_to(raised, self.arm2, 5)
            self.move_to(self.source_grasp_pose, self.arm2, 5)
            self.gripper_control(0., self.arm2)
            self.move_to(raised, self.arm2, 5)

        phase('return_source', return_source)
        self.finish()

Pipette.Expert = PipetteExpert
PipetteTransfer.Expert = PipetteTransferExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = Pipette.load()
    expert = Pipette.Expert(spec)
    for i in trange(100):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
