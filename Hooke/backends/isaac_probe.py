"""Native Isaac 4.5 complete-scene probe. Experimental, never reports parity PASS.

--mode preview renders the reset scene. --mode controls runs PhysX using source
actuator commands, not recorded object poses. Open-loop replay is diagnostic;
it does not replace a closed-loop task implementation or contact qualification.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['preview', 'controls'], default='preview')
parser.add_argument('--max-steps', type=int, default=0)
parser.add_argument('--no-render', action='store_true', help='Measure dynamics without RGB readback')
args = parser.parse_args()
args.source = args.source.resolve(); args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
report = {'status': 'RUNNING', 'mode': args.mode, 'parity_qualified': False}
result_file = args.output/'result.json'
result_file.write_text(json.dumps(report))
started = time.perf_counter()
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'active_gpu': int(os.environ.get('HOOKE_ISAAC_RENDER_GPU', '6')),
    'physics_gpu': 0, 'multi_gpu': False, 'renderer': 'PathTracing', 'width': 640, 'height': 480,
    'samples_per_pixel_per_frame': 1, 'max_bounces': 2, 'max_specular_transmission_bounces': 4,
    'max_volume_bounces': 0, 'anti_aliasing': 0, 'extra_args': ['--/rtx/rendermode=PathTracing']})
try:
    import numpy as np
    from PIL import Image
    from pxr import UsdPhysics, UsdGeom
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backends.usd_scene import SceneBridge
    meta = json.loads((args.source/'scene.json').read_text())
    dt = meta['timestep_s']
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt, device='cpu')
    bridge = SceneBridge(world.stage, args.source, args.output)
    conversion = bridge.build()
    m = bridge.m
    # Apply native joint drives in USD angular units (degrees). General affine
    # tendon actuation is evaluated explicitly below, including force limits.
    drives = {}
    for j, path in bridge.joint_paths.items():
        angular = m['jnt_type'][j] == 3
        factor = np.pi/180 if angular else 1.
        drive = UsdPhysics.DriveAPI.Apply(world.stage.GetPrimAtPath(path), 'angular' if angular else 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(float(m['jnt_stiffness'][j]*factor))
        drive.CreateDampingAttr(float(m['dof_damping'][m['jnt_dofadr'][j]]*factor))
        drive.CreateTargetPositionAttr(float(m['qpos_spring'][m['jnt_qposadr'][j]]/factor))
        drives[j] = drive
    actuators = {}
    for a, trans in enumerate(m['actuator_trntype']):
        if trans == 0:
            j = int(m['actuator_trnid'][a,0]); drive = drives[j]
            factor = np.pi/180 if m['jnt_type'][j] == 3 else 1.
            if not np.allclose(m['actuator_gear'][a], [1,0,0,0,0,0]):
                raise NotImplementedError('Non-unit actuator gearing needs explicit integration.')
            kp, kv = -m['actuator_biasprm'][a,1], -m['actuator_biasprm'][a,2]
            if not np.isclose(kp, m['actuator_gainprm'][a,0]):
                raise NotImplementedError('Nonstandard affine joint actuation needs explicit integration.')
            drive.GetStiffnessAttr().Set(float(kp*factor))
            drive.GetDampingAttr().Set(float((kv+m['dof_damping'][m['jnt_dofadr'][j]])*factor))
            drive.CreateMaxForceAttr(float(max(abs(m['actuator_forcerange'][a]))) if m['actuator_forcelimited'][a] else 1e20)
            drive.GetTargetPositionAttr().Set(float(m['reset_ctrl'][a]/factor))
            actuators[a] = (j, factor)
        elif trans != 3:
            raise NotImplementedError(f'Unsupported actuator transmission {trans}')
    articulations = []
    for root in bridge.articulation_roots:
        b = root['body']
        def descends(child):
            while child and child != b:
                child = int(m['body_parentid'][child])
            return child == b
        if not any(descends(int(v)) for v in m['jnt_bodyid']):
            world.stage.GetPrimAtPath(root['root_path']).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            continue
        art = world.scene.add(SingleArticulation(prim_path=root['root_path'], name=f'art{b}'))
        articulations.append(art)
    cameras = {}
    # Capture the original front and wrist cameras, retaining their source poses.
    for name in ([] if args.no_render else dict.fromkeys(meta['task_info']['camera_mapping'].values())):
        index = meta['names']['camera'].index(name)
        cameras[index] = Camera(prim_path=bridge.camera_paths[index], resolution=(640,480))
    world.reset()
    dof_map = {}
    for art in articulations:
        ids = [int(name[1:]) for name in art.dof_names]
        tensor_view = art._articulation_view._physics_view
        armatures = np.array([[m['dof_armature'][m['jnt_dofadr'][j]] for j in ids]], dtype=np.float32)
        tensor_view.set_dof_armatures(armatures, np.array([0], dtype=np.uint32))
        np.testing.assert_allclose(tensor_view.get_dof_armatures(), armatures, atol=1e-8)
        art.set_joint_positions(np.array([m['reset_qpos'][m['jnt_qposadr'][j]] for j in ids]))
        art.set_joint_velocities(np.zeros(len(ids)))
        for local, j in enumerate(ids):
            dof_map[j] = (art, local)
    assert len(dof_map) == len(bridge.joint_paths), (len(dof_map), len(bridge.joint_paths))
    world.physics_sim_view.update_articulations_kinematic()
    from omni.physx import get_physx_interface
    get_physx_interface().update_transformations(True, True, True, False)
    for camera in cameras.values():
        camera.initialize()
        camera.set_clipping_range(.01, 100.)
    world.stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    for _ in range(0 if args.no_render else 24):
        world.render()
    def capture(frame):
        for i, camera in cameras.items():
            rgb = camera.get_rgba()
            if rgb is None or rgb.shape != (480,640,4):
                raise RuntimeError(f'No valid RGB for source camera {i}')
            folder = args.output/f'camera_{i}'; folder.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(rgb[:,:,:3], dtype=np.uint8)).save(folder/f'{frame:05d}.png')
    capture(0)
    trace = []; physics_wall = 0.
    if args.mode == 'controls':
        original = np.load(args.source/'trajectory.npz', allow_pickle=False)
        controls = original['control_before_step']
        if args.max_steps:
            controls = controls[:args.max_steps]
        stride = max(1, round(1/(20*dt)))
        for step, ctrl in enumerate(controls):
            for a,(j,factor) in actuators.items():
                value = np.clip(ctrl[a], *m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
                art, local = dof_map[j]
                art.apply_action(ArticulationAction(joint_positions=np.array([value]), joint_indices=np.array([local])))
            q = np.empty(len(dof_map)); v = np.empty(len(dof_map))
            efforts = {art.name: np.zeros(art.num_dof) for art in articulations}
            for art in articulations:
                qs, vs = art.get_joint_positions(), art.get_joint_velocities()
                for local, name in enumerate(art.dof_names):
                    q[int(name[1:])] = qs[local]; v[int(name[1:])] = vs[local]
            for a, trans in enumerate(m['actuator_trntype']):
                if trans != 3:
                    continue
                t = int(m['actuator_trnid'][a,0]); start = int(m['tendon_adr'][t]); count = int(m['tendon_num'][t])
                spans = list(range(start,start+count))
                if not all(m['wrap_type'][s] == 1 for s in spans):
                    raise NotImplementedError('Spatial tendons require another adapter.')
                js = m['wrap_objid'][spans]; weights = m['wrap_prm'][spans]
                length, velocity = np.dot(weights,q[js]), np.dot(weights,v[js])
                u = np.clip(ctrl[a], *m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
                bias = m['actuator_biasprm'][a]
                force = m['actuator_gainprm'][a,0]*u + bias[0]+bias[1]*length+bias[2]*velocity
                if m['actuator_forcelimited'][a]:
                    force = np.clip(force, *m['actuator_forcerange'][a])
                for j, weight in zip(js, weights):
                    art, local = dof_map[int(j)]; efforts[art.name][local] += force*weight
            for art in articulations:
                art.set_joint_efforts(efforts[art.name])
            tick = time.perf_counter(); world.step(render=False); physics_wall += time.perf_counter()-tick
            actual = np.empty(len(dof_map))
            for art in articulations:
                for name, value in zip(art.dof_names, art.get_joint_positions()):
                    actual[int(name[1:])] = value
            if not np.isfinite(actual).all():
                raise RuntimeError(f'Non-finite PhysX state at step {step}')
            trace.append(actual)
            if not args.no_render and (step+1) % stride == 0:
                world.render(); capture((step+1)//stride)
        np.savez_compressed(args.output/'trajectory.npz', qpos=np.asarray(trace),
                            time=np.arange(1,len(trace)+1)*dt)
        difference = np.asarray(trace)-original['qpos'][:len(trace)]
        report.update(joint_rmse=np.sqrt(np.mean(difference**2,axis=0)).tolist(),
                      joint_names=meta['names']['joint'], final_joint_qpos=trace[-1].tolist())
    report.update(status='PROBE_COMPLETED_UNQUALIFIED',
                  steps=len(trace), physics_wall_s=physics_wall, simulation_s=len(trace)*dt,
                  source_geoms=len(meta['names']['geom']), exported_geoms=len(bridge.geom_paths),
                  source_joints=len(meta['names']['joint']), simulated_dofs=len(dof_map),
                  cameras={i:meta['names']['camera'][i] for i in cameras},
                  limitations=conversion['limitations'], total_wall_s=time.perf_counter()-started)
except Exception as exc:
    report.update(status='ERROR', error=str(exc), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    result_file.write_text(json.dumps(report, indent=2))
    print('HOOKE_PROBE '+json.dumps(report), flush=True)
    app.close()
