"""Render recorded phase endpoints without executing a policy or advancing physics.

Run from Hooke/: python -m backends.keyframes --episode <expert-folder>
    --phases aspirate dispense return_source --output <new-folder> --gpu 6
Joint states and recorded liquid surfaces are restored; other dynamic
instrument overlays are outside this replay's scope.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import cache
import json
import os
from pathlib import Path
import pickle
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image

from backends.evidence import file_sha256
from backends.visual_state import ellipse_frame, LiveVisuals
from backends.worker_client import IsaacWorker

ROOT = Path(__file__).resolve().parents[2]


@cache
def load_replay_plugins():
    """Register the repository's SDF types before reading cached model binaries."""
    mujoco.mj_loadPluginLibrary(str(ROOT / 'Hooke/libmjlab.so.3.3.0'))


def load_replay_model(path):
    """Load a recorded binary after registering the plugins it may reference."""
    load_replay_plugins()
    return mujoco.MjModel.from_binary_path(str(path))


def phase_indices(times, phases, requested):
    """Select an actual saved sample for each complete, uniquely named phase."""
    times = np.asarray(times)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Expected finite, strictly increasing recorded times')
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Expected distinct requested phases')
    selected = []
    for name in requested:
        matches = [phase for phase in phases if phase['phase'] == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one recorded phase: {name}')
        end = matches[0]['end_s']
        if not np.isfinite(end) or end < times[0] - 1e-9 or end > times[-1] + 1e-9:
            raise ValueError(f'Phase outside the recorded trajectory: {name}')
        selected.append((name, int(np.argmin(abs(times - end)))))
    return selected


def recorded_liquid_visuals(data, liquids, index):
    """Draw the original logged liquid boundary using its actual geometry frame."""
    from meshplane import Mesh, MeshPlane, make_plane_frame
    geometry = []
    for liquid in liquids:
        if not liquid['present'][index]:
            continue
        normal, distance = liquid['normal'][index], liquid['distance'][index]
        plane = MeshPlane(Mesh(liquid['vertices'], liquid['faces'].astype(np.uint64), liquid['boundary']))
        plane.set_plane_normal(*normal)
        mesh = plane.calculate_mesh(distance)
        frame = make_plane_frame(normal)
        points = np.asarray(mesh.vertices)[mesh.boundary] @ frame
        center, axes = ellipse_frame(points[:, :2])
        local = np.eye(3)
        local[:2, :2] = axes
        geom = liquid['geom_id']
        rotation = data.geom_xmat[geom].reshape(3, 3) @ frame
        geometry.append(dict(type=int(mujoco.mjtGeom.mjGEOM_CYLINDER), size=[1., 1e-4, 0.],
            pos=(data.geom_xpos[geom] + rotation @ np.r_[center, distance]).tolist(),
            mat=(rotation @ local).ravel().tolist(), rgba=[0., 0., 1., 1.]))
    return dict(textures=[], geometry=geometry)


@contextmanager
def mujoco_renderer(model, gpu):
    """Serialize EGL replay with native jobs on the same server GPU."""
    lock_dir = ROOT / 'temp/backend_parity'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f'gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory >= 2048:
            raise RuntimeError(f'GPU {gpu} is busy ({memory} MiB); wait for the running job')
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            yield renderer


def render_episode(episode, output, requested, gpu):
    result = json.loads((episode / 'result.json').read_text())
    if result['mode'] != 'expert' or result['status'] not in ('TASK_SUCCEEDED', 'TASK_FAILED'):
        raise ValueError('Expected a completed, recorded expert episode')
    model = load_replay_model(episode / 'source/model.mjb')
    data = mujoco.MjData(model)
    with np.load(episode / 'trajectory.npz', allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in ('qpos', 'qvel', 'time')}
    count = len(trajectory['time'])
    for key, width in (('qpos', model.nq), ('qvel', model.nv)):
        if trajectory[key].shape != (count, width) or not np.isfinite(trajectory[key]).all():
            raise ValueError(f'Invalid recorded {key}')
    samples = phase_indices(trajectory['time'], result.get('expert_phases', []), requested)
    liquid_path = episode / 'task_log/episode/liquid.pkl'
    liquids = pickle.loads(liquid_path.read_bytes()) if liquid_path.exists() else []
    for liquid in liquids:
        if any(len(liquid[key]) != count + 1 for key in ('present', 'normal', 'distance', 'volume_m3')):
            raise ValueError('Liquid log must include reset plus every recorded physics step')
    if output.exists():
        raise ValueError('Use a new output directory to preserve existing evidence')
    output.mkdir(parents=True)
    frames = []
    native = result['backend'] == 'isaac'
    if result['backend'] not in ('isaac', 'mujoco'):
        raise ValueError('Unsupported recorded backend')
    context = IsaacWorker(output / 'worker', gpu) if native else mujoco_renderer(model, gpu)
    with context as renderer:
        if native:
            renderer.call('load', source=str((episode / 'source').resolve()), output=str(output.resolve()),
                          render=True, managed_render=True, physics_options=result.get('physics_options'))
            initialization = renderer.call('info')
            if initialization['steps'] != 0 or initialization['time'] != 0:
                raise RuntimeError('Native render initialization advanced physics')
        for phase, index in samples:
            qpos, qvel = trajectory['qpos'][index], trajectory['qvel'][index]
            audit = {}
            if native:
                restored = renderer.call('reset', qpos=qpos.tolist(), qvel=qvel.tolist())
                data.qpos[:], data.qvel[:] = restored['qpos'], restored['qvel']
                mujoco.mj_forward(model, data)
                delta = float(np.max(abs(data.qpos - qpos)))
                error = max(np.linalg.norm(np.asarray(pose[:3]) - data.xpos[int(body)])
                            for body, pose in restored['body_poses'].items())
                if delta >= 1e-6 or error >= 5e-6:
                    raise RuntimeError('Native restored joints or geometry differ from the recorded state')
                before = renderer.call('info')
                renderer.call('render', visuals=recorded_liquid_visuals(data, liquids, index + 1))
                after = renderer.call('info')
                if any(before[key] != after[key] for key in ('time', 'steps')):
                    raise RuntimeError('Native keyframe render advanced physics')
                audit = dict(max_qpos_refresh_delta=delta, max_restored_fk_position_error_m=float(error),
                             initialization_physics_steps=0, physics_steps_during_render=0, render_calls_to_settle_rgb=1,
                             render_passes_to_settle_rgb=8)
            else:
                data.qpos[:], data.qvel[:] = qpos, qvel
                data.time = float(trajectory['time'][index])
                mujoco.mj_forward(model, data)
            for camera in result['cameras']:
                filename = f'{phase}-camera-{camera["id"]}.png'
                path = output / filename
                if native:
                    source = output / f'camera_{camera["id"]}' / f'{after["frames"] - 1:05d}.png'
                    path.write_bytes(source.read_bytes())
                else:
                    renderer.update_scene(data, camera=camera['name'])
                    LiveVisuals.apply_mujoco(None, renderer, recorded_liquid_visuals(data, liquids, index + 1))
                    Image.fromarray(renderer.render()).save(path)
                frames.append(dict(phase=phase, camera=camera['name'], camera_id=camera['id'], image=filename,
                    image_sha256=file_sha256(path), trajectory_index=index,
                    recorded_time_s=float(trajectory['time'][index]),
                    liquid_volumes_m3=[float(liquid['volume_m3'][index + 1]) for liquid in liquids], **audit))
    report = dict(kind='recorded_trajectory_keyframe_render', closed_loop_validation=False,
        task=result['task'], backend=result['backend'], seed=result['seed'],
        motion_pass=result.get('assessment', {}).get('success') is True,
        ideal_volume_pass=result.get('volume_assessment', {}).get('success'),
        renderer_script_sha256=file_sha256(Path(__file__)),
        renderer_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        inputs_sha256={name: file_sha256(episode / name) for name in
                      ('result.json', 'trajectory.npz', 'source/model.mjb', 'source/model.npz', 'source/scene.xml')},
        liquid_log_sha256=file_sha256(liquid_path) if liquid_path.exists() else None,
        frames=frames, parity_qualified=False, scientific_process_validated=False)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=6)
    args = parser.parse_args()
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.gpu)
    report = render_episode(args.episode.resolve(), args.output.resolve(), args.phases, args.gpu)
    print(f'Rendered {len(report["frames"])} recorded keyframes')


if __name__ == '__main__':
    main()
