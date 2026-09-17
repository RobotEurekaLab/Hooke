"""Record an unchanged catalogue expert and its compiled scene for backend parity.

Run from Hooke/ with the project's MuJoCo Python. Rendering is optional and is
excluded from measured physics stepping time. Every state follows a real step;
controls are recorded BEFORE that step so replay has an unambiguous convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

import mujoco
import numpy as np


def write_snapshot(expert, out: Path):
    """Archive all public numeric model fields, names, canonical XML and reset state.

    The NPZ retains source parameters even when a destination engine has no
    equivalent. Archiving a field is never considered implementation support.
    """
    model, data = expert.model, expert.data
    arrays = {}
    for name in dir(model):
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            arrays[name] = value.copy()
    for name in ('qpos', 'qvel', 'ctrl', 'act', 'eq_active', 'qfrc_applied', 'xfrc_applied',
                 'plugin_state', 'userdata', 'xpos', 'xquat', 'cam_xpos', 'cam_xmat', 'site_xpos'):
        arrays['reset_' + name] = getattr(data, name).copy()
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    arrays['reference_xpos'] = reference.xpos.copy()
    arrays['reference_xquat'] = reference.xquat.copy()
    np.savez_compressed(out / 'model.npz', **arrays)
    mujoco.mj_saveModel(model, str(out / 'model.mjb'))
    tree = ET.fromstring(expert.spec.to_xml())
    defaults = {}
    for parent in tree.iter():
        for element in list(parent):
            if element.tag != 'default' or not element.get('class'):
                continue
            key = element.get('class')
            if key in defaults and len(element) == 0:
                # Attached models can serialize an empty duplicate namespace.
                # Removing it is safe only when the first namespace itself
                # defines no defaults (it may contain named child namespaces).
                if all(child.tag == 'default' for child in defaults[key]):
                    parent.remove(element)
            else:
                defaults[key] = element
    # MuJoCo 3.3's MjSpec XML writer omits mesh inertia modes inherited
    # through attached models. Without them shell meshes can fail to reload,
    # and exact inertia meshes silently change mass to the legacy estimate.
    meshes = {mesh.name: mesh for mesh in expert.spec.meshes}
    for element in tree.findall('./asset/mesh'):
        source_mesh = meshes[element.get('name')]
        element.set('inertia', source_mesh.inertia.name.removeprefix('mjMESH_INERTIA_').lower())
    source_assets = {}
    asset_dir = Path(__file__).resolve().parents[1] / 'assets'
    for element in tree.iter():
        if 'file' not in element.attrib:
            continue
        original = element.attrib['file']
        candidates = [Path(original), expert.default_scene.parent / original, asset_dir / original]
        found = {p.resolve() for p in candidates if p.is_file()}
        if len(found) != 1:
            raise ValueError(f'Asset path must resolve uniquely: {original}: {found}')
        source = found.pop()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        relative = Path('assets') / (digest + source.suffix)
        (out / 'assets').mkdir(exist_ok=True)
        shutil.copyfile(source, out / relative)
        element.set('file', relative.as_posix())
        source_assets[relative.as_posix()] = {'source': str(source), 'sha256': digest}
    ET.ElementTree(tree).write(out / 'scene.xml', encoding='unicode')
    # The exported XML must compile independently of source working directories.
    restored = mujoco.MjModel.from_xml_path(str((out / 'scene.xml').resolve()))
    for field in ('body_mass', 'body_inertia', 'jnt_range', 'actuator_gainprm'):
        np.testing.assert_allclose(getattr(restored, field), getattr(model, field), rtol=1e-4, atol=1e-7)
    # Frame flattening may reorder geoms within a body. Compare semantic groups,
    # retaining names and owning bodies, rather than assuming stable XML IDs.
    def geometry_groups(compiled):
        groups = defaultdict(list)
        for i in range(compiled.ngeom):
            body = int(compiled.geom_bodyid[i])
            key = (compiled.body(body).name, compiled.geom(i).name, int(compiled.geom_type[i]))
            groups[key].append(tuple(compiled.geom_size[i]))
        return {key: sorted(values) for key, values in groups.items()}
    expected, actual = geometry_groups(model), geometry_groups(restored)
    if expected.keys() != actual.keys():
        raise ValueError('Geometry names/body ownership changed in XML roundtrip')
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-4, atol=1e-7, err_msg=str(key))
    names = {}
    for kind, count in [('body', model.nbody), ('joint', model.njnt), ('geom', model.ngeom),
                        ('mesh', model.nmesh), ('material', model.nmat), ('texture', model.ntex),
                        ('camera', model.ncam), ('site', model.nsite), ('actuator', model.nu),
                        ('tendon', model.ntendon), ('equality', model.neq)]:
        obj = getattr(mujoco.mjtObj, 'mjOBJ_' + kind.upper())
        names[kind] = [mujoco.mj_id2name(model, obj, i) for i in range(count)]
    metadata = {
        'schema_version': 1, 'mujoco_version': mujoco.__version__,
        'source_scene': str(expert.default_scene), 'task': expert.task,
        'timestep_s': float(model.opt.timestep), 'gravity': model.opt.gravity.tolist(),
        'camera_clipping_m': [float(model.vis.map.znear * model.stat.extent),
                              float(model.vis.map.zfar * model.stat.extent)],
        'model_options': {k:int(getattr(model.opt,k)) for k in ('disableflags','enableflags','integrator','cone','solver','noslip_iterations')},
        'names': names, 'task_info': {k: v for k, v in expert.task_info.items() if k in ('seed', 'camera_mapping', 'prefix')},
        'model_fields': sorted(arrays),
        'source_assets': source_assets,
        'isaac_qualification': 'unqualified',
        'scope': 'Compiled source archive; no claim that archived physics fields are implemented in Isaac.',
    }
    metadata['archive_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (out/'model.npz', out/'model.mjb', out/'scene.xml')}
    (out / 'scene.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def record(task_name: str, seed: int, out: Path, render: bool, width: int, height: int,
           counterbalance: bool = False):
    from archetypes.task_catalog import CATALOG
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'task': task_name, 'seed': seed, 'backend': 'mujoco'}
    result_path = out / 'result.json'
    result_path.write_text(json.dumps(report))
    entry = CATALOG[task_name]
    if counterbalance:
        if task_name != 'close_fume_hood':
            raise ValueError('Counterbalance experiment only applies to close_fume_hood')
        from expert_common import set_gravcomp
        cls, expert_cls = entry.load_classes()
        spec = cls.load()
        set_gravcomp(spec.body('/fume_hood:sash'))
        expert = expert_cls(spec)
    else:
        expert = entry.make_expert()
    expert.reset(seed)
    mujoco.mj_forward(expert.model, expert.data)
    meta = write_snapshot(expert, out)
    data = expert.data
    samples = {k: [] for k in ('control_before_step', 'qpos', 'qvel', 'time', 'ncon', 'gripper_sash_contacts')}
    report['variant'] = 'experimental_gravity_compensated_sash' if counterbalance else 'original'
    gripper_geoms = {i for i in range(expert.model.ngeom)
                     if '2f85:' in (expert.model.body(int(expert.model.geom_bodyid[i])).name or '')}
    sash_geoms = {i for i in range(expert.model.ngeom)
                  if expert.model.body(int(expert.model.geom_bodyid[i])).name in ('/fume_hood:sash', '/fume_hood:handle')}
    initial_qpos = data.qpos.copy()
    renderer = mujoco.Renderer(expert.model, height=height, width=width) if render else None
    if render:
        from backends.source_renderer import center_directional_shadows
    cameras = list(dict.fromkeys(expert.task_info.get('camera_mapping', {}).values()))
    frame_times = []
    frame_stride = max(1, round(1 / (20 * expert.dt)))
    physics_wall = 0.0
    first_source_success = None
    original_step = expert.manager.step
    # Instrument manager, preserving all original task/system/controller code.
    def step():
        nonlocal physics_wall, first_source_success
        samples['control_before_step'].append(data.ctrl.copy())
        started = time.perf_counter()
        original_step()
        physics_wall += time.perf_counter() - started
        for field in ('qpos', 'qvel'):
            samples[field].append(getattr(data, field).copy())
        samples['time'].append(data.time)
        samples['ncon'].append(data.ncon)
        contacts = sum((int(c.geom1) in gripper_geoms and int(c.geom2) in sash_geoms) or
                       (int(c.geom2) in gripper_geoms and int(c.geom1) in sash_geoms)
                       for c in data.contact)
        samples['gripper_sash_contacts'].append(contacts)
        if first_source_success is None and expert.check():
            first_source_success = float(data.time)
        if renderer and (len(samples['time']) - 1) % frame_stride == 0:
            from PIL import Image
            for i, camera in enumerate(cameras):
                folder = out / f'camera_{i}'
                folder.mkdir(exist_ok=True)
                renderer.update_scene(data, camera=camera)
                center_directional_shadows(renderer, expert.model)
                Image.fromarray(renderer.render()).save(folder / f'{len(frame_times):05d}.png')
            frame_times.append(float(data.time))
    expert.manager.step = step
    started = time.perf_counter()
    try:
        expert.execute()
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        warnings = data.warning.number.tolist()
        source_success = bool(expert.check())
        report.update(status='PASS' if source_success and finite and not any(warnings) else 'FAIL',
                      source_success=source_success, finite_state=finite, warnings=warnings)
    except Exception as exc:
        report.update(status='ERROR', error=str(exc), traceback=traceback.format_exc())
    finally:
        if renderer:
            renderer.close()
        expert.manager.step = original_step
    wall = time.perf_counter() - started
    np.savez_compressed(out / 'trajectory.npz', initial_qpos=initial_qpos,
                        **{k: np.asarray(v) for k, v in samples.items()})
    (out / 'frames.json').write_text(json.dumps({'cameras': cameras, 'time_s': frame_times,
                                                'resolution': [width, height]}, indent=2))
    report.update(steps=len(samples['time']), simulation_s=float(data.time),
                  physics_wall_s=physics_wall, total_expert_wall_s=wall,
                  physics_steps_per_s=len(samples['time']) / physics_wall if physics_wall else None,
                  physics_realtime_factor=float(data.time) / physics_wall if physics_wall else None,
                  timing_scope='manager.step only; total includes IK/JAX warmup, recording and optional RGB',
                  final_joint_qpos={name: float(data.qpos[expert.model.jnt_qposadr[i]])
                                    for i, name in enumerate(meta['names']['joint'])
                                    if expert.model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)},
                  frames_per_camera=len(frame_times), isaac_qualification='unqualified')
    report.update(declared_time_limit_s=expert.time_limit,
                  within_declared_time_limit=bool(data.time <= expert.time_limit),
                  first_source_success_s=first_source_success,
                  gripper_sash_contact_steps=sum(v > 0 for v in samples['gripper_sash_contacts']),
                  qualification_status='UNQUALIFIED: source success predicate alone is insufficient')
    if not np.isclose(data.time, len(samples['time'])*expert.dt, atol=1e-8):
        report.update(status='ERROR', error='Task stepped physics outside Manager.step; trajectory is incomplete')
    result_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='close_fume_hood')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--counterbalance-fume-hood', action='store_true',
                        help='Experimental source variant; compensates sash gravity without changing repository assets')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    rows = [record(args.task, seed, args.output / f'seed_{seed:03d}', args.render and seed == args.seed,
                   args.width, args.height, args.counterbalance_fume_hood) for seed in range(args.seed, args.seed + args.episodes)]
    (args.output / 'summary.json').write_text(json.dumps({'task': args.task, 'results': rows,
        'successes': sum(r['status'] == 'PASS' for r in rows), 'episodes': len(rows)}, indent=2))
    raise SystemExit(0 if all(r['status'] == 'PASS' for r in rows) else 1)


if __name__ == '__main__':
    main()
