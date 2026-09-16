"""Asynchronous backend jobs; serves only selected images and result files."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from flask import Blueprint, jsonify, request, send_file, abort
from archetypes.task_catalog import CATALOG

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'temp/backend_parity'
JOBS=EVIDENCE/'jobs'
bp=Blueprint('backends',__name__)
_jobs={}
_lock=threading.Lock()


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default if default is not None else {}


def evidence():
    rows={}
    for folder in ('catalog_native_check','catalog_native_simple','catalog_native_complex','catalog_native_final','catalog_native_v6_final','catalog_native_v6_reset_fix'):
        for row in read_json(EVIDENCE/folder/'batch_results.json').get('results',[]):
            rows[row['task']]={**row,'folder':folder}
    return rows


def expert_evidence():
    """Expose task outcomes separately from short scene-loading checks."""
    rows={}
    sources=[EVIDENCE/folder/'task_results.json' for folder in
             ('tasks_mujoco','native_v4_tasks','native_v5_tasks','native_v6_regression_r3','native_v6_remaining','native_v6_vortex_final','native_v6_pipette_final')]
    for path in sources:
        for row in read_json(path).get('results',[]):
            if row.get('seed')!=0 or row.get('mode')!='expert':continue
            summary={key:row.get(key) for key in ('status','source_check','source_check_kind','simulation_s',
                         'within_declared_time_limit','max_fk_position_error_m','total_wall_s')}
            rows.setdefault(row['task'],{})[row['backend']]=summary
    return rows


@bp.get('/backends')
def page():return send_file(Path(__file__).with_name('static')/'backends.html')


@bp.get('/api/backends/catalog')
def catalog():
    inventory={r['task']:r for r in read_json(EVIDENCE/'inventory.json').get('entries',[])}
    reports=evidence();experts=expert_evidence()
    return jsonify(tasks=[{'name':e.name,'description':e.description,'category':e.category,
                           'display_only':inventory.get(e.name,{}).get('display_only',False),
                           'check_constant_true':inventory.get(e.name,{}).get('check_constant_true',False),
                           'check_constant_false':inventory.get(e.name,{}).get('check_constant_false',False),
                           'expert_validation':experts.get(e.name,{}),
                           'isaac':reports.get(e.name)} for e in CATALOG.values()],
                   completed=len(reports),passed=sum(r.get('status') in ('LOAD_STEP_OK','LOAD_STEP_RENDER_OK') and r.get('finite',False) for r in reports.values()),
                   total=len(CATALOG),parity_qualified=False)


@bp.get('/api/backends/preview/<task>')
def cached_preview(task):
    if task not in CATALOG:abort(404)
    row=evidence().get(task,{})
    if not row.get('status','').startswith('LOAD_STEP'):abort(404)
    folder=EVIDENCE/row['folder']/task
    images=sorted(folder.glob('camera_*/*.png'))
    if not images:abort(404)
    # The first frame of the first task camera is the reset scene.
    return send_file(images[0],mimetype='image/png',max_age=0)


def terminate(job):
    process=job['process']
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=5)


def monitor(job):
    # Slow SDF tasks can legitimately exceed a fixed wall-clock allowance.
    # The runner already bounds simulation duration; stop stalled jobs when
    # they produce no state progress for five minutes instead.
    try:
        while job['process'].poll() is None:
            try:job['process'].wait(timeout=15)
            except subprocess.TimeoutExpired:
                latest=job['created']
                for name in ('result.json','progress.json'):
                    path=job['output']/name
                    try:latest=max(latest,path.stat().st_mtime)
                    except FileNotFoundError:pass
                if time.time()-latest>300:
                    job['terminal']='TIME_LIMIT';terminate(job);break
    finally:job['log'].close()


def snapshot(job):
    result=read_json(job['output']/'result.json',{'status':'STARTING'})
    progress=read_json(job['output']/'progress.json')
    process=job.get('process');code=process.poll() if process else None
    if job.get('terminal'):result['status']=job['terminal']
    elif process and code is not None and result.get('status') in ('STARTING','RUNNING'):
        result.update(status='ERROR',error=f'Worker exited before writing a result (exit {code}).')
    elif not process and result.get('status') in ('STARTING','RUNNING'):
        result.update(status='INTERRUPTED',error='Server restarted; this previous job is no longer monitored.')
    base=job['output']/('isaac' if job['backend']=='isaac' else 'mujoco')
    cameras=[]
    for folder in sorted(base.glob('camera_*')):
        frames=sorted(folder.glob('*.png'))
        if frames:
            cameras.append({'id':int(folder.name[7:]),'frames':len(frames),
                            'url':f"/api/backends/jobs/{job['id']}/image/{folder.name[7:]}/{frames[-1].stem}"})
    return {**result,'id':job['id'],'backend':job['backend'],'task':job['task'],'progress':progress,'images':cameras,
            'exit_code':code,'running':bool(process and code is None and not job.get('terminal'))}


def get_job(identifier):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):abort(404)
    with _lock:
        if identifier not in _jobs:
            meta=read_json(JOBS/identifier/'job.json')
            if not meta:abort(404)
            _jobs[identifier]={**meta,'output':JOBS/identifier}
        return _jobs[identifier]


@bp.post('/api/backends/jobs')
def start_job():
    body=request.get_json(silent=True) or {}
    task=body.get('task');backend=body.get('backend');mode=body.get('mode','expert')
    if task not in CATALOG or backend not in ('mujoco','isaac') or mode not in ('preview','expert'):
        return jsonify(error='Invalid task, backend or mode'),400
    try:
        seed=int(body.get('seed',0));seconds=float(body.get('seconds',2))
        if not 0<=seed<=2**31-1 or not 0<seconds<=10:raise ValueError()
    except (TypeError,ValueError,OverflowError):return jsonify(error='Invalid seed or duration'),400
    with _lock:
        if any(j['backend']==backend and j.get('process') and j['process'].poll() is None for j in _jobs.values()):
            return jsonify(error=f'{backend} 已有运行中的任务，请等待或停止该任务。'),409
        identifier=uuid.uuid4().hex;output=JOBS/identifier;output.mkdir(parents=True)
        meta={'id':identifier,'task':task,'backend':backend,'mode':mode,'seed':seed,'created':time.time()}
        (output/'job.json').write_text(json.dumps(meta))
        env=os.environ.copy();env.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='6',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        log=(output/'run.log').open('wb')
        try:
            process=subprocess.Popen([sys.executable,'-m','backends.run','--task',task,'--backend',backend,'--mode',mode,
                                      '--seed',str(seed),'--seconds',str(seconds),'--output',str(output)],
                                     cwd=ROOT/'Hooke',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        except OSError:
            log.close();raise
        job={**meta,'output':output,'process':process,'log':log};_jobs[identifier]=job
        threading.Thread(target=monitor,args=(job,),daemon=True).start()
    return jsonify(snapshot(job)),202


@bp.get('/api/backends/jobs/<identifier>')
def job_status(identifier):return jsonify(snapshot(get_job(identifier)))


@bp.post('/api/backends/jobs/<identifier>/stop')
def stop_job(identifier):
    job=get_job(identifier)
    if job.get('process') and job['process'].poll() is None:
        job['terminal']='CANCELLED';terminate(job)
    return jsonify(snapshot(job))


@bp.get('/api/backends/jobs/<identifier>/image/<int:camera>/<int:frame>')
def job_image(identifier,camera,frame):
    job=get_job(identifier)
    if camera<0 or frame<0:abort(404)
    path=job['output']/job['backend']/f'camera_{camera}'/f'{frame:05d}.png'
    if not path.is_file():abort(404)
    return send_file(path,mimetype='image/png',max_age=0)


@bp.get('/api/backends/jobs/<identifier>/result')
def job_result(identifier):
    return jsonify(snapshot(get_job(identifier)))
