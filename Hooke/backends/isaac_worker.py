"""Isolated Isaac process controlled over a local Unix socket (JSON messages)."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--socket',type=Path,required=True)
args=parser.parse_args()
from isaacsim import SimulationApp
# A batch worker never hot-reloads extensions. On shared servers, extension
# watches can exhaust the per-user inotify quota before a scene even loads.
extra_args=['--/rtx/rendermode=PathTracing', '--/app/extensions/fsWatcherEnabled=false']
# Bound the shared-server task pool. Zero remains an explicit opt-in to Kit's
# automatic sizing; large automatic pools intermittently abort at startup here.
task_threads=int(os.environ.get('HOOKE_ISAAC_TASK_THREADS','8'))
if not 0 <= task_threads <= 128:
 raise ValueError('HOOKE_ISAAC_TASK_THREADS must be between 0 and 128')
if task_threads:
 extra_args.append(f'--/plugins/carb.tasking.plugin/threadCount={task_threads}')
app=SimulationApp({'headless':True,'active_gpu':int(os.environ.get('HOOKE_ISAAC_RENDER_GPU','6')),
 'physics_gpu':0,'multi_gpu':False,'renderer':'PathTracing','width':640,'height':480,
 'samples_per_pixel_per_frame':1,'max_bounces':2,'max_specular_transmission_bounces':4,
 'max_volume_bounces':0,'anti_aliasing':0,'extra_args':extra_args})
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backends.isaac_runtime import NativeScene
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage
scene=None
server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
 server.bind(str(args.socket));os.chmod(args.socket,0o600);server.listen(1)
 print('HOOKE_WORKER_READY',flush=True)
 connection,_=server.accept()
 with connection,connection.makefile('rwb') as stream:
  while True:
   line=stream.readline(16*1024*1024)
   if not line:break
   shutdown=False
   try:
    request=json.loads(line);op=request['op']
    if op=='load':
     if scene is not None:scene.close();scene=None
     elif World.instance() is not None:
      # A failed constructor can leave a partial World behind. Discard it
      # before processing the next catalogue entry, not just successful scenes.
      World.instance().stop();World.instance().clear();World.clear_instance();create_new_stage()
     scene=NativeScene(request['source'],request['output'],request.get('render',False),request.get('physics_options'),request.get('managed_render',False))
     result={'state':scene.observe(),'conversion':scene.conversion,'gains':scene.gains}
    elif op=='step':result=scene.step(request['control'],request.get('extra_forces'),request.get('eq_active'))
    elif op=='reset':result=scene.reset(request.get('qpos'),request.get('qvel'))
    elif op=='render':result=scene.render_frame(request.get('visuals',{}))
    elif op=='observe':result=scene.observe()
    elif op=='collider_info':
     from omni.physx import get_physx_cooking_interface
     cooking=get_physx_cooking_interface();result=[]
     ids=request.get('geoms',[])
     if len(ids)>32:raise ValueError('Inspect at most 32 colliders')
     for geom in ids:
      path=scene.bridge.geom_paths[int(geom)];meshes=[]
      for index in range(cooking.get_nb_convex_mesh_data(path)):
       mesh=cooking.get_convex_mesh_data(path,index)
       if 'vertices' not in mesh or 'polygons' not in mesh:
        meshes.append({'unavailable':True,'returned_fields':list(mesh)})
        continue
       meshes.append({'vertices':[[float(v[0]),float(v[1]),float(v[2])] for v in mesh['vertices']],
                      'planes':[[float(p['plane'][j]) for j in range(4)] for p in mesh['polygons']]})
      result.append({'geom':geom,'meshes':meshes})
    elif op=='profile_steps':
     # Bounded local diagnostic; advances real physics with supplied controls.
     import cProfile,pstats,io,time
     count=int(request.get('steps',100))
     if not 1<=count<=500:raise ValueError('Profile steps must be 1..500')
     profile=cProfile.Profile();started=time.perf_counter();profile.enable()
     for _ in range(count):state=scene.step(request['control'])
     profile.disable();wall=time.perf_counter()-started
     report=io.StringIO();pstats.Stats(profile,stream=report).sort_stats('cumulative').print_stats(35)
     result={'state':state,'steps':count,'wall_s':wall,'profile':report.getvalue()}
    elif op=='info':result={'time':scene.time,'steps':scene.steps,'frames':scene.frame_count,'physics_wall_s':scene.physics_wall}
    elif op=='shutdown':result={'closed':True};shutdown=True
    else:raise ValueError(f'Unknown operation: {op}')
    response={'ok':True,'result':result}
   except Exception as exc:
    traceback.print_exc()
    response={'ok':False,'error':str(exc),'traceback':traceback.format_exc()}
   stream.write((json.dumps(response,allow_nan=False)+'\n').encode());stream.flush()
   if shutdown:break
finally:
 if scene is not None:scene.close()
 server.close()
 args.socket.unlink(missing_ok=True)
 app.close()
