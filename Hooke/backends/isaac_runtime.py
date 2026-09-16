"""Reusable native PhysX scene runtime. Import after SimulationApp has started.

Only PhysX advances positions and velocities. Source model/IK code lives in a
separate process; observations and contact reports are returned from this runtime.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx.bindings._physx import SETTING_NUM_THREADS, SETTING_COLLISION_APPROXIMATE_CYLINDERS
import carb
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.sensors.camera import Camera
from backends.usd_scene import SceneBridge, rotation
from backends.contact_parameters import constraint_parameters
from backends.render_settings import RenderSettings


class NativeScene:
    def __init__(self, source, output, render=False, physics_options=None, managed_render=False):
        self.source, self.output = Path(source), Path(output)
        self.render_settings = RenderSettings.from_environment()
        self.output.mkdir(parents=True, exist_ok=True)
        self.render_enabled = render
        self.managed_render=managed_render
        self.visual_geometry={}
        self.dt = json.loads((self.source/'scene.json').read_text())['timestep_s']
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=self.dt, device='cpu')
        self.bridge = SceneBridge(self.world.stage, self.source, self.output,physics_options)
        carb.settings.get_settings().set_int(SETTING_NUM_THREADS,self.bridge.physics_options['simulation_threads'])
        approximate_cylinders=self.bridge.physics_options['approximate_cylinders']
        carb.settings.get_settings().set_bool(SETTING_COLLISION_APPROXIMATE_CYLINDERS,approximate_cylinders)
        if approximate_cylinders:
            self.bridge.limits.append('Diagnostic: analytic cylinder colliders use PhysX convex approximations.')
        if np.any(self.bridge.m['geom_type']==8):
            self.world.get_physics_context().enable_gpu_dynamics(True)
            self.world.get_physics_context().set_broadphase_type('GPU')
        gravity=np.asarray(self.bridge.meta['gravity'])
        scene=UsdPhysics.Scene(self.world.get_physics_context().get_current_physics_scene_prim())
        from pxr import Gf
        magnitude=float(np.linalg.norm(gravity))
        scene.CreateGravityMagnitudeAttr(magnitude)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(*map(float,gravity/magnitude if magnitude else [0,0,-1])))
        self.conversion = self.bridge.build()
        self.m = self.bridge.m
        self.eq_active=self.m.get('reset_eq_active',self.m['eq_active0']).copy()
        self.eq_events=[]
        self.time = 0.; self.steps = 0; self.physics_wall = 0.; self.frame_count = 0
        self.contacts = []
        self.geom_ids = {path:i for i,path in self.bridge.geom_paths.items()}
        self._configure_drives()
        self.conversion['implicit_tendon_drives']={a:dict(joint=v['joint'],weight=v['weight']) for a,v in self.tendon_drives.items()}
        self.conversion['implicit_friction_joints']=list(self.friction_drives)
        (self.output/'conversion.json').write_text(json.dumps(self.conversion,indent=2))
        for b in self.bridge.rigid:
            api = PhysxSchema.PhysxContactReportAPI.Apply(self.world.stage.GetPrimAtPath(self.bridge.body_paths[b]))
            api.CreateThresholdAttr(0.)
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._contact_report)
        self.arts = []
        for root in self.bridge.articulation_roots:
            b = root['body']
            def descends(child):
                while child and child != b:
                    child = int(self.m['body_parentid'][child])
                return child == b
            if not any(descends(int(v)) for v in self.m['jnt_bodyid']):
                self.world.stage.GetPrimAtPath(root['root_path']).RemoveAPI(UsdPhysics.ArticulationRootAPI)
                continue
            art = self.world.scene.add(SingleArticulation(prim_path=root['root_path'], name=f'art{b}'))
            self.arts.append(art)
        self.cameras = {}
        self.free = {}
        for j,b in self.bridge.free_bodies.items():
            self.free[j]=self.world.scene.add(SingleRigidPrim(prim_path=self.bridge.body_paths[b],name=f'free{j}'))
        if render:
            names = dict.fromkeys(self.bridge.meta['task_info']['camera_mapping'].values())
            for name in names:
                i = self.bridge.names['camera'].index(name)
                self.cameras[i] = Camera(prim_path=self.bridge.camera_paths[i],
                                         resolution=(self.render_settings.width,self.render_settings.height))
        self.world.reset()
        self.world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=False, update_velocities_to_usd=False)
        self.views = []
        self.dof_map = {}
        self.gains = {}
        for art in self.arts:
            ids = np.array([int(name[1:]) for name in art.dof_names], dtype=np.int32)
            qa = self.m['jnt_qposadr'][ids]; va = self.m['jnt_dofadr'][ids]
            view = art._articulation_view._physics_view
            view.set_dof_armatures(self.m['dof_armature'][va].astype(np.float32)[None,:], np.array([0],np.uint32))
            body_ids=[int(Path(path).name[1:]) if Path(path).name.startswith('b') else None for path in view.link_paths[0]]
            self.views.append({'art':art,'view':view,'ids':ids,'qa':qa,'va':va,'body_ids':body_ids})
            for local,j in enumerate(ids): self.dof_map[int(j)] = (len(self.views)-1,local)
            self.gains[art.name] = [x.tolist() for x in art.get_articulation_controller().get_gains()]
        if len(self.dof_map) != len(self.bridge.joint_paths):
            raise ValueError('Not every source scalar joint was instantiated')
        self.reset()
        for camera in self.cameras.values():
            camera.initialize(); camera.set_clipping_range(.01,100.)
        if render:
            for _ in range(40): self.world.render()
            if not managed_render:self.capture()
        self.world.stage.GetRootLayer().Export(str(self.output/'scene.usda'))

    def _configure_drives(self):
        m = self.m
        self.actuators = {}
        self.force_actuators = {}
        self.tendon_drives = {}
        self.friction_drives = {}
        for j,path in self.bridge.joint_paths.items():
            factor = np.pi/180 if m['jnt_type'][j] == 3 else 1.
            drive = UsdPhysics.DriveAPI.Apply(self.world.stage.GetPrimAtPath(path), 'angular' if m['jnt_type'][j] == 3 else 'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(float(m['jnt_stiffness'][j]*factor))
            drive.CreateDampingAttr(float(m['dof_damping'][m['jnt_dofadr'][j]]*factor))
            qa=int(m['jnt_qposadr'][j])
            drive.CreateTargetPositionAttr(float((m['qpos_spring'][qa]-m['qpos0'][qa])/factor))
        for a, trans in enumerate(m['actuator_trntype']):
            if trans == 3:
                self._configure_tendon_drive(a)
                continue
            if trans != 0 or not np.allclose(m['actuator_gear'][a], [1,0,0,0,0,0]):
                raise NotImplementedError(f'Actuator {a}: transmission/gearing needs force adapter')
            j = int(m['actuator_trnid'][a,0])
            factor = np.pi/180 if m['jnt_type'][j] == 3 else 1.
            kp,kv = -m['actuator_biasprm'][a,1],-m['actuator_biasprm'][a,2]
            if m['actuator_dyntype'][a]!=0 or m['actuator_gaintype'][a]!=0 or m['actuator_biastype'][a] not in (0,1):
                raise NotImplementedError(f'Actuator {a}: dynamic/nonlinear force model requires an adapter')
            if (kp==0 and kv==0) or m['jnt_actfrclimited'][j]:
                # Source joint force limits apply to actuator forces only,
                # excluding passive damping. Keep the passive native drive
                # and evaluate these affine actuators separately.
                self.force_actuators[a]=j
                continue
            if kp<0 or kv<0:raise NotImplementedError('Negative actuator stiffness/damping')
            drive = UsdPhysics.DriveAPI(self.world.stage.GetPrimAtPath(self.bridge.joint_paths[j]),'angular' if m['jnt_type'][j] == 3 else 'linear')
            drive.GetStiffnessAttr().Set(float((kp+m['jnt_stiffness'][j])*factor))
            drive.GetDampingAttr().Set(float((kv+m['dof_damping'][m['jnt_dofadr'][j]])*factor))
            drive.CreateMaxForceAttr(float(max(abs(m['actuator_forcerange'][a]))) if m['actuator_forcelimited'][a] else 1e20)
            if kp:
                target=(m['actuator_gainprm'][a,0]*m['reset_ctrl'][a]+m['actuator_biasprm'][a,0])/kp
                drive.GetTargetPositionAttr().Set(float((target-m['qpos0'][m['jnt_qposadr'][j]])/factor))
            else:
                drive.CreateTargetVelocityAttr(float(m['actuator_gainprm'][a,0]*m['reset_ctrl'][a]/kv/factor))
            self.actuators[a] = j
        actuated=set(self.actuators.values())|set(self.force_actuators.values())
        for a,trans in enumerate(m['actuator_trntype']):
            if trans==3:
                t=int(m['actuator_trnid'][a,0]);start=int(m['tendon_adr'][t]);count=int(m['tendon_num'][t])
                actuated.update(map(int,m['wrap_objid'][start:start+count]))
        for j,path in self.bridge.joint_paths.items():
            va=int(m['jnt_dofadr'][j]);loss=float(m['dof_frictionloss'][va])
            if loss<=0 or j in actuated or m['jnt_stiffness'][j]:continue
            # A capped implicit zero-velocity drive supplies static friction
            # even at rest. Explicit sign/tanh forces cannot hold a light
            # vertical slider and can oscillate across zero each timestep.
            factor=np.pi/180 if m['jnt_type'][j]==3 else 1.
            axis='angular' if m['jnt_type'][j]==3 else 'linear'
            drive=UsdPhysics.DriveAPI(self.world.stage.GetPrimAtPath(path),axis)
            noslip=self.bridge.meta.get('model_options',{}).get('noslip_iterations')
            if noslip is None:
                # Compatibility with archives made before this option was
                # included in scene.json; canonical XML preserves it.
                import xml.etree.ElementTree as ET
                option=ET.parse(self.source/'scene.xml').find('option')
                noslip=int(option.get('noslip_iterations','0')) if option is not None else 0
            if noslip:
                damping=loss/1e-7
            else:
                _,acceleration_damping=constraint_parameters(m['dof_solref'][va],m['dof_solimp'][va],self.dt,0.)
                damping=acceleration_damping/float(m['dof_invweight0'][va])
            drive.GetDampingAttr().Set(damping*factor)
            drive.CreateMaxForceAttr(loss)
            drive.CreateTargetVelocityAttr(0.)
            self.friction_drives[j]=loss

    def _configure_tendon_drive(self,a):
        """Reduce a fixed tendon on equality-coupled DOFs to one implicit drive.

        On q_j=q_leader+offset_j, L=W*q_leader+C and the total generalized
        force is W*F. This preserves virtual work and actuator force limits.
        Explicit high-gain tendon damping on tiny gripper links is unstable.
        """
        m=self.m;t=int(m['actuator_trnid'][a,0]);start=int(m['tendon_adr'][t]);count=int(m['tendon_num'][t])
        spans=np.arange(start,start+count)
        if not np.all(m['wrap_type'][spans]==1):return
        js=m['wrap_objid'][spans];weights=m['wrap_prm'][spans];leader=int(js[0]);coupled={leader};equations=[]
        for _ in range(len(js)):
            for i in np.flatnonzero((m['eq_type']==2)&self.eq_active):
                x,y=map(int,(m['eq_obj1id'][i],m['eq_obj2id'][i]))
                if y>=0 and np.allclose(m['eq_data'][i,:5],[0,1,0,0,0]) and (x in coupled or y in coupled):
                    coupled.update((x,y));equations.append(int(i))
        if not set(map(int,js))<=coupled or not np.all(m['jnt_type'][js]==m['jnt_type'][leader]):return
        if np.any(m['jnt_actfrclimited'][js]):return
        if m['actuator_dyntype'][a]!=0 or m['actuator_gaintype'][a]!=0 or m['actuator_biastype'][a]!=1:return
        if not np.allclose(m['actuator_gear'][a],[1,0,0,0,0,0]):return
        kp,kv=-m['actuator_biasprm'][a,1:3];weight=float(weights.sum())
        if kp<=0 or kv<0 or abs(weight)<1e-9:return
        if m['actuator_forcelimited'][a] and not np.isclose(m['actuator_forcerange'][a,0],-m['actuator_forcerange'][a,1]):return
        qa=int(m['jnt_qposadr'][leader]);va=int(m['jnt_dofadr'][leader])
        constant=float(np.dot(weights,m['qpos0'][m['jnt_qposadr'][js]]-m['qpos0'][qa]))
        axis='angular' if m['jnt_type'][leader]==3 else 'linear';factor=np.pi/180 if axis=='angular' else 1.
        drive=UsdPhysics.DriveAPI(self.world.stage.GetPrimAtPath(self.bridge.joint_paths[leader]),axis)
        drive.GetStiffnessAttr().Set(float((kp*weight**2+m['jnt_stiffness'][leader])*factor))
        drive.GetDampingAttr().Set(float((kv*weight**2+m['dof_damping'][va])*factor))
        maximum=float(max(abs(m['actuator_forcerange'][a]))*abs(weight)) if m['actuator_forcelimited'][a] else 1e20
        drive.CreateMaxForceAttr(maximum)
        self.tendon_drives[a]={'joint':leader,'weight':weight,'constant':constant,'equalities':sorted(set(equations))}

    def reset(self, qpos=None, qvel=None):
        qpos = self.m['reset_qpos'] if qpos is None else np.asarray(qpos)
        qvel = self.m['reset_qvel'] if qvel is None else np.asarray(qvel)
        for item in self.views:
            item['view'].set_dof_positions((qpos[item['qa']]-self.m['qpos0'][item['qa']]).astype(np.float32)[None,:],np.array([0],np.uint32))
            item['view'].set_dof_velocities(qvel[item['va']].astype(np.float32)[None,:],np.array([0],np.uint32))
        for j,body in self.free.items():
            qa=int(self.m['jnt_qposadr'][j]);va=int(self.m['jnt_dofadr'][j])
            # MuJoCo accepts non-unit reset quaternions and normalizes them
            # during kinematics/integration. PhysX rejects the same pose.
            orientation=np.asarray(qpos[qa+3:qa+7],dtype=float)
            norm=float(np.linalg.norm(orientation))
            if not np.isfinite(norm) or norm<1e-12:
                raise ValueError(f'Invalid free-joint quaternion for joint {j}')
            orientation=orientation/norm
            body.set_world_pose(qpos[qa:qa+3],orientation)
            r=rotation(orientation);angular=r@qvel[va+3:va+6]
            offset=r@self.bridge.mass_properties[self.bridge.free_bodies[j]]['center']
            body.set_linear_velocity(qvel[va:va+3]+np.cross(angular,offset))
            body.set_angular_velocity(angular)
        self.world.physics_sim_view.update_articulations_kinematic()
        get_physx_interface().update_transformations(True,True,True,False)
        self.time=0.; self.steps=0; self.contacts=[]
        return self.observe()

    def _contact_report(self, headers, points):
        for header in headers:
            p0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            p1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if p0 not in self.geom_ids or p1 not in self.geom_ids: continue
            a,b = self.geom_ids[p0], self.geom_ids[p1]
            for index in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                p=points[index]
                position,normal,impulse=p.position,p.normal,p.impulse
                # PhysX normal/impulse point shape1 -> shape0; MuJoCo's
                # contact force acts geom1 -> geom2. Swap pair ownership.
                # Float3's Python sequence conversion repeatedly crosses the
                # binding and probes out-of-range indices. Read its three
                # fields directly, preserving every point and impulse.
                self.contacts.append({'geom1':b,'geom2':a,'pos':[position.x,position.y,position.z],
                                      'normal':[normal.x,normal.y,normal.z],'distance':float(p.separation),
                                      'force':[impulse.x/self.dt,impulse.y/self.dt,impulse.z/self.dt]})

    def observe(self):
        qpos=self.m['reset_qpos'].copy(); qvel=np.zeros_like(self.m['reset_qvel'])
        body_poses={}
        for item in self.views:
            qpos[item['qa']]=item['view'].get_dof_positions()[0]+self.m['qpos0'][item['qa']]
            qvel[item['va']]=item['view'].get_dof_velocities()[0]
            for body,transform in zip(item['body_ids'],item['view'].get_link_transforms()[0]):
                if body is not None:
                    body_poses[body] = [float(transform[k]) for k in (0,1,2,6,3,4,5)]
        for j,body in self.free.items():
            qa=int(self.m['jnt_qposadr'][j]);va=int(self.m['jnt_dofadr'][j])
            position,orientation=body.get_world_pose()
            qpos[qa:qa+3]=position
            qpos[qa+3:qa+7]=orientation
            r=rotation(orientation);angular=body.get_angular_velocity()
            offset=r@self.bridge.mass_properties[self.bridge.free_bodies[j]]['center']
            qvel[va:va+3]=body.get_linear_velocity()-np.cross(angular,offset)
            qvel[va+3:va+6]=r.T@angular
            body_poses[self.bridge.free_bodies[j]]=qpos[qa:qa+7].tolist()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise RuntimeError('Non-finite PhysX state')
        coupling_errors={}
        for i in np.flatnonzero((self.m['eq_type']==2)&self.eq_active):
            a,b=self.m['eq_obj1id'][i],self.m['eq_obj2id'][i]
            if b<0:continue
            qa,qb=self.m['jnt_qposadr'][[a,b]]
            x=qpos[qb]-self.m['qpos0'][qb]
            expected=np.polynomial.polynomial.polyval(x,self.m['eq_data'][i,:5])
            coupling_errors[int(i)]=float(qpos[qa]-self.m['qpos0'][qa]-expected)
        self._observed={'qpos':qpos.tolist(),'qvel':qvel.tolist(),'time':self.time,'contacts':self.contacts,
                        'body_poses':body_poses,'joint_coupling_errors':coupling_errors}
        return self._observed

    def step(self, control, extra_forces=None, eq_active=None, eq_data=None):
        m=self.m
        if eq_data is not None:
            parameters = np.asarray(eq_data,dtype=float)
            if parameters.shape != m['eq_data'].shape or not np.isfinite(parameters).all():
                raise ValueError('Equality parameter size or values are invalid')
            changed = np.flatnonzero(np.any(parameters != m['eq_data'],axis=1))
            if np.any(m['eq_type'][changed] != 1) or np.any(self.eq_active[changed]):
                raise NotImplementedError('Only inactive weld capture parameters can change')
            m['eq_data'][:] = parameters
        if eq_active is not None and not np.array_equal(eq_active,self.eq_active):
            for tendon in self.tendon_drives.values():
                if any(not eq_active[i] for i in tendon['equalities']):
                    raise NotImplementedError('Disabling an implicit tendon coupling requires rebuilding its drive')
            self.eq_events.append({'time':self.time,'previous':self.eq_active.tolist(),'active':list(eq_active)})
            (self.output/'constraint_events.json').write_text(json.dumps(self.eq_events,indent=2))
            self.bridge.equalities(eq_active,previous=self.eq_active)
            self.eq_active=np.asarray(eq_active,dtype=bool).copy()
        ctrl=np.asarray(control,dtype=np.float64)
        if ctrl.shape != m['reset_ctrl'].shape or not np.isfinite(ctrl).all():
            raise ValueError('Control size or values are invalid')
        state=self._observed; q=np.asarray(state['qpos']); v=np.asarray(state['qvel'])
        forces=np.zeros_like(v) if extra_forces is None else np.asarray(extra_forces,dtype=float).copy()
        actuator_forces=np.zeros_like(v)
        if forces.shape != v.shape or not np.isfinite(forces).all():raise ValueError('Generalized force size or values are invalid')
        targets=[(m['qpos_spring'][item['qa']]-m['qpos0'][item['qa']]).astype(np.float32) for item in self.views]
        velocity_targets=[np.zeros(len(item['ids']),np.float32) for item in self.views]
        for a,j in self.actuators.items():
            art,local=self.dof_map[j]
            value=np.clip(ctrl[a],*m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
            kp,kv=-m['actuator_biasprm'][a,1:3]
            numerator=m['actuator_gainprm'][a,0]*value+m['actuator_biasprm'][a,0]
            qa=m['jnt_qposadr'][j];ks=m['jnt_stiffness'][j]
            if kp:
                targets[art][local]=(numerator+ks*m['qpos_spring'][qa])/(kp+ks)-m['qpos0'][qa]
            else:
                velocity_targets[art][local]=numerator/(kv+m['dof_damping'][m['jnt_dofadr'][j]])
        for a,j in self.force_actuators.items():
            value=np.clip(ctrl[a],*m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
            bias=m['actuator_biasprm'][a];qa=m['jnt_qposadr'][j];va=m['jnt_dofadr'][j]
            force=m['actuator_gainprm'][a,0]*value+bias[0]+bias[1]*q[qa]+bias[2]*v[va]
            if m['actuator_forcelimited'][a]:force=np.clip(force,*m['actuator_forcerange'][a])
            actuator_forces[va]+=force
        for a,tendon in self.tendon_drives.items():
            j=tendon['joint'];weight=tendon['weight'];art,local=self.dof_map[j];qa=m['jnt_qposadr'][j]
            value=np.clip(ctrl[a],*m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
            kp=-m['actuator_biasprm'][a,1];ks=m['jnt_stiffness'][j]
            numerator=weight*(m['actuator_gainprm'][a,0]*value+m['actuator_biasprm'][a,0]-kp*tendon['constant'])
            targets[art][local]=(numerator+ks*m['qpos_spring'][qa])/(kp*weight**2+ks)-m['qpos0'][qa]
        for a,trans in enumerate(m['actuator_trntype']):
            if trans != 3 or a in self.tendon_drives:continue
            t=int(m['actuator_trnid'][a,0]); start=int(m['tendon_adr'][t]); count=int(m['tendon_num'][t])
            spans=np.arange(start,start+count)
            if not np.all(m['wrap_type'][spans] == 1):raise NotImplementedError('Spatial tendon force adapter required')
            js=m['wrap_objid'][spans]; weights=m['wrap_prm'][spans]
            qa=m['jnt_qposadr'][js]; va=m['jnt_dofadr'][js]
            u=np.clip(ctrl[a],*m['actuator_ctrlrange'][a]) if m['actuator_ctrllimited'][a] else ctrl[a]
            bias=m['actuator_biasprm'][a]
            force=m['actuator_gainprm'][a,0]*u+bias[0]+bias[1]*np.dot(weights,q[qa])+bias[2]*np.dot(weights,v[va])
            if m['actuator_forcelimited'][a]:force=np.clip(force,*m['actuator_forcerange'][a])
            np.add.at(actuator_forces,va,weights*force)
        for j in self.dof_map:
            if m['jnt_actfrclimited'][j]:
                va=m['jnt_dofadr'][j]
                actuator_forces[va]=np.clip(actuator_forces[va],*m['jnt_actfrcrange'][j])
        forces+=actuator_forces
        friction=m['dof_frictionloss']*np.tanh(v/.001)
        for j in self.friction_drives:
            va=int(m['jnt_dofadr'][j]);friction[va]=0.
            # The native drive is reserved for dry friction. Retain source
            # viscous damping separately, without clipping it to frictionloss.
            forces[va]-=m['dof_damping'][va]*v[va]
        forces-=friction
        for item,target,velocity in zip(self.views,targets,velocity_targets):
            item['view'].set_dof_position_targets(target[None,:],np.array([0],np.uint32))
            item['view'].set_dof_velocity_targets(velocity[None,:],np.array([0],np.uint32))
            item['view'].set_dof_actuation_forces(forces[item['va']].astype(np.float32)[None,:],np.array([0],np.uint32))
        for j,body in self.free.items():
            va=m['jnt_dofadr'][j];qa=m['jnt_qposadr'][j]
            force=forces[va:va+3];torque=rotation(q[qa+3:qa+7])@forces[va+3:va+6]
            if np.any(force) or np.any(torque):
                body._rigid_prim_view.apply_forces_and_torques_at_pos(
                    forces=force[None,:],torques=torque[None,:],
                    positions=q[None,qa:qa+3],is_global=True)
        self.contacts=[]
        started=time.perf_counter()
        self.world.step(render=False)
        self.physics_wall+=time.perf_counter()-started
        self.steps+=1; self.time=self.steps*self.dt
        if self.render_enabled and not self.managed_render and self.steps % self.render_settings.step_interval(self.dt) == 0:
            get_physx_interface().update_transformations(True,True,False,False)
            self.world.render(); self.capture()
        return self.observe()

    def render_frame(self, visuals):
        """Render source display/liquid updates after the task's manager step."""
        if not self.render_enabled:raise RuntimeError('Rendering is disabled for this scene')
        before=self.observe()
        from backends.usd_visuals import apply_visuals
        stats=apply_visuals(self.world.stage,self.m,visuals,self.visual_geometry)
        get_physx_interface().update_transformations(True,True,False,False)
        # RGB annotators buffer render frames for moving actors as well as
        # texture changes. Drain that latency before pairing RGB with state.
        for _ in range(8):self.world.render()
        self.capture()
        after=self.observe()
        if any(not np.allclose(before[key],after[key],rtol=0,atol=1e-6) for key in ('qpos','qvel')):
            raise RuntimeError('Rendering changed the native physics state')
        return {'frames':self.frame_count,**stats}

    def capture(self):
        from PIL import Image
        for i,camera in self.cameras.items():
            rgba=camera.get_rgba()
            if rgba is None or rgba.shape != (self.render_settings.height,self.render_settings.width,4):raise RuntimeError('Camera did not produce RGB')
            folder=self.output/f'camera_{i}';folder.mkdir(exist_ok=True)
            path=folder/f'{self.frame_count:05d}.png'
            temporary=path.with_suffix('.tmp')
            Image.fromarray(np.asarray(rgba[:,:,:3],dtype=np.uint8)).save(temporary,format='PNG')
            temporary.replace(path)
        self.frame_count+=1

    def close(self):
        self.subscription=None
        self.world.stop()
        self.world.clear()
        World.clear_instance()
        create_new_stage()
