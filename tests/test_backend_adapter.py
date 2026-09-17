"""Physical contracts that must survive the cross-process backend boundary."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Hooke'))
from backends.source_forces import body_wrench_forces, passive_residual, update_kinematics
from backends.closed_loop import PhysXTaskAdapter
from backends.task_result import task_result
from backends.mass_properties import compose_welded_mass,rotation,quaternion
from backends.collision_rules import collision_participants
from backends.contact_parameters import compliant_parameters
from backends.visual_state import ellipse_frame,serialize_liquid_geom
from backends.texture_mapping import plane_uv,source_uv_to_usd


class ForceContracts(unittest.TestCase):
    def test_body_wrenches_preserve_virtual_work_at_offset_mass_center(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body quat=".7071 0 0 .7071"><freejoint/><geom type="sphere" pos=".02 .01 .03" size=".1" mass="1"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);data.qvel[:]=[.1,.2,.3,.4,.5,.6]
        mujoco.mj_forward(model,data)
        data.xfrc_applied[1]=[1.,2.,3.,.1,.2,.3]
        wrench=body_wrench_forces(model,data)
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,1,velocity,False)
        power=np.dot(data.xfrc_applied[1,:3],velocity[3:])+np.dot(data.xfrc_applied[1,3:],velocity[:3])
        self.assertAlmostEqual(np.dot(wrench,data.qvel),power,places=12)

    def test_small_display_uses_entire_texture_and_floor_keeps_spatial_repeat(self):
        corners=np.array([[-.049369,-.012886],[.049369,-.012886],[.049369,.012886],[-.049369,.012886]])
        np.testing.assert_allclose(plane_uv(corners,[.049369,.012886],[1,1],False),
                                   [[0,0],[1,0],[1,1],[0,1]],atol=1e-7)
        floor=plane_uv([[-2,-2],[2,2]],[2,2],[5,5],True)
        np.testing.assert_allclose(floor[1]-floor[0],[10,10])
        original=np.array([[.2,.1],[.8,.9]])
        np.testing.assert_allclose(source_uv_to_usd(original),[[.2,.9],[.8,.1]],atol=1e-7)
        np.testing.assert_allclose(original,[[.2,.1],[.8,.9]])

    def test_liquid_cylinder_keeps_its_thin_height_across_renderers(self):
        original=mujoco.MjvGeom();restored=mujoco.MjvGeom()
        mujoco.mjv_initGeom(original,mujoco.mjtGeom.mjGEOM_CYLINDER,np.array([1.,1e-4,0.]),
                           np.array([.1,.2,.3]),np.eye(3).ravel(),np.array([0.,0.,1.,1.],dtype=np.float32))
        payload=serialize_liquid_geom(original)
        self.assertAlmostEqual(payload['size'][1],1e-4,places=10)
        mujoco.mjv_initGeom(restored,payload['type'],np.asarray(payload['size']),np.asarray(payload['pos']),
                           np.asarray(payload['mat']).ravel(),np.asarray(payload['rgba'],dtype=np.float32))
        np.testing.assert_array_equal(restored.size,original.size)
        np.testing.assert_array_equal(restored.mat,original.mat)

    def test_liquid_visual_fit_recovers_rotated_circle_and_ellipse(self):
        angles=np.linspace(0,2*np.pi,128,endpoint=False)
        circle=np.column_stack((np.cos(angles),np.sin(angles)))
        rotation2=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
        for radii in ([.008,.008],[.013,.004]):
            transform=rotation2@np.diag(radii);origin=np.array([.023,-.04])
            boundary=circle@transform.T+origin
            center,axes=ellipse_frame(boundary)
            np.testing.assert_allclose(center,origin,atol=1e-12)
            np.testing.assert_allclose(axes@axes.T,transform@transform.T,atol=1e-12)
            self.assertGreater(np.linalg.det(axes),0)

    def test_compliance_mapping_matches_isolated_source_static_penetration(self):
        for mass in (.1,10.):
            model=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".002"/><default><geom solref=".02 1" solimp=".95 .95 .001"/></default><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .1"><freejoint/><geom type="sphere" size=".1" mass="{mass}"/></body></worldbody></mujoco>''')
            data=mujoco.MjData(model)
            for _ in range(1000):mujoco.mj_step(model,data)
            fields={key:getattr(model,key) for key in ('geom_solref','geom_solimp','geom_contype','geom_conaffinity')}
            stiffness,_=compliant_parameters(fields,1,model.opt.timestep)
            self.assertAlmostEqual(.1-data.qpos[2],9.81/stiffness,places=8)

    def test_explicit_pair_enables_zero_mask_colliders(self):
        model={'geom_contype':[1,0,0,0],'geom_conaffinity':[1,0,0,0],
               'pair_geom1':[1],'pair_geom2':[2]}
        colliders,pairs=collision_participants(model)
        self.assertEqual(colliders,{0,1,2});self.assertEqual(pairs,{(1,2)})
        self.assertNotIn(3,colliders)

    def test_massless_joint_owner_retains_descendant_mass_and_offset(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/><body pos=".1 .2 .3" quat="0 1 0 0"><inertial pos=".01 0 0" mass="2" diaginertia=".2 .3 .4"/></body><body pos="-.1 .2 .3"><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/></body></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        m={k:getattr(model,k) for k in ('body_mass','body_ipos','body_iquat','body_inertia','body_gravcomp')}
        m.update(reference_xpos=data.xpos,reference_xquat=data.xquat)
        self.assertEqual(model.body_mass[1],0.)
        p=compose_welded_mass(m,1,[1,2,3])
        self.assertAlmostEqual(p['mass'],3.)
        np.testing.assert_allclose(p['center'],[.04,.2,.3],atol=1e-12)
        np.testing.assert_allclose(p['tensor'],np.diag([.3,.4294,.5294]),atol=1e-12)
        axes=rotation(p['axes']);np.testing.assert_allclose(axes@np.diag(p['diagonal'])@axes.T,p['tensor'],atol=1e-12)
        for q in ([0,1,0,0],[0,0,1,0],[0,0,0,1],[.5,.5,.5,.5]):
            np.testing.assert_allclose(rotation(quaternion(rotation(q))),rotation(q),atol=1e-12)

    def test_partial_progress_is_not_task_success(self):
        self.assertFalse(task_result(.3)['source_success'])
        self.assertTrue(task_result(1.)['source_success'])
        self.assertTrue(task_result(np.bool_(True))['source_success'])

    def test_native_joint_forces_are_not_counted_twice(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint name="j" stiffness="12" damping="3" springref="20"/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);data.qpos[0]=.7;data.qvel[0]=1.3
        mujoco.mj_forward(model,data)
        self.assertGreater(abs(data.qfrc_passive[0]),1.)
        np.testing.assert_allclose(passive_residual(model,data),[0.],atol=1e-12)

    def test_collision_free_fk_matches_forward_positions_and_passive_forces(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/><body pos=".1 0 0"><joint name="j" stiffness="5" damping=".3"/><geom type="box" size=".1 .02 .02" mass=".2"/><site name="s" pos=".1 0 0"/></body></body></worldbody><actuator><position joint="j" kp="4"/></actuator></mujoco>''')
        expected=mujoco.MjData(model);actual=mujoco.MjData(model)
        expected.qpos[-1]=.4;expected.qvel[:]=np.arange(model.nv)*.02
        actual.qpos[:]=expected.qpos;actual.qvel[:]=expected.qvel
        mujoco.mj_forward(model,expected);update_kinematics(model,actual)
        self.assertGreater(expected.ncon,0);self.assertEqual(actual.ncon,0)
        for key in ('xpos','xquat','site_xpos','site_xmat','cvel','qfrc_passive','qfrc_bias','actuator_length','actuator_velocity'):
            np.testing.assert_allclose(getattr(actual,key),getattr(expected,key),atol=1e-12,err_msg=key)

    def test_tendon_force_survives_native_joint_subtraction(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint name="a" stiffness="12" damping="3"/><geom type="sphere" size=".1" mass="1"/></body><body pos="1 0 0"><joint name="b"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><tendon><fixed stiffness="20" springlength="0"><joint joint="a" coef="1"/><joint joint="b" coef="2"/></fixed></tendon></mujoco>''')
        data=mujoco.MjData(model);data.qpos[:]=[.2,.3];data.qvel[:]=[.1,.2]
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(passive_residual(model,data),[-16.,-32.],atol=1e-10)

    def adapter(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .1"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        adapter=PhysXTaskAdapter.__new__(PhysXTaskAdapter)
        adapter.task=SimpleNamespace(model=model,data=data)
        adapter.max_fk_position_error=adapter.max_fk_rotation_error=0.
        adapter.apply_state({'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'time':.002,
                             'contacts':[{'geom1':0,'geom2':1,'pos':[0,0,0],'normal':[0,0,1],
                                          'distance':0.,'force':[0,0,9.81]}]})
        return adapter,model,data

    def test_native_contacts_have_no_stale_mujoco_solver_indices(self):
        adapter,model,data=self.adapter()
        self.assertEqual(data.nefc,0);self.assertEqual(data.ncon,1)
        self.assertEqual(data.contact[0].efc_address,-1)
        self.assertEqual(data.contact[0].geom1,0)
        self.assertEqual(data.contact[0].geom2,1)
        np.testing.assert_array_equal(data.contact.geom,np.column_stack((data.contact.geom1,data.contact.geom2)))
        force=np.zeros(6);adapter.contact_force(model,data,0,force)
        np.testing.assert_allclose(force,[9.81,0,0,0,0,0])
        adapter.post_constraint(model,data)
        np.testing.assert_allclose(data.cfrc_ext[1,3:],[0,0,9.81])
        np.testing.assert_allclose(data.cfrc_ext[0,3:],[0,0,-9.81])

    def test_original_api_restored_after_controller_exception(self):
        adapter,_,_=self.adapter();step=mujoco.mj_step;force=mujoco.mj_contactForce
        with self.assertRaisesRegex(RuntimeError,'controller'):
            with adapter:raise RuntimeError('controller failed')
        self.assertIs(mujoco.mj_step,step);self.assertIs(mujoco.mj_contactForce,force)

    def test_physics_step_does_not_invoke_stateful_task_check(self):
        adapter,model,data=self.adapter()
        calls=[]
        adapter.task.check=lambda:calls.append('mutated task history') or True
        adapter.rows=[];adapter.contact_steps=0
        state={'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),
               'time':.004,'contacts':[]}
        adapter.worker=SimpleNamespace(call=lambda operation,**kwargs:state)
        adapter.step(model,data)
        self.assertEqual(calls,[])
        self.assertEqual(data.time,.004)
        self.assertEqual(len(adapter.rows),1)

    def test_container_acceleration_tracks_native_velocity_change(self):
        adapter,model,data=self.adapter()
        velocity=data.qvel.copy();velocity[0]+=.2
        adapter.apply_state({'qpos':data.qpos.tolist(),'qvel':velocity.tolist(),'time':.004,'contacts':[]})
        adapter.post_constraint(model,data)
        acceleration=np.zeros(6)
        mujoco.mj_objectAcceleration(model,data,mujoco.mjtObj.mjOBJ_BODY,1,acceleration,False)
        self.assertAlmostEqual(acceleration[3],100.,places=8)
        # MuJoCo's proper-acceleration convention includes support against
        # gravity at rest, as expected by the original liquid surface model.
        self.assertAlmostEqual(acceleration[5],9.81,places=8)
        adapter.apply_state({'qpos':data.qpos.tolist(),'qvel':np.zeros(model.nv).tolist(),'time':0.,'contacts':[]})
        np.testing.assert_array_equal(data.qacc,np.zeros(model.nv))

    def test_spatial_acceleration_does_not_read_old_equality_rows(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody><equality><connect body1="ball" anchor="0 0 0"/></equality></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data);self.assertGreater(data.ne,0)
        adapter=PhysXTaskAdapter.__new__(PhysXTaskAdapter);adapter.task=SimpleNamespace(model=model,data=data)
        adapter.max_fk_position_error=adapter.max_fk_rotation_error=0.
        adapter.apply_state({'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'time':.002,
                             'contacts':[{'geom1':0,'geom2':1,'pos':[0,0,0],'normal':[0,0,1],'distance':0.,'force':[0,0,9.81]}]})
        self.assertEqual((data.ne,data.nf,data.nl,data.nefc),(0,0,0,0))
        adapter.post_constraint(model,data)
        self.assertTrue(np.isfinite(data.cacc).all())

    def test_touch_sensor_reads_native_force_and_clears_source_contact(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/><site name="touch" type="box" pos="0 0 -.09" size=".1 .1 .1"/></body></worldbody><sensor><touch site="touch"/></sensor></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        self.assertGreater(data.sensordata[0],0.)
        adapter=PhysXTaskAdapter.__new__(PhysXTaskAdapter);adapter.task=SimpleNamespace(model=model,data=data)
        adapter.max_fk_position_error=adapter.max_fk_rotation_error=0.
        state={'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'time':.002,'contacts':[]}
        adapter.apply_state(state);self.assertEqual(data.sensordata[0],0.)
        state['contacts']=[{'geom1':0,'geom2':1,'pos':[0,0,0],'normal':[0,0,1],'distance':0.,'force':[0,0,17.5]}]
        adapter.apply_state(state);self.assertAlmostEqual(data.sensordata[0],17.5)


if __name__=='__main__':unittest.main()
