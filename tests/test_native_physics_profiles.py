"""Export/runtime boundary tests for optional native physics metadata, on CPU."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

CANDIDATE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CANDIDATE/'Hooke'))
from backends.baseline import write_snapshot
from backends.physics_options import merge_native_physics_options,validate_physics_options
from real_labs.native_profiles import native_physics_options_for_manifest
from real_labs.preflight import inspect_package
from real_labs.runtime import InstrumentSession
from real_labs.epfl_jack_geometry import add_scissor_jack

XML='''<mujoco><option timestep=".002"/><worldbody>
<camera name="overview" pos="0 -1 .5"/><camera name="workstation" pos="0 -.5 .3"/>
<geom name="arch_roof_fixture" type="box" pos="0 0 1" size=".5 .5 .01" contype="0" conaffinity="0"/>
<body name="moving"><joint name="slider" type="slide" axis="1 0 0"/><geom name="marker" type="box" size=".1 .1 .1" mass="1" contype="0" conaffinity="0"/></body>
<body name="fixed" pos=".3 0 0"><geom name="fixed_marker" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/></body>
</worldbody><equality>
<connect name="lower" body1="moving" body2="fixed" anchor="0 0 0"/>
<connect name="upper" body1="moving" body2="fixed" anchor="0 .02 0"/>
<connect name="deck" body1="moving" body2="fixed" anchor="0 .04 0"/>
</equality></mujoco>'''
BINDINGS=dict(zip(('lower_scissor','upper_scissor','deck_slot'),('lower','upper','deck')))
PROFILE={'native_substeps':4,'soft_connect_constraints':False}


def manifest(selected=True):
    item={'id':'fixture','kind':'test_fixture','role':'Synthetic metadata wiring test','joints':{},'actuators':{}}
    if selected:item['mechanisms']={'bath':{'kind':'scissor_jack_v1','equalities':dict(BINDINGS)}}
    return {'id':'a_future_laboratory','task':'metadata_fixture','equipment':[item],
            'scene_sha256':hashlib.sha256(XML.encode()).hexdigest()}


def actual_jack_package():
    """Small real production jack, with complete operational bindings."""
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.002')
    assets = ET.SubElement(root, 'asset')
    for material in ('metal', 'rubber'):
        ET.SubElement(assets, 'material', name=f'fixture__mat_{material}', rgba='.3 .3 .3 1')
    world = ET.SubElement(root, 'worldbody')
    for name in ('overview', 'workstation'):
        ET.SubElement(world, 'camera', name=name, pos='0 -1 .5')
    ET.SubElement(world, 'geom', name='arch_roof_fixture', type='box', size='.5 .5 .01', pos='0 0 1', contype='0', conaffinity='0')
    parent = ET.SubElement(world, 'body', name='fixture')
    ET.SubElement(parent, 'site', name='datum', size='.001')
    assembly = add_scissor_jack(SimpleNamespace(root=root, name='fixture'), parent, origin=(0, 0, 0))
    payload = ET.SubElement(assembly['deck'], 'body', name='payload', pos='.075 0 .018')
    ET.SubElement(payload, 'inertial', mass='1.8', pos='0 0 .065', diaginertia='.009 .009 .0144')
    ET.SubElement(payload, 'site', name='rim', pos='0 0 .140', size='.001')
    mechanism = dict(kind='scissor_jack_v1', joints=assembly['joints'], equalities=assembly['equalities'],
                     actuator=assembly['actuator'], deck_site=assembly['deck_site'], datum_site='datum',
                     observation_site='rim', range=[.245, .375], deck_to_observation_offset_m=.140)
    xml = ET.tostring(root, encoding='unicode')
    metadata = manifest(False)
    metadata['equipment'][0]['mechanisms'] = {'bath': mechanism}
    metadata['scene_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
    return xml, metadata


def runtime_prefix():
    """Execute the actual constructor through World creation, without Isaac."""
    tree=ast.parse((CANDIDATE/'Hooke/backends/isaac_runtime.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeScene')
    init=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
    index=next(i for i,node in enumerate(init.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='world' for t in node.targets))
    init.body=init.body[:index+1]
    calls=[]
    namespace={'Path':Path,'json':json,'np':np,'RenderSettings':SimpleNamespace(from_environment=lambda:None),
               'World':lambda **kwargs:calls.append(kwargs)}
    exec(compile(ast.Module(body=[init],type_ignores=[]),'<actual-runtime-constructor-prefix>','exec'),namespace)
    return type('RuntimePrefix',(),{'__init__':namespace['__init__']}),calls


class NativeProfileBoundaryTests(unittest.TestCase):
    def test_merge_preserves_inputs_and_caller_precedence(self):
        metadata={'native_physics_options':{**PROFILE,'world_origin_m':[1,2,3]}}
        original=deepcopy(metadata)
        actual=merge_native_physics_options(metadata,{'native_substeps':1,'soft_connect_constraints':True})
        self.assertEqual(actual,{'native_substeps':1,'soft_connect_constraints':True,'world_origin_m':[1.,2.,3.]})
        actual['world_origin_m'][0]=9
        self.assertEqual(metadata,original)
        self.assertEqual(merge_native_physics_options({}),{})

    def test_rejects_unknown_nonfinite_and_malformed_options(self):
        bad=[None,[],{3:4},{'unknown':1},{'native_substeps':True},{'native_substeps':4.0},
             {'native_substeps':0},{'native_substeps':33},{'solver_position_iterations':256},
             {'solver_velocity_iterations':float('nan')},{'simulation_threads':0},
             {'contact_offset_m':0},{'sdf_contact_offset_m':float('inf')},{'contact_offset_m':10**1000},
             {'compliant_contact_scale':-1},{'connect_impedance_fraction':1.1},{'soft_connect_constraints':1},
             {'solver_type':'other'},{'world_origin_m':[1,2]},{'world_origin_m':[0,0,float('nan')]},
             {'world_origin_m':'123'},{'world_origin_m':{0:1,1:1,2:1}},{'world_origin_m':[True,0,0]}]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):validate_physics_options(value)
                with self.assertRaises(ValueError):merge_native_physics_options({'native_physics_options':value},{'native_substeps':1})

    def test_stabilization_is_explicit_and_caller_can_override_archived_setting(self):
        self.assertEqual(validate_physics_options({'enable_stabilization': False}),
                         {'enable_stabilization': False})
        self.assertEqual(merge_native_physics_options(
            {'native_physics_options': {'enable_stabilization': True}},
            {'enable_stabilization': False}), {'enable_stabilization': False})
        for value in (0, 1, 'false', None, np.bool_(False)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                validate_physics_options({'enable_stabilization': value})

    def test_automatic_profile_requires_all_connects_to_be_owned(self):
        model=mujoco.MjModel.from_xml_string(XML)
        self.assertEqual(native_physics_options_for_manifest(manifest(),model),PROFILE)
        self.assertEqual(native_physics_options_for_manifest(manifest(False),model),{})
        with self.assertRaisesRegex(ValueError,'ownership'):native_physics_options_for_manifest(manifest())
        mixed=XML.replace('</equality>','<connect name="unrelated_fluid_hose" body1="moving" body2="fixed" anchor="0 0 0"/></equality>')
        with self.assertRaisesRegex(ValueError,'unowned'):
            native_physics_options_for_manifest(manifest(),mujoco.MjModel.from_xml_string(mixed))
        missing=manifest();missing['equipment'][0]['mechanisms']['bath']['equalities']['deck_slot']='missing'
        with self.assertRaisesRegex(ValueError,'missing'):native_physics_options_for_manifest(missing,model)

    def test_snapshot_profile_changes_only_optional_metadata(self):
        spec=mujoco.MjSpec.from_string(XML);model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        with tempfile.TemporaryDirectory(prefix='hooke-profile-') as directory:
            root=Path(directory);legacy=root/'legacy';profiled=root/'profiled';legacy.mkdir();profiled.mkdir()
            expert=SimpleNamespace(spec=spec,model=model,data=data,task='fixture',default_scene=root/'scene.xml',task_info={})
            original=write_snapshot(expert,legacy)
            expert.native_physics_options=PROFILE
            selected=write_snapshot(expert,profiled)
            self.assertNotIn('native_physics_options',original)
            self.assertEqual(selected['native_physics_options'],PROFILE)
            for filename in ('model.mjb','scene.xml'):
                self.assertEqual((legacy/filename).read_bytes(),(profiled/filename).read_bytes())
            with np.load(legacy/'model.npz') as a,np.load(profiled/'model.npz') as b:
                self.assertEqual(a.files,b.files)
                for name in a.files:np.testing.assert_array_equal(a[name],b[name],err_msg=name)
            self.assertEqual(selected['isaac_qualification'],'unqualified')

    def test_real_labs_export_wires_optional_profile_and_report(self):
        for selected in (False,True):
            with self.subTest(selected=selected),tempfile.TemporaryDirectory(prefix='hooke-profile-export-') as directory:
                xml, metadata = actual_jack_package() if selected else (XML, manifest(False))
                root=Path(directory);(root/'scene.xml').write_text(xml)
                (root/'manifest.json').write_text(json.dumps(metadata))
                report=inspect_package(root,native_archive=True)
                archive=json.loads((root/'native_source/scene.json').read_text())
                self.assertEqual(archive.get('native_physics_options',{}),PROFILE if selected else {})
                self.assertEqual(report['exports']['native_archive'].get('native_physics_options',{}),PROFILE if selected else {})
                self.assertFalse(report['acceptance']['isaac_runtime_validated'])
                self.assertEqual(archive['timestep_s'],.002)
                expected_model = mujoco.MjModel.from_xml_string(xml)
                session = InstrumentSession(expected_model, metadata)
                with np.load(root/'native_source/model.npz') as snapshot:
                    for name in ('qpos', 'qvel', 'ctrl', 'eq_active', 'qfrc_applied', 'xfrc_applied'):
                        np.testing.assert_array_equal(snapshot['reset_' + name], getattr(session.data, name))
                    for name in ('body_mass', 'body_inertia', 'geom_pos', 'geom_quat', 'eq_data'):
                        np.testing.assert_array_equal(snapshot[name], getattr(expected_model, name))
                    if selected:
                        self.assertGreater(float(np.max(np.abs(snapshot['reset_ctrl']))), .001)
                self.assertEqual(session.data.time, 0.)
                self.assertEqual(report['exports']['native_archive']['reset_state'],
                                 'Canonical InstrumentSession reset; zero physics steps')

    def test_runtime_merges_profile_before_world_creation(self):
        Runtime,calls=runtime_prefix()
        with tempfile.TemporaryDirectory(prefix='hooke-profile-runtime-') as directory:
            root=Path(directory)
            for archived,overrides,expected in (({},None,1),({'native_physics_options':PROFILE},None,4),
                    ({'native_physics_options':PROFILE},{'native_substeps':1,'soft_connect_constraints':True},1)):
                (root/'scene.json').write_text(json.dumps({'timestep_s':.002,**archived}))
                runtime=Runtime(root,root/'output',physics_options=overrides)
                self.assertEqual(runtime.native_substeps,expected)
                self.assertEqual(runtime.dt,.002)
                self.assertEqual(calls[-1]['physics_dt'],.002/expected)
                self.assertEqual(calls[-1]['rendering_dt'],.002)
            count=len(calls)
            with self.assertRaises(ValueError):Runtime(root,root/'output',physics_options={'native_substeps':True})
            self.assertEqual(len(calls),count)


if __name__=='__main__':unittest.main(verbosity=2)
