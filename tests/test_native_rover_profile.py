"""CPU capability/binding contracts for composable native export profiles."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco


ROOT = Path(__file__).resolve().parents[1]
ROVER_PROFILE = {
    'compound_cylinder_colliders': True,
    'contact_offset_m': 5e-5,
    'sdf_contact_offset_m': 5e-5,
    'solver_type': 'PGS',
    'simulation_threads': 4,
    'explicit_affine_actuators': True,
    'native_substeps': 4,
}
spec = importlib.util.spec_from_file_location('tested_native_profiles', ROOT / 'Hooke/real_labs/native_profiles.py')
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def fixture(scissor=False):
    wheels = ''.join(f'<body name="wheel_body_{i}" pos="{i * .2} 0 0">'
                     f'<joint name="wheel_joint_{i}" type="hinge"/>'
                     f'<geom name="wheel_geom_{i}" type="cylinder" size=".05 .02" mass="1"/>'
                     '</body>' for i in range(6))
    actuators = ''.join(f'<velocity name="wheel_motor_{i}" joint="wheel_joint_{i}" kv="1"/>' for i in range(6))
    equality = ''.join(f'<connect name="{name}" body1="base" body2="fixture" anchor="0 0 {i*.01}"/>'
                       for i, name in enumerate(('lower', 'upper', 'deck'))) if scissor else ''
    xml = ('<mujoco><worldbody><geom name="outside" type="cylinder" size=".1 .1"/>'
           '<body name="fixture"><geom name="fixed" size=".01"/></body>'
           '<body name="base"><freejoint name="free_base"/><geom name="proximal" size=".1"/>'
           '<geom name="distal" size=".01" pos="0 0 .2"/>' + wheels + '</body></worldbody>'
           '<actuator>' + actuators + '</actuator><equality>' + equality + '</equality></mujoco>')
    item = {'id': 'arbitrary_mobile_manipulator', 'kind': 'rover', 'body': 'base', 'freejoint': 'free_base',
            'params': {'free_base': True}, 'joints': {f'wheel_{i}': f'wheel_joint_{i}' for i in range(6)},
            'actuators': {f'wheel_{i}': f'wheel_motor_{i}' for i in range(6)},
            'actuator_modes': {f'wheel_{i}': 'velocity' for i in range(6)},
            'arm_interfaces': {'joint': {'engineered_pair_classes': [
                {'minimum_clearance_m': .001, 'fixed_or_proximal': ['proximal'], 'moving_or_distal': ['distal']}]}},
            'mechanisms': {}}
    if scissor:
        item['mechanisms']['jack'] = {'kind': 'scissor_jack_v1', 'equalities':
                                     dict(zip(('lower_scissor', 'upper_scissor', 'deck_slot'), ('lower', 'upper', 'deck')))}
    return mujoco.MjModel.from_xml_string(xml), {'id': 'unrelated_scene', 'institution': 'Synthetic test', 'equipment': [item]}


class NativeRoverProfileTests(unittest.TestCase):
    def test_valid_capability_ignores_all_identity_labels(self):
        model, manifest = fixture()
        before = deepcopy(manifest)
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model),
                         ROVER_PROFILE)
        manifest.update(id='different_lab', institution='Different institution', title='Different title')
        manifest['equipment'][0]['id'] = 'another_platform'
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model),
                         ROVER_PROFILE)
        self.assertEqual(before['equipment'][0]['joints'], manifest['equipment'][0]['joints'])

    def test_no_profile_for_ordinary_or_unqualified_platform(self):
        self.assertEqual(profiles.native_physics_options_for_manifest({'equipment': []}), {})
        for field, value in (('kind', 'ordinary_instrument'), ('params', {'free_base': False}), ('arm_interfaces', {})):
            model, manifest = fixture()
            manifest['equipment'][0][field] = value
            self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model), {})

    def test_selected_platform_requires_compiled_bindings(self):
        _, manifest = fixture()
        with self.assertRaisesRegex(ValueError, 'ownership'):
            profiles.native_physics_options_for_manifest(manifest)

    def test_missing_or_wrong_joint_bindings_fail(self):
        for mutation in (lambda item: item.update(freejoint='missing'),
                         lambda item: item.update(body='fixture'),
                         lambda item: item['joints'].pop('wheel_0'),
                         lambda item: item['joints'].update(wheel_0='free_base'),
                         lambda item: item['actuators'].update(wheel_0='wheel_motor_1'),
                         lambda item: item['actuator_modes'].update(wheel_0='position')):
            with self.subTest(mutation=mutation):
                model, manifest = fixture()
                mutation(manifest['equipment'][0])
                with self.assertRaises(ValueError):
                    profiles.native_physics_options_for_manifest(manifest, model)

    def test_duplicate_physical_wheel_bindings_fail(self):
        model, manifest = fixture()
        item = manifest['equipment'][0]
        item['joints']['wheel_0'] = item['joints']['wheel_1']
        item['actuators']['wheel_0'] = item['actuators']['wheel_1']
        with self.assertRaisesRegex(ValueError, 'distinct'):
            profiles.native_physics_options_for_manifest(manifest, model)

    def test_declared_interfaces_require_owned_real_surfaces_and_clearances(self):
        for field, value in (('minimum_clearance_m', float('nan')), ('minimum_clearance_m', True),
                             ('minimum_clearance_m', 0), ('fixed_or_proximal', []),
                             ('fixed_or_proximal', ['missing']), ('moving_or_distal', ['outside'])):
            with self.subTest(field=field, value=value):
                model, manifest = fixture()
                pair = manifest['equipment'][0]['arm_interfaces']['joint']['engineered_pair_classes'][0]
                pair[field] = value
                with self.assertRaises(ValueError):
                    profiles.native_physics_options_for_manifest(manifest, model)

    def test_combined_rover_scissor_profile_composes(self):
        model, manifest = fixture(scissor=True)
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model),
                         dict(ROVER_PROFILE, soft_connect_constraints=False))
        manifest['equipment'][0]['arm_interfaces'] = {}
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model),
                         {'native_substeps': 4, 'soft_connect_constraints': False})

    def test_observable_effort_checks_every_scene_actuator(self):
        for field, value in (('actuator_biastype', 2), ('actuator_dyntype', 1)):
            with self.subTest(field=field):
                model, manifest = fixture()
                getattr(model, field)[-1] = value
                with self.assertRaisesRegex(NotImplementedError, 'stateless affine'):
                    profiles.native_physics_options_for_manifest(manifest, model)
        model, manifest = fixture()
        model.actuator_gear[-1, 0] = 2.
        with self.assertRaisesRegex(NotImplementedError, 'unit scalar'):
            profiles.native_physics_options_for_manifest(manifest, model)

    def test_combining_profiles_does_not_bypass_connect_ownership(self):
        model, manifest = fixture(scissor=True)
        manifest['equipment'][0]['mechanisms']['jack']['equalities']['deck_slot'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unowned'):
            profiles.native_physics_options_for_manifest(manifest, model)

    def test_conflicting_profile_options_fail_instead_of_overwriting(self):
        model, manifest = fixture(scissor=True)
        with patch.object(profiles, '_rover_options', return_value={'native_substeps': 1}):
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                profiles.native_physics_options_for_manifest(manifest, model)


if __name__ == '__main__':
    unittest.main()
