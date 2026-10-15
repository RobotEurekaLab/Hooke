"""Capability-driven staging profile validation; no simulation acceptance claim."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import mujoco
import xml.etree.ElementTree as ET
from real_labs import native_profiles as profiles
from real_labs.builder import build_scene

def fixture(contact=""):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="seat_body"><geom name="seat" type="box" size=".1 .1 .01"/>
        <geom name="other_seat" type="box" size=".1 .1 .01" pos=".5 0 0"/></body>
      <geom name="static_sample" size=".01"/>
      <body name="rover"><freejoint/><geom name="rover_geom" size=".1"/></body>
      <body name="sample"><freejoint/><geom name="sample_geom" size=".01"/></body>
      <body name="moving_seat"><joint/><geom name="moving_geom" size=".01"/></body>
    </worldbody><contact>{contact}</contact></mujoco>''')
    item = {'sample_transfer_interface': {
        'kind': 'rigid_sample_staging_v1', 'sample_geom': 'sample_geom',
        'stage_seat_geom': 'seat', 'native_contact_offset_m': 0.00095}}
    def bound(kind, name):
        try:
            return int(getattr(model, kind)(name).id)
        except (KeyError, TypeError) as error:
            raise ValueError('Missing binding') from error
    owned = lambda body: body == model.body('rover').id
    return model, item, bound, owned


class TransferProfileTests(unittest.TestCase):
    def test_only_declared_static_seat_gets_override(self):
        model, item, bound, owned = fixture()
        before = deepcopy(item)
        self.assertEqual(profiles._sample_transfer_options(item, model, bound, owned),
                         {'contact_offset_overrides_m': {'seat': 0.00095}})
        self.assertEqual(item, before)
        self.assertEqual(profiles._sample_transfer_options({}, model, bound, owned), {})

    def test_invalid_binding_and_physical_ownership_fail(self):
        for key, value in (('kind', 'unknown'), ('sample_geom', 'missing'),
                           ('sample_geom', 'seat'), ('sample_geom', 'rover_geom'),
                           ('sample_geom', 'static_sample'), ('stage_seat_geom', 'missing'),
                           ('stage_seat_geom', 'moving_geom')):
            with self.subTest(key=key, value=value):
                model, item, bound, owned = fixture()
                item['sample_transfer_interface'][key] = value
                with self.assertRaises(ValueError):
                    profiles._sample_transfer_options(item, model, bound, owned)

    def test_invalid_numeric_offsets_fail(self):
        for value in (None, True, 0, -1, float('nan'), float('inf'), '0.001'):
            with self.subTest(value=value):
                model, item, bound, owned = fixture()
                item['sample_transfer_interface']['native_contact_offset_m'] = value
                with self.assertRaises(ValueError):
                    profiles._sample_transfer_options(item, model, bound, owned)

    def test_inactive_or_incompatible_surfaces_fail(self):
        for inactive in (True, False):
            model, item, bound, owned = fixture()
            seat, sample = model.geom('seat').id, model.geom('sample_geom').id
            model.geom_contype[seat] = 0 if inactive else 2
            model.geom_conaffinity[seat] = 0 if inactive else 2
            model.geom_contype[sample] = model.geom_conaffinity[sample] = 1
            with self.assertRaises(ValueError):
                profiles._sample_transfer_options(item, model, bound, owned)

    def test_body_exclusion_rejects_declared_transfer(self):
        args = fixture('<exclude body1="seat_body" body2="sample"/>')
        with self.assertRaisesRegex(ValueError, 'permit mutual contact'):
            profiles._sample_transfer_options(args[1], args[0], *args[2:])

    def test_exact_explicit_pair_overrides_body_exclusion_and_masks(self):
        model, item, bound, owned = fixture(
            '<exclude body1="seat_body" body2="sample"/>'
            '<pair geom1="seat" geom2="sample_geom"/>')
        for name in ('seat', 'sample_geom'):
            model.geom_contype[model.geom(name).id] = 0
            model.geom_conaffinity[model.geom(name).id] = 0
        self.assertEqual(profiles._sample_transfer_options(item, model, bound, owned),
                         {'contact_offset_overrides_m': {'seat': 0.00095}})

    def test_neighbor_explicit_pair_does_not_override_selected_exclusion(self):
        args = fixture('<exclude body1="seat_body" body2="sample"/>'
                       '<pair geom1="other_seat" geom2="sample_geom"/>')
        with self.assertRaisesRegex(ValueError, 'permit mutual contact'):
            profiles._sample_transfer_options(args[1], args[0], *args[2:])

    def test_whole_nasa_binding_resolves_exact_candidate(self):
        root, manifest = build_scene('nasa_planetary')
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        item = next(row for row in manifest['equipment'] if row['id'] == 'rover')
        expected = {'compound_cylinder_colliders': True,
                    'contact_offset_m': 5e-5, 'sdf_contact_offset_m': 5e-5,
                    'solver_type': 'PGS', 'simulation_threads': 4,
                    'explicit_affine_actuators': True, 'native_substeps': 4,
                    'contact_offset_overrides_m': {'nasa_task__stage_seat': 0.00095}}
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model), expected)
        manifest.update(id='unrelated_scene', institution='Any institution')
        item['id'] = 'unrelated_rover'
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model), expected)
        del item['sample_transfer_interface']
        expected.pop('contact_offset_overrides_m')
        self.assertEqual(profiles.native_physics_options_for_manifest(manifest, model), expected)

    def test_disjoint_seat_bindings_compose_and_conflicts_fail(self):
        manifest = {'equipment': [{}, {}]}
        with patch.object(profiles, '_rover_options', side_effect=[
                {'contact_offset_overrides_m': {'seat_a': .001}},
                {'contact_offset_overrides_m': {'seat_b': .002}}]):
            self.assertEqual(profiles.native_physics_options_for_manifest(manifest),
                             {'contact_offset_overrides_m': {'seat_a': .001, 'seat_b': .002}})
        with patch.object(profiles, '_rover_options', side_effect=[
                {'contact_offset_overrides_m': {'seat': .001}},
                {'contact_offset_overrides_m': {'seat': .002}}]):
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                profiles.native_physics_options_for_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
