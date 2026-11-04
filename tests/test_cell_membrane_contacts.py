"""Membrane appearance follows observed tool contacts in cell coordinates."""

from pathlib import Path
import sys
import unittest

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))

from microscopy.cell_appearance import cell_fields
from microscopy.cell_deformation import (
    handling_contact_descriptor, needle_contact_descriptor, needle_contact_offset,
)
from microscopy.cell_optics import CellMicroscope
from microscopy.tasks import make_task


class MembraneContactTests(unittest.TestCase):
    radii = np.full(3, 18e-6)

    def fields(self, x, y, contacts=(), slab=(0., 2e-6), **kwargs):
        return cell_fields(np.asarray([x])*1e-6, np.asarray([y])*1e-6,
                           self.radii, self.radii/3, index=0,
                           membrane_contacts=contacts, axial_slab=slab, **kwargs)

    def test_right_and_left_dents_follow_contact_and_leave_other_sides_unchanged(self):
        x, y = [17.6, -17.6, 0., 12.4], [0., 0., 17.6, 12.4]
        original = self.fields(x, y)[2]
        for sign, changed in ((1, 0), (-1, 1)):
            with self.subTest(side=sign):
                contact = ((sign*18e-6, 0., 0., 1e-6),)
                deformed = self.fields(x, y, contact)[2]
                self.assertLess(deformed[0, changed], .1*original[0, changed])
                unchanged = [i for i in range(4) if i != changed]
                np.testing.assert_array_equal(deformed[:, unchanged], original[:, unchanged])

    def test_contact_height_changes_only_the_intersecting_focal_slab(self):
        contact = ((0., 0., 18e-6, 1e-6),)
        for slab in ((0., 2e-6), (-17.5e-6, 1e-6)):
            np.testing.assert_array_equal(self.fields([0.], [0.], slab=slab)[2],
                                          self.fields([0.], [0.], contact, slab=slab)[2])
        original = self.fields([0.], [0.], slab=(17.5e-6, 1e-6))[2]
        deformed = self.fields([0.], [0.], contact, slab=(17.5e-6, 1e-6))[2]
        self.assertLess(deformed.item(), .1*original.item())

    def test_scalar_without_contact_cannot_invent_a_dent(self):
        for a, b in zip(self.fields([17.6, 12.4], [0., 12.4]),
                        self.fields([17.6, 12.4], [0., 12.4], indentation_m=2e-6)):
            np.testing.assert_array_equal(a, b)
        self.assertIsNone(needle_contact_offset([19e-6, 0., 0.], self.radii, [1., 0., 0.]))

    def test_off_axis_ellipsoid_contact_lies_on_actual_backward_needle_ray(self):
        radii = np.array([18., 14., 4.])*1e-6
        tip = np.array([7., -4., 1.])*1e-6
        back = np.array([1., .1, .2])
        back /= np.linalg.norm(back)
        contact = needle_contact_offset(tip, radii, back)
        self.assertAlmostEqual(np.linalg.norm(contact/radii), 1.)
        self.assertGreater((contact-tip)@back, 0.)
        np.testing.assert_allclose(np.cross(contact-tip, back), 0., atol=1e-20)
        descriptor = needle_contact_descriptor(contact, radii, back, .5e-6)
        normal = contact/radii**2
        normal /= np.linalg.norm(normal)
        self.assertAlmostEqual(descriptor["indentation_m"], .5e-6*(normal@back))

    def test_two_jaw_contacts_deform_both_actual_sides(self):
        descriptors = [handling_contact_descriptor([0., sign, 0.], self.radii, 1e-6)
                       for sign in (-1., 1.)]
        contacts = tuple((*d["offset_m"], d["indentation_m"]) for d in descriptors)
        original = self.fields([0., 0., 17.6], [17.6, -17.6, 0.])[2]
        deformed = self.fields([0., 0., 17.6], [17.6, -17.6, 0.], contacts)[2]
        self.assertTrue(np.all(deformed[0, :2] < .1*original[0, :2]))
        self.assertEqual(deformed[0, 2], original[0, 2])

    def test_projection_cache_includes_contact_position(self):
        scope = CellMicroscope()
        args = (0, -20e-6, 20e-6, 193, tuple(self.radii), tuple(self.radii/3))
        right = scope._projection(*args, ((18e-6, 0., 0., 1e-6),), (0., 2e-6))
        left = scope._projection(*args, ((-18e-6, 0., 0., 1e-6),), (0., 2e-6))
        self.assertGreater(np.max(np.abs(right[1]-left[1])), .5)
        self.assertIs(right, scope._projection(*args, ((18e-6, 0., 0., 1e-6),), (0., 2e-6)))

    def test_relaxation_anchor_moves_with_cell_and_not_with_withdrawn_needle(self):
        task = make_task("suction_injection")
        task.reset(0)
        mechanics = task.mechanics
        self.assertEqual(mechanics.image_state(task.data)["membrane_contacts"], [])
        # Contact history is cell-local; a residual indentation stays attached
        # to that membrane patch after the needle has moved away.
        mechanics._membrane_contact_offset_m = np.array([18e-6, 0., 0.])
        mechanics.indentation_m = .5e-6
        first = mechanics.image_state(task.data)
        mechanics.position_m += [5e-6, -3e-6, 2e-6]
        task.data.qpos[task.model.joint("injector_x").qposadr[0]] += 25e-6
        mujoco.mj_forward(task.model, task.data)
        second = mechanics.image_state(task.data)
        self.assertEqual(first["membrane_contacts"], second["membrane_contacts"])
        self.assertGreater(np.linalg.norm(np.asarray(second["tip_m"])-first["tip_m"]), 20e-6)
        np.testing.assert_allclose(np.asarray(second["centres_m"][0])-first["centres_m"][0],
                                   [5e-6, -3e-6, 2e-6], atol=1e-16)


if __name__ == "__main__":
    unittest.main()
