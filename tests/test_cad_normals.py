"""An actual STEP box retains sharp face normals after placement and export."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))


@unittest.skipUnless(importlib.util.find_spec('OCP'), 'OpenCascade is an optional offline CAD dependency')
class CadSurfaceNormals(unittest.TestCase):
    def test_trimmed_cylinder_extent_and_installation_planes(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.gp import gp_Ax3, gp_Cylinder, gp_Dir, gp_Pnt
        from microscopy.cad_import import inspect_cylinders

        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            path = Path(folder)/'trimmed.step'
            # Axis origin Z=100 is not a physical shoulder of the Z=3..7 face.
            surface = gp_Cylinder(gp_Ax3(gp_Pnt(0, 0, 100), gp_Dir(0, 0, 1)), 12.7)
            face = BRepBuilderAPI_MakeFace(surface, 0., 2*np.pi, -97., -93.).Face()
            writer = STEPControl_Writer()
            writer.Transfer(face, STEPControl_AsIs)
            writer.Transfer(BRepPrimAPI_MakeCylinder(4., 9.).Shape(), STEPControl_AsIs)
            writer.Write(str(path))
            interfaces = inspect_cylinders(path)
            trimmed = next(item for item in interfaces['cylinders'] if abs(item['radius_mm']-12.7) < 1e-6)
            self.assertAlmostEqual(trimmed['point_mm'][2], 100.)
            np.testing.assert_allclose(np.array(trimmed['bounds_mm'])[[2, 5]], [3., 7.], atol=1e-6)
            self.assertAlmostEqual(trimmed['centre_mm'][2], 5.)
            self.assertAlmostEqual(trimmed['area_mm2'], 2*np.pi*12.7*4, places=5)
            shoulders = sorted(item['point_mm'][2] for item in interfaces['planes'])
            np.testing.assert_allclose(shoulders, [0., 9.], atol=1e-6)
            self.assertEqual(interfaces['units'], 'mm')

    def test_placed_step_box_keeps_hard_edges_correct_winding_and_si_units(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.TopLoc import TopLoc_Location
        from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
        from microscopy.cad_import import convert

        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            root = Path(folder)
            transform = gp_Trsf()
            transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(0,0,1)), .37)
            transform.SetTranslationPart(gp_Vec(100,200,300))
            shape = BRepPrimAPI_MakeBox(10,20,30).Shape().Located(TopLoc_Location(transform))
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            writer.Write(str(root/'box.step'))
            report = convert(root/'box.step', root/'mesh')
            part = report['parts'][0]
            text = (root/'mesh'/part['file']).read_text().splitlines()
            vertices = np.array([[float(v) for v in line.split()[1:]] for line in text if line.startswith('v ')])
            normals = np.array([[float(v) for v in line.split()[1:]] for line in text if line.startswith('vn ')])
            triangles = [[token.split('/') for token in line.split()[1:]] for line in text if line.startswith('f ')]
            self.assertEqual(len(report['parts']), 1)
            self.assertEqual(len(vertices), 24, 'Six CAD faces must retain separate corner normals')
            self.assertEqual(len(normals), len(vertices))
            self.assertTrue(part['watertight'])
            self.assertFalse(part['appearance_watertight'])
            np.testing.assert_allclose(vertices.min(axis=0)*1000, np.array(report['cad_bounds_mm'])[:3], atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-7)
            centre = vertices.mean(axis=0)
            for face in triangles:
                xyz = vertices[[int(item[0])-1 for item in face]]
                normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
                normal /= np.linalg.norm(normal)
                self.assertGreater(normal @ (xyz.mean(axis=0)-centre), 0, 'Triangles must face outward')
                for item in face:
                    self.assertGreater(normal @ normals[int(item[2])-1], .999999)


if __name__ == '__main__':
    unittest.main()
