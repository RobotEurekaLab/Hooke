"""Run with the installed Isaac USD bindings; a source fixture is optional."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Hooke'))
try:
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    from backends.usd_visuals import apply_visuals
except ImportError:
    Usd=None


@unittest.skipIf(Usd is None,'Isaac USD bindings are not available in this Python')
class VisualContracts(unittest.TestCase):
    def test_zero_alpha_hides_source_surface_and_preserves_its_collider(self):
        fixture=os.environ.get('HOOKE_TEST_USD_SOURCE')
        if not fixture:self.skipTest('Set HOOKE_TEST_USD_SOURCE to a compiled source snapshot')
        from backends.usd_scene import SceneBridge

        with tempfile.TemporaryDirectory() as output:
            stage=Usd.Stage.CreateInMemory()
            bridge=SceneBridge(stage,Path(fixture),Path(output))
            bridge.build()
            index=next(i for i in sorted(bridge.colliders) if bridge.m['geom_matid'][i]>=0)
            material=int(bridge.m['geom_matid'][index])
            for rgba,material_alpha,visibility in (
                    ([.1,.2,.3,0.],1.,UsdGeom.Tokens.invisible),
                    ([.5,.5,.5,1.],0.,UsdGeom.Tokens.invisible),
                    ([.1,.2,.3,1.],0.,UsdGeom.Tokens.inherited)):
                bridge.m['geom_rgba'][index]=rgba
                bridge.m['mat_rgba'][material,3]=material_alpha
                bridge.geometry(index)
                prim=stage.GetPrimAtPath(bridge.geom_paths[index])
                self.assertEqual(UsdGeom.Imageable(prim).GetVisibilityAttr().Get(),visibility)
                self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
                stage.RemovePrim(prim.GetPath())

    def test_cell_and_nucleus_keep_their_micrometre_bounds_and_alpha(self):
        stage=Usd.Stage.CreateInMemory();cache={}
        position=np.array([.01,.02,.03])
        geometry=[dict(type=4,role=role,size=[radius]*3,pos=position.tolist(),
                       mat=np.eye(3).ravel().tolist(),rgba=[.5,.7,.6,alpha],
                       surface=dict(roughness=.8,specular_color=[.015]*3))
                  for role,radius,alpha in (('cell_shell',18e-6,.55),('cell_nucleus',6e-6,.9))]
        result=apply_visuals(stage,{},dict(geometry=geometry),cache)
        self.assertEqual(result['liquid_surfaces'],0)
        for i,shape in enumerate(geometry):
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(f'/World/RuntimeVisuals/geometry{i}'))
            points=np.asarray(mesh.GetPointsAttr().Get())
            faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            triangles=points[faces]
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            self.assertTrue(np.all(np.sum(normals*triangles.mean(axis=1),axis=1)>0))
            np.testing.assert_allclose(np.linalg.norm(points,axis=1),1.,atol=1e-7)
            bound=UsdGeom.BBoxCache(0.,[UsdGeom.Tokens.default_]).ComputeWorldBound(mesh.GetPrim()).ComputeAlignedRange()
            np.testing.assert_allclose(bound.GetSize(),2*np.asarray(shape['size']),atol=1e-12)
            np.testing.assert_allclose((np.asarray(bound.GetMin())+bound.GetMax())/2,position,atol=1e-12)
            shader=UsdShade.Shader.Get(stage,f'/World/RuntimeVisuals/material{i}/Coverage')
            self.assertAlmostEqual(shader.GetInput('opacity_constant').Get(),shape['rgba'][3],places=6)
            self.assertTrue(shader.GetInput('enable_opacity').Get())
            self.assertFalse(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        geometry[0]['rgba'][3]=0.
        apply_visuals(stage,{},dict(geometry=geometry),cache)
        self.assertEqual(UsdGeom.Imageable(cache[0][0]).GetVisibilityAttr().Get(),UsdGeom.Tokens.invisible)
        geometry[0]['rgba'][3]=.55
        apply_visuals(stage,{},dict(geometry=geometry),cache)
        self.assertEqual(UsdGeom.Imageable(cache[0][0]).GetVisibilityAttr().Get(),UsdGeom.Tokens.inherited)

    def test_legacy_liquid_and_hose_are_counted_separately(self):
        stage=Usd.Stage.CreateInMemory();cache={}
        cylinder=dict(type=5,size=[.001,.01,0.],pos=[0.,0.,0.],mat=np.eye(3).ravel().tolist(),rgba=[0.,0.,1.,1.])
        result=apply_visuals(stage,{},dict(geometry=[cylinder,dict(cylinder,role='pneumatic_hose')]),cache)
        self.assertEqual(result['liquid_surfaces'],1)
        self.assertEqual(result['runtime_geometries'],2)
        self.assertFalse(stage.GetPrimAtPath('/World/RuntimeVisuals/material0/Coverage'))


if __name__=='__main__':unittest.main()
