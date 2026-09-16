"""Experimental compiled-MuJoCo to USD bridge (import inside Isaac Sim).

The archive is the source of truth. Source IDs survive USD name normalization.
This exporter rejects unsupported topology instead of dropping joints silently.
A successfully exported stage is NOT a qualified replacement physics backend.
"""
from __future__ import annotations
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Vt, UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema
from backends.mass_properties import compose_welded_mass
from backends.collision_rules import collision_participants
from backends.contact_parameters import compliant_parameters,constraint_parameters


def quat(q):
    return Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:])))


def vec(p):
    return Gf.Vec3f(*map(float, p))


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(prim, p, q):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
    xf.AddOrientOp().Set(quat(q))


class SceneBridge:
    def __init__(self, stage, source, output, physics_options=None):
        self.stage = stage
        self.source, self.output = Path(source), Path(output)
        # NPZ is a compressed archive: indexing it repeatedly decompresses the
        # same fields. Materialize once for both export and every physics step.
        with np.load(self.source/'model.npz', allow_pickle=False) as archive:
            self.m = {key: archive[key] for key in archive.files}
        self.meta = json.loads((self.source/'scene.json').read_text())
        self.names = self.meta['names']
        self.body_paths = {i: f'/World/b{i}' for i in range(len(self.names['body']))}
        self.joint_paths = {}
        self.geom_paths = {}
        self.camera_paths = {}
        self.articulation_roots = []
        self.free_bodies = {}
        self.physics_options={
            # Millimeter contact skins bridge the submillimeter clearances
            # between the source tube's cap/body collision pieces. Use a
            # finer skin throughout scenes containing thread geometry.
            'contact_offset_m':float(os.environ.get('HOOKE_ISAAC_CONTACT_OFFSET',
                                      '.00005' if np.any(self.m['geom_type']==8) else '.001')),
            'sdf_contact_offset_m':float(os.environ.get('HOOKE_ISAAC_SDF_CONTACT_OFFSET','.00005')),
            'solver_velocity_iterations':int(os.environ.get('HOOKE_ISAAC_VELOCITY_ITERATIONS','8')),
            'compliant_contact_scale':float(os.environ.get('HOOKE_ISAAC_COMPLIANT_SCALE','1')),
            'compliant_impedance_fraction':float(os.environ.get('HOOKE_ISAAC_IMPEDANCE_FRACTION','0')),
            'soft_connect_constraints':os.environ.get('HOOKE_ISAAC_SOFT_CONNECT','1')=='1',
            'connect_impedance_fraction':float(os.environ.get('HOOKE_ISAAC_CONNECT_IMPEDANCE_FRACTION','0')),
            'approximate_cylinders':os.environ.get('HOOKE_ISAAC_APPROXIMATE_CYLINDERS','0')=='1',
        }
        if physics_options:
            unknown=set(physics_options)-self.physics_options.keys()
            if unknown:raise ValueError(f'Unknown physics options: {sorted(unknown)}')
            self.physics_options.update(physics_options)
        for key in ('contact_offset_m','sdf_contact_offset_m','compliant_contact_scale'):
            value=float(self.physics_options[key])
            if not np.isfinite(value) or value<0 or (key!='compliant_contact_scale' and value==0):
                raise ValueError(f'Invalid {key}')
            self.physics_options[key]=value
        iterations=self.physics_options['solver_velocity_iterations']
        if int(iterations)!=iterations or not 1<=iterations<=255:raise ValueError('Invalid solver iterations')
        if not isinstance(self.physics_options['approximate_cylinders'],bool):raise ValueError('Invalid cylinder approximation option')
        fraction=float(self.physics_options['compliant_impedance_fraction'])
        if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Invalid impedance fraction')
        fraction=float(self.physics_options['connect_impedance_fraction'])
        if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Invalid connect impedance fraction')
        if not isinstance(self.physics_options['soft_connect_constraints'],bool):raise ValueError('Invalid connect constraint option')
        self.limits = [
            'Unqualified: MuJoCo soft contacts/solref/solimp, noslip solver and elliptic friction differ from PhysX.',
            'Joint armature is applied through the PhysX tensor API; dry friction still requires calibration.',
            'Unqualified: lighting, reflectance and transparency use USD Preview Surface; no pixel-equivalence claim.',
            'Finite ground mesh uses the source plane display extent; infinite collision plane is not reproduced.',
            'Source camera pose/FOV retained; output resolution and exposure must be qualified separately.',
        ]

    def name(self, prim, kind, index):
        prim.CreateAttribute('hooke:sourceId', Sdf.ValueTypeNames.Int).Set(index)
        prim.CreateAttribute('hooke:sourceName', Sdf.ValueTypeNames.String).Set(self.names[kind][index] or '')

    def build(self):
        m, stage = self.m, self.stage
        if np.any(~np.isin(m['jnt_type'], [0, 2, 3])):
            raise NotImplementedError('Ball joints require an adapter.')
        if np.any(~np.isin(m['geom_type'], [0, 2, 3, 4, 5, 6, 7, 8])):
            raise NotImplementedError('Unsupported geometry type; export stopped.')
        plugins=ET.parse(self.source/'scene.xml').findall('./extension/plugin')
        self.plugins={p.get('plugin') for p in plugins}
        unsupported=self.plugins-{'mjlab.sdf.thread','mjlab.passive.detent'}
        if unsupported:raise NotImplementedError(f'Unsupported plugins: {sorted(unsupported)}')
        if any(np.any(m[field]>=0) for field in ('actuator_plugin','sensor_plugin')):
            raise NotImplementedError('Actuator/sensor plugins require an adapter.')
        if 'mjlab.sdf.thread' in self.plugins:
            self.limits.append('Thread collision uses a GPU mesh SDF (256 cells); analytic MuJoCo thread SDF equivalence is not qualified.')
        if 'mjlab.passive.detent' in self.plugins:
            self.limits.append('Passive detent forces must be evaluated by the source force adapter each step.')
        if self.physics_options['compliant_contact_scale']:
            self.limits.append('Compliant contacts use a calibrated constant source impedance; nonlinear residual dependence and coupled friction remain unqualified.')
        self.colliders,self.explicit_pairs=collision_participants(m)
        self.sdf_contacts={i for i in self.colliders if m['geom_type'][i]==8}
        for a,b in self.explicit_pairs:
            if m['geom_type'][a]==8 or m['geom_type'][b]==8:self.sdf_contacts.update((a,b))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage, '/World')
        UsdGeom.Scope.Define(stage, '/World/Joints')
        UsdGeom.Scope.Define(stage, '/World/Materials')
        # A moving source body may have zero mass and carry all its inertia
        # in welded descendants. Merge each moving weld group exactly;
        # skipping the massless joint owner would pin a free object to world.
        self.rigid = {i for i in range(1,len(self.names['body']))
                      if (m['body_weldid'][i]==0 and m['body_mass'][i]>0) or m['body_jntnum'][i]>0}
        self.mass_properties={}
        for i in self.rigid:
            members=np.flatnonzero(m['body_weldid']==i) if m['body_weldid'][i] else [i]
            self.mass_properties[i]=compose_welded_mass(m,i,members)
        for i in range(1,len(self.names['body'])):
            owner=int(m['body_weldid'][i])
            if owner and owner!=i:self.body_paths[i]=f'/World/b{owner}/b{i}'
        for i in range(len(self.names['body'])):
            prim = UsdGeom.Xform.Define(stage, self.body_paths[i]).GetPrim()
            owner=int(m['body_weldid'][i])
            if owner and owner!=i:
                p=rotation(m['reset_xquat'][owner]).T@(m['reset_xpos'][i]-m['reset_xpos'][owner])
                q=quat(m['reset_xquat'][owner]).GetInverse()*quat(m['reset_xquat'][i])
                pose(prim,p,[q.GetReal(),*q.GetImaginary()])
            else:pose(prim, m['reset_xpos'][i], m['reset_xquat'][i])
            self.name(prim, 'body', i)
            if i in self.rigid:
                UsdPhysics.RigidBodyAPI.Apply(prim)
                properties=self.mass_properties[i]
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(properties['mass'])
                mass.CreateCenterOfMassAttr(vec(properties['center']))
                mass.CreateDiagonalInertiaAttr(vec(properties['diagonal']))
                mass.CreatePrincipalAxesAttr(quat(properties['axes']))
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                comp=properties['gravcomp']
                rb.CreateDisableGravityAttr(bool(np.isclose(comp,1.)))
                rb.CreateLinearDampingAttr(0.)
                rb.CreateAngularDampingAttr(0.)
                if not (np.isclose(comp,0.) or np.isclose(comp,1.)):
                    raise NotImplementedError('Partial gravity compensation needs force integration.')
        for i in range(len(self.names['geom'])):
            self.geometry(i)
        self.joints()
        if len(self.joint_paths)+len(self.free_bodies)!=len(m['jnt_type']):
            raise ValueError('A source joint was omitted from the physical scene')
        self.equalities(m.get('reset_eq_active',m['eq_active0']))
        self.filters()
        self.cameras()
        for i in range(len(self.names['site'])):
            path = self.body_paths[int(m['site_bodyid'][i])] + f'/site{i}'
            marker = UsdGeom.Xform.Define(stage, path).GetPrim()
            pose(marker, m['site_pos'][i], m['site_quat'][i])
            self.name(marker, 'site', i)
        light = UsdLux.DistantLight.Define(stage, '/World/Sun')
        light.CreateIntensityAttr(1500.)
        light.CreateAngleAttr(.5)
        light.CreateColorAttr(Gf.Vec3f(.8, .8, .8))
        UsdLux.DomeLight.Define(stage, '/World/Ambient').CreateIntensityAttr(450.)
        report = {'status': 'EXPERIMENTAL_UNQUALIFIED', 'source': str(self.source),'conversion_version':6,
                  'body_count': len(self.body_paths), 'geom_count': len(self.geom_paths),
                  'joint_count': len(self.joint_paths), 'camera_count': len(self.camera_paths),
                  'free_joints': self.free_bodies,
                  'instantiated_source_joints':len(self.joint_paths)+len(self.free_bodies),
                  'welded_mass_groups':{i:{'mass':p['mass'],'members':list(map(int,p['members']))} for i,p in self.mass_properties.items()},
                  'site_frame_count': len(self.names['site']),
                  'source_equalities': len(m['eq_type']), 'source_tendons': len(m['tendon_adr']),
                  'physics_options':self.physics_options,
                  'explicit_contact_pairs':sorted(self.explicit_pairs),
                  'collider_count':len(self.colliders),
                  'articulation_roots': self.articulation_roots,
                  'body_paths': self.body_paths, 'geom_paths': self.geom_paths,
                  'joint_paths': self.joint_paths, 'camera_paths': self.camera_paths,
                  'limitations': self.limits}
        (self.output/'conversion.json').write_text(json.dumps(report, indent=2))
        return report

    def material(self, i, prim):
        m = self.m
        mat_id = int(m['geom_matid'][i])
        rgba = m['geom_rgba'][i].copy()
        # MuJoCo uses material RGBA only when geom RGBA has the default value.
        if mat_id >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
            rgba = m['mat_rgba'][mat_id].copy()
        material = UsdShade.Material.Define(self.stage, f'/World/Materials/g{i}')
        shader = UsdShade.Shader.Define(self.stage, f'/World/Materials/g{i}/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(vec(rgba[:3]))
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(
            float(np.clip(1 - m['mat_shininess'][mat_id], .05, 1.)) if mat_id >= 0 else .7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        # Preserve source-generated texture pixels. Ground UVs are generated below.
        tex_ids = m['mat_texid'][mat_id] if mat_id >= 0 else []
        tex_id = int(tex_ids[1]) if len(tex_ids) > 1 else -1
        if tex_id >= 0 and m['tex_type'][tex_id] == 0:
            from PIL import Image
            w, h, c = [int(m[k][tex_id]) for k in ('tex_width','tex_height','tex_nchannel')]
            start = int(m['tex_adr'][tex_id])
            pixels = m['tex_data'][start:start+w*h*c].reshape(h,w,c)
            folder = self.output/'textures'; folder.mkdir(exist_ok=True)
            file = folder/f't{tex_id}.png'; Image.fromarray(pixels).save(file)
            uv = UsdShade.Shader.Define(self.stage, f'/World/Materials/g{i}/UV')
            uv.CreateIdAttr('UsdPrimvarReader_float2')
            uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            tex = UsdShade.Shader.Define(self.stage, f'/World/Materials/g{i}/Texture')
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(file.resolve()))
            tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
            tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
            shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(m['geom_friction'][i, 0]))
        physics.CreateDynamicFrictionAttr(float(m['geom_friction'][i, 0]))
        physics.CreateRestitutionAttr(0.)
        native_material=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        native_material.CreateFrictionCombineModeAttr('max')
        scale=self.physics_options['compliant_contact_scale']
        if scale and i in self.colliders:
            stiffness,damping=compliant_parameters(m,i,self.meta['timestep_s'],self.physics_options['compliant_impedance_fraction'])
            native_material.CreateCompliantContactAccelerationSpringAttr(True)
            native_material.CreateCompliantContactStiffnessAttr(stiffness*scale)
            native_material.CreateCompliantContactDampingAttr(damping*np.sqrt(scale))
            # PhysX stores compliant stiffness as negative restitution:
            # min selects the stiffer material, including dedicated pairs.
            native_material.CreateRestitutionCombineModeAttr('min')
            native_material.CreateDampingCombineModeAttr('max')
        UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')

    def geometry(self, i):
        m, stage = self.m, self.stage
        typ = int(m['geom_type'][i]); size = m['geom_size'][i]
        path = self.body_paths[int(m['geom_bodyid'][i])] + f'/g{i}'
        self.geom_paths[i] = path
        if typ in (0, 7, 8):
            g = UsdGeom.Mesh.Define(stage, path)
            g.CreateSubdivisionSchemeAttr('none')
            if typ in (7, 8):
                mid = int(m['geom_dataid'][i])
                va, vn, fa, fn = [int(m[k][mid]) for k in ('mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum')]
                points = m['mesh_vert'][va:va+vn]
                faces = m['mesh_face'][fa:fa+fn]
                g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, np.int32)))
                g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
                na = int(m['mesh_normaladr'][mid]); ni = m['mesh_facenormal'][fa:fa+fn].flatten()
                if np.all(ni >= 0):
                    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(m['mesh_normal'][na+ni].astype(np.float32)))
                    g.SetNormalsInterpolation('faceVarying')
                ta = int(m['mesh_texcoordadr'][mid]); ti = m['mesh_facetexcoord'][fa:fa+fn].flatten()
                if ta >= 0 and np.all(ti >= 0):
                    from backends.texture_mapping import source_uv_to_usd
                    UsdGeom.PrimvarsAPI(g).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying').Set(
                        Vt.Vec2fArray.FromNumpy(source_uv_to_usd(m['mesh_texcoord'][ta+ti])))
            else:
                sx, sy = (float(s) if s > 0 else 5. for s in size[:2])
                g.CreatePointsAttr([(-sx,-sy,0),(sx,-sy,0),(sx,sy,0),(-sx,sy,0)])
                g.CreateFaceVertexCountsAttr([4]); g.CreateFaceVertexIndicesAttr([0,1,2,3])
                mat = int(m['geom_matid'][i]); repeat = m['mat_texrepeat'][mat] if mat >= 0 else [1,1]
                from backends.texture_mapping import plane_uv
                coordinates=plane_uv([(-sx,-sy),(sx,-sy),(sx,sy),(-sx,sy)],
                                     [sx,sy],repeat,mat>=0 and bool(m['mat_texuniform'][mat]))
                UsdGeom.PrimvarsAPI(g).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
                    Vt.Vec2fArray.FromNumpy(coordinates))
            g.CreateDoubleSidedAttr(True)
        elif typ == 6:
            g = UsdGeom.Cube.Define(stage, path); g.CreateSizeAttr(2.)
        elif typ in (2, 4):
            g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(1. if typ == 4 else float(size[0]))
        else:
            g = (UsdGeom.Capsule if typ == 3 else UsdGeom.Cylinder).Define(stage, path)
            g.CreateRadiusAttr(float(size[0])); g.CreateHeightAttr(2*float(size[1])); g.CreateAxisAttr('Z')
        prim = g.GetPrim()
        pose(prim, m['geom_pos'][i], m['geom_quat'][i])
        if typ in (4,6):
            UsdGeom.Xformable(prim).AddScaleOp().Set(vec(size))
        self.name(prim, 'geom', i)
        self.material(i, prim)
        # Match default MuJoCo visual groups 0, 1, 2; retain hidden collision geometry.
        if m['geom_group'][i] > 2:
            UsdGeom.Imageable(prim).MakeInvisible()
        if i in self.colliders:
            UsdPhysics.CollisionAPI.Apply(prim)
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            offset=self.physics_options['sdf_contact_offset_m'] if i in self.sdf_contacts else self.physics_options['contact_offset_m']
            api.CreateContactOffsetAttr(max(offset, float(m['geom_margin'][i])))
            api.CreateRestOffsetAttr(0.)
            if typ == 7:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            elif typ == 8:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(256)

    def parent_rigid(self, body):
        p = int(self.m['body_parentid'][body])
        while p and p not in self.rigid:
            p = int(self.m['body_parentid'][p])
        return p

    def joints(self):
        m, stage = self.m, self.stage
        for b in sorted(self.rigid):
            parent = self.parent_rigid(b)
            j = int(m['body_jntadr'][b]) if m['body_jntnum'][b] else -1
            if j >= 0 and m['jnt_type'][j] == 0:
                # A MuJoCo free joint is a floating rigid body, not six USD
                # hinges. Its pose and twist are read from PhysX rigid bodies.
                self.free_bodies[j] = b
                continue
            count=int(m['body_jntnum'][b])
            if count>1:
                # MuJoCo applies multiple joints on a body in declaration
                # order. A serial chain preserves their distinct anchors.
                previous=self.body_paths[parent] if parent else None
                for offset in range(count):
                    index=j+offset
                    terminal=offset==count-1
                    child=self.body_paths[b] if terminal else f'/World/compound{b}_{offset}'
                    if not terminal:
                        prim=UsdGeom.Xform.Define(stage,child).GetPrim()
                        pose(prim,m['reset_xpos'][b],m['reset_xquat'][b])
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                        mass=UsdPhysics.MassAPI.Apply(prim)
                        mass.CreateMassAttr(1e-6)
                        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9))
                        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                    self.scalar_joint(index,b,parent if offset==0 else b,previous,child)
                    previous=child
                self.limits.append(f'Compound body {b} uses {count-1} auxiliary links, each adding 1e-6 kg and 1e-9 kg m².')
                continue
            path = f'/World/Joints/j{j}' if j >= 0 else f'/World/Joints/fixed{b}'
            schema = UsdPhysics.FixedJoint if j < 0 else (UsdPhysics.RevoluteJoint if m['jnt_type'][j] == 3 else UsdPhysics.PrismaticJoint)
            joint = schema.Define(stage, path)
            if parent:
                joint.CreateBody0Rel().SetTargets([self.body_paths[parent]])
            root_path=path
            if not parent and j>=0:
                # PhysX requires a fixed root link before an articulation's
                # first scalar DOF. This world-fixed frame adds no moving
                # mass and preserves a source hinge/slide directly on world.
                anchor=f'/World/rootAnchor{b}'
                prim=UsdGeom.Xform.Define(stage,anchor).GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass=UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.))
                root_path=f'/World/Joints/rootAnchor{b}'
                fixed=UsdPhysics.FixedJoint.Define(stage,root_path)
                fixed.CreateBody1Rel().SetTargets([anchor])
                joint.CreateBody0Rel().SetTargets([anchor])
            joint.CreateBody1Rel().SetTargets([self.body_paths[b]])
            rp = rotation(m['reference_xquat'][parent]); rb = rotation(m['reference_xquat'][b])
            pp, pb = m['reference_xpos'][parent], m['reference_xpos'][b]
            anchor = m['jnt_pos'][j] if j >= 0 else np.zeros(3)
            local0 = rp.T @ (pb + rb@anchor - pp)
            align = Gf.Rotation(Gf.Vec3d(1,0,0), Gf.Vec3d(*map(float,m['jnt_axis'][j]))).GetQuat() if j >= 0 else Gf.Quatd(1.)
            joint.CreateLocalPos0Attr(vec(local0)); joint.CreateLocalPos1Attr(vec(anchor))
            q0 = quat(m['reference_xquat'][parent]).GetInverse()*quat(m['reference_xquat'][b])*Gf.Quatf(align)
            joint.CreateLocalRot0Attr(q0); joint.CreateLocalRot1Attr(Gf.Quatf(align))
            joint.CreateCollisionEnabledAttr(False)
            if j >= 0:
                self.joint_paths[j] = path; self.name(joint.GetPrim(), 'joint', j)
                joint.CreateAxisAttr('X')
                factor = 180/math.pi if m['jnt_type'][j] == 3 else 1.
                if m['jnt_limited'][j]:
                    reference=float(m['qpos0'][m['jnt_qposadr'][j]])
                    joint.CreateLowerLimitAttr(float((m['jnt_range'][j,0]-reference)*factor))
                    joint.CreateUpperLimitAttr(float((m['jnt_range'][j,1]-reference)*factor))
                PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr(0.)
            if not parent:
                # Each independent source tree becomes its own articulation.
                root=stage.GetPrimAtPath(root_path)
                UsdPhysics.ArticulationRootAPI.Apply(root)
                art = PhysxSchema.PhysxArticulationAPI.Apply(root)
                art.CreateEnabledSelfCollisionsAttr(True)
                art.CreateSolverPositionIterationCountAttr(32)
                art.CreateSolverVelocityIterationCountAttr(self.physics_options['solver_velocity_iterations'])
                self.articulation_roots.append({'body': b, 'body_path': self.body_paths[b], 'root_path': root_path})

    def scalar_joint(self,j,body,parent,parent_path,child_path):
        m=self.m
        path=f'/World/Joints/j{j}'
        schema=UsdPhysics.RevoluteJoint if m['jnt_type'][j]==3 else UsdPhysics.PrismaticJoint
        joint=schema.Define(self.stage,path)
        if parent_path:joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        rp=rotation(m['reference_xquat'][parent]);rb=rotation(m['reference_xquat'][body])
        anchor=m['jnt_pos'][j]
        local=rp.T@(m['reference_xpos'][body]+rb@anchor-m['reference_xpos'][parent])
        align=Gf.Quatf(Gf.Rotation(Gf.Vec3d(1,0,0),Gf.Vec3d(*map(float,m['jnt_axis'][j]))).GetQuat())
        joint.CreateLocalPos0Attr(vec(local));joint.CreateLocalPos1Attr(vec(anchor))
        joint.CreateLocalRot0Attr(quat(m['reference_xquat'][parent]).GetInverse()*quat(m['reference_xquat'][body])*align)
        joint.CreateLocalRot1Attr(align);joint.CreateAxisAttr('X');joint.CreateCollisionEnabledAttr(False)
        self.joint_paths[j]=path;self.name(joint.GetPrim(),'joint',j)
        if m['jnt_limited'][j]:
            factor=180/math.pi if m['jnt_type'][j]==3 else 1.
            ref=m['qpos0'][m['jnt_qposadr'][j]]
            joint.CreateLowerLimitAttr(float((m['jnt_range'][j,0]-ref)*factor))
            joint.CreateUpperLimitAttr(float((m['jnt_range'][j,1]-ref)*factor))

    def equalities(self, active=None, previous=None):
        m = self.m
        active=m['eq_active0'] if active is None else np.asarray(active,dtype=bool)
        for i, typ in enumerate(m['eq_type']):
            if previous is not None and bool(active[i])==bool(previous[i]):continue
            a, b = int(m['eq_obj1id'][i]), int(m['eq_obj2id'][i])
            if typ==2 and b<0:
                # An instrument can engage its lid lock during a task.
                # Keep the original articulation DOF and constrain it with an
                # external fixed joint; changing its DOF count invalidates
                # the controller's tensor handles.
                existing=self.stage.GetPrimAtPath(f'/World/Joints/lock{i}')
                if existing:
                    UsdPhysics.Joint(existing).GetJointEnabledAttr().Set(bool(active[i]))
                    continue
                source=UsdPhysics.Joint(self.stage.GetPrimAtPath(self.joint_paths[a]))
                lock=UsdPhysics.FixedJoint.Define(self.stage,f'/World/Joints/lock{i}')
                lock.CreateBody0Rel().SetTargets(source.GetBody0Rel().GetTargets())
                lock.CreateBody1Rel().SetTargets(source.GetBody1Rel().GetTargets())
                p0=source.GetLocalPos0Attr().Get();r0=source.GetLocalRot0Attr().Get()
                value=float(m['eq_data'][i,0])
                if m['jnt_type'][a]==3:
                    r0=r0*Gf.Quatf(Gf.Rotation(Gf.Vec3d(1,0,0),value*180/math.pi).GetQuat())
                else:p0=p0+r0.Transform(Gf.Vec3f(value,0,0))
                lock.CreateLocalPos0Attr(p0);lock.CreateLocalRot0Attr(r0)
                lock.CreateLocalPos1Attr(source.GetLocalPos1Attr().Get())
                lock.CreateLocalRot1Attr(source.GetLocalRot1Attr().Get())
                lock.CreateExcludeFromArticulationAttr(True)
                lock.CreateJointEnabledAttr(bool(active[i]))
                continue
            if not active[i]:
                if typ == 0:
                    prim=self.stage.GetPrimAtPath(f'/World/Joints/closure{i}')
                    if prim:UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                elif typ == 2 and a in self.joint_paths:
                    axis='rotX'
                    self.stage.GetPrimAtPath(self.joint_paths[a]).RemoveAPI(PhysxSchema.PhysxMimicJointAPI,axis)
                continue
            if typ == 0:
                soft=self.physics_options['soft_connect_constraints']
                schema=UsdPhysics.Joint if soft else UsdPhysics.SphericalJoint
                joint = schema.Define(self.stage, f'/World/Joints/closure{i}')
                joint.CreateJointEnabledAttr(True)
                def frame(body,anchor):
                    owner=int(m['body_weldid'][body]) or body
                    local=rotation(m['reference_xquat'][owner]).T@(m['reference_xpos'][body]+rotation(m['reference_xquat'][body])@anchor-m['reference_xpos'][owner])
                    return self.body_paths[owner] if body else None,vec(local)
                pa,xa=frame(a,m['eq_data'][i,:3]);pb,xb=frame(b,m['eq_data'][i,3:6])
                if pa:joint.CreateBody0Rel().SetTargets([pa])
                if pb:joint.CreateBody1Rel().SetTargets([pb])
                joint.CreateLocalPos0Attr(xa);joint.CreateLocalPos1Attr(xb)
                joint.CreateExcludeFromArticulationAttr(True)
                if soft:
                    for axis in ('transX','transY','transZ','rotX','rotY','rotZ'):
                        limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                        # USD low > high locks an axis. Leave all six free;
                        # only the three linear springs constrain the anchor.
                        limit.CreateLowAttr(-float('inf'));limit.CreateHighAttr(float('inf'))
                    stiffness,damping=constraint_parameters(m['eq_solref'][i],m['eq_solimp'][i],
                        self.meta['timestep_s'],self.physics_options['connect_impedance_fraction'])
                    inverse_weight=float(m['body_invweight0'][a,0]+m['body_invweight0'][b,0])
                    if inverse_weight<=0:raise ValueError('A compliant connect requires a dynamic body')
                    for axis in ('transX','transY','transZ'):
                        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
                        drive.CreateTypeAttr('force')
                        drive.CreateStiffnessAttr(stiffness/inverse_weight)
                        drive.CreateDampingAttr(damping/inverse_weight)
                        drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
                    note='Experimental: connect equality uses force springs from source reference mass and impedance; nonlinear residual dependence remains unqualified.'
                    if note not in self.limits:self.limits.append(note)
            elif typ == 2 and np.allclose(m['eq_data'][i,2:5], 0) and b >= 0:
                for j in (a,b):
                    if m['jnt_type'][j]==3 and not m['jnt_limited'][j]:
                        # PhysX rejects continuous revolute mimic joints.
                        # Its documented workaround is finite bounds outside
                        # any reachable range; retain unwrapped joint angles.
                        joint=UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath(self.joint_paths[j]))
                        joint.CreateLowerLimitAttr(-1e10);joint.CreateUpperLimitAttr(1e10)
                        note=f'Continuous mimic joint {j} uses PhysX bounds ±1e10 degrees (source is unlimited).'
                        if note not in self.limits:self.limits.append(note)
                # PhysX 4.5's mimic API accepts only rotX/rotY/rotZ instance
                # names, including on a one-DOF prismatic joint. A transX
                # schema is silently ignored by the parser. The token selects
                # the sole DOF; it does not change its linear units.
                axis='rotX'
                reference_axis='rotX'
                factor_a=180/math.pi if m['jnt_type'][a]==3 else 1.
                factor_b=180/math.pi if m['jnt_type'][b]==3 else 1.
                coefficient=float(m['eq_data'][i,1])
                # MuJoCo equality polynomials already use q - qpos0,
                # which is also the coordinate of the exported USD joint.
                constant=float(m['eq_data'][i,0])
                api = PhysxSchema.PhysxMimicJointAPI.Apply(self.stage.GetPrimAtPath(self.joint_paths[a]), axis)
                api.GetReferenceJointRel().SetTargets([self.joint_paths[b]])
                api.GetReferenceJointAxisAttr().Set(reference_axis)
                api.GetGearingAttr().Set(-coefficient*factor_a/factor_b)
                api.GetOffsetAttr().Set(-constant*factor_a)
            else:
                raise NotImplementedError(f'Equality {i}, type {typ} requires an adapter.')

    def filters(self):
        """Preserve source collision rules without quadratic visual-geom lists."""
        from collections import defaultdict
        m=self.m
        excludes=set(map(int,m['exclude_signature']))
        colliders=defaultdict(list)
        for i in self.geom_paths:
            if i in self.colliders:
                colliders[int(m['geom_bodyid'][i])].append(i)
        body_filters=defaultdict(list);geom_filters=defaultdict(list)
        bodies=sorted(colliders)
        for index,a in enumerate(bodies):
            for b in bodies[index+1:]:
                ga=np.asarray(colliders[a],dtype=int);gb=np.asarray(colliders[b],dtype=int)
                wa,wb=int(m['body_weldid'][a]),int(m['body_weldid'][b])
                pa=int(m['body_weldid'][m['body_parentid'][wa]])
                pb=int(m['body_weldid'][m['body_parentid'][wb]])
                same=wa==wb
                adjacent=wa!=0 and wb!=0 and (wa==pb or wb==pa)
                if same or adjacent or ((a<<16)+b) in excludes:
                    allowed=np.zeros((len(ga),len(gb)),dtype=bool)
                else:
                    allowed=((m['geom_contype'][ga,None]&m['geom_conaffinity'][gb])!=0)|((m['geom_conaffinity'][ga,None]&m['geom_contype'][gb])!=0)
                # Explicit MJCF contact pairs bypass the collision masks.
                # Thread and dedicated grasp colliders often use zero masks
                # specifically so they collide only through these pairs.
                if self.explicit_pairs:
                    ia={int(g):k for k,g in enumerate(ga)};ib={int(g):k for k,g in enumerate(gb)}
                    for g1,g2 in self.explicit_pairs:
                        if g1 in ia and g2 in ib:allowed[ia[g1],ib[g2]]=True
                        if g2 in ia and g1 in ib:allowed[ia[g2],ib[g1]]=True
                if allowed.all():continue
                if not allowed.any() and a in self.rigid and b in self.rigid:
                    body_filters[a].append(self.body_paths[b])
                else:
                    for ai,bi in zip(*np.where(~allowed)):
                        geom_filters[int(ga[ai])].append(self.geom_paths[int(gb[bi])])
        for body,targets in body_filters.items():
            UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(self.body_paths[body])).CreateFilteredPairsRel().SetTargets(targets)
        for geom,targets in geom_filters.items():
            UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(self.geom_paths[geom])).CreateFilteredPairsRel().SetTargets(targets)

    def cameras(self):
        m = self.m
        for i in range(len(self.names['camera'])):
            if m['cam_mode'][i] != 0:
                raise NotImplementedError('Tracking cameras require a runtime adapter.')
            path = self.body_paths[int(m['cam_bodyid'][i])] + f'/camera{i}'
            camera = UsdGeom.Camera.Define(self.stage, path)
            pose(camera.GetPrim(), m['cam_pos'][i], m['cam_quat'][i])
            self.name(camera.GetPrim(), 'camera', i)
            resolution = m['cam_resolution'][i]
            aspect = float(resolution[0]/resolution[1]) if resolution[1] else 4/3
            focal = 24.
            vertical = 2*focal*math.tan(math.radians(float(m['cam_fovy'][i]))/2)
            camera.CreateFocalLengthAttr(focal)
            camera.CreateVerticalApertureAttr(vertical)
            camera.CreateHorizontalApertureAttr(vertical*aspect)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
            self.camera_paths[i] = path
