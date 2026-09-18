"""USD display textures and analytical visual shapes; no collision APIs."""
from pathlib import Path
import numpy as np
from pxr import UsdGeom,UsdShade,Sdf,Gf,Vt


def cell_mesh(stage, path):
    """Smooth unit sphere for the analytical cell ellipsoid, without physics APIs."""
    longitudes=64;latitudes=32
    theta=np.arange(1,latitudes)*np.pi/latitudes
    phi=np.arange(longitudes)*2*np.pi/longitudes
    rings=np.stack(np.broadcast_arrays(np.sin(theta)[:,None]*np.cos(phi),
                                      np.sin(theta)[:,None]*np.sin(phi),
                                      np.cos(theta)[:,None]),axis=-1).reshape(-1,3)
    points=np.vstack(([0.,0.,1.],rings,[0.,0.,-1.])).astype(np.float32)
    faces=[]
    for j in range(longitudes):
        following=(j+1)%longitudes
        faces.append((0,1+j,1+following))
        for ring in range(latitudes-2):
            a=1+ring*longitudes+j;b=1+ring*longitudes+following
            faces.extend(((a,a+longitudes,b),(b,a+longitudes,b+longitudes)))
        last=1+(latitudes-2)*longitudes
        faces.append((len(points)-1,last+following,last+j))
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(-1.),Gf.Vec3f(1.)])
    return mesh


def apply_surface(shader, surface, rgba):
    """Use fractional coverage for illustrative alpha, without glass refraction.

    The portable preview remains available. The installed RTX MDL implements
    alpha coverage; these parameters do not specify biological optical indices.
    """
    color=Gf.Vec3f(*map(float,rgba[:3]))
    specular=Gf.Vec3f(*map(float,surface['specular_color']))
    roughness=float(surface['roughness'])
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(specular)
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.)
    material=UsdShade.Material(shader.GetPrim().GetParent())
    path=material.GetPath().AppendChild('Coverage')
    mdl=UsdShade.Shader.Get(shader.GetPrim().GetStage(),path)
    if not mdl:
        mdl=UsdShade.Shader.Define(shader.GetPrim().GetStage(),path)
        mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        mdl.SetSourceAsset('OmniPBR.mdl','mdl')
        mdl.SetSourceAssetSubIdentifier('OmniPBR','mdl')
        material.CreateSurfaceOutput('mdl').ConnectToSource(mdl.ConnectableAPI(),'out')
        mdl.CreateInput('enable_opacity',Sdf.ValueTypeNames.Bool).Set(True)
        mdl.CreateInput('opacity_threshold',Sdf.ValueTypeNames.Float).Set(0.)
        mdl.CreateInput('metallic_constant',Sdf.ValueTypeNames.Float).Set(0.)
    mdl.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(color)
    mdl.CreateInput('reflection_roughness_constant',Sdf.ValueTypeNames.Float).Set(roughness)
    mdl.CreateInput('specular_level',Sdf.ValueTypeNames.Float).Set(float(np.mean(specular)))
    mdl.CreateInput('opacity_constant',Sdf.ValueTypeNames.Float).Set(float(rgba[3]))


def apply_visuals(stage, model, visuals, geometry_cache):
    textures=visuals.get('textures',[])
    for texture in textures:
        target=int(texture['id']);filename=Path(texture['file']).resolve()
        if not 0<=target<len(model['tex_type']) or not filename.is_file():
            raise ValueError('Invalid runtime texture')
        for geom,material in enumerate(model['geom_matid']):
            if material>=0 and model['mat_texid'][material,1]==target:
                shader=UsdShade.Shader.Get(stage,f'/World/Materials/g{geom}/Texture')
                if shader:shader.GetInput('file').Set(str(filename))
    geometry=visuals.get('geometry',[])
    for i,geom in enumerate(geometry):
        kind=geom['type']
        if kind not in (4,5):raise NotImplementedError('Unsupported runtime visual geometry')
        if i not in geometry_cache:
            path=f'/World/RuntimeVisuals/geometry{i}'
            if kind==5:
                shape=UsdGeom.Cylinder.Define(stage,path)
                shape.CreateAxisAttr('Z')
            elif geom.get('role') in ('cell_shell','cell_nucleus'):
                shape=cell_mesh(stage,path)
            else:
                shape=UsdGeom.Sphere.Define(stage,path)
            transform=UsdGeom.Xformable(shape.GetPrim()).AddTransformOp()
            material=UsdShade.Material.Define(stage,f'/World/RuntimeVisuals/material{i}')
            shader=UsdShade.Shader.Define(stage,f'/World/RuntimeVisuals/material{i}/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f)
            shader.CreateInput('opacity',Sdf.ValueTypeNames.Float)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
            geometry_cache[i]=(shape,transform,shader)
        shape,transform,shader=geometry_cache[i]
        matrix=np.eye(4);matrix[:3,:3]=np.asarray(geom['mat']).reshape(3,3).T;matrix[3,:3]=geom['pos']
        if kind==5:
            if not isinstance(shape,UsdGeom.Cylinder):raise ValueError('Runtime geometry type changed')
            shape.CreateRadiusAttr(float(geom['size'][0]));shape.CreateHeightAttr(2*float(geom['size'][1]))
        else:
            if not isinstance(shape,(UsdGeom.Sphere,UsdGeom.Mesh)):raise ValueError('Runtime geometry type changed')
            if isinstance(shape,UsdGeom.Sphere):shape.CreateRadiusAttr(1.)
            matrix[:3,:3]=np.diag(geom['size'])@matrix[:3,:3]
        if not np.isfinite(matrix).all():raise ValueError('Non-finite runtime visual transform')
        transform.Set(Gf.Matrix4d(*matrix.ravel().tolist()))
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*map(float,geom['rgba'][:3])))
        shader.GetInput('opacity').Set(float(geom['rgba'][3]))
        surface = geom.get('surface')
        if surface is not None:
            apply_surface(shader,surface,geom['rgba'])
        imageable=UsdGeom.Imageable(shape.GetPrim())
        if float(geom['rgba'][3]) == 0:imageable.MakeInvisible()
        else:imageable.MakeVisible()
    for i,(shape,_,_) in geometry_cache.items():
        if i>=len(geometry):UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
    return {'texture_updates':len(textures),'liquid_surfaces':sum(geom['type']==5 and geom.get('role')!='pneumatic_hose' for geom in geometry),
            'runtime_geometries':len(geometry)}
