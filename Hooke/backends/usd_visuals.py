"""USD display textures and liquid surfaces; no physics or collision APIs."""
from pathlib import Path
import numpy as np
from pxr import UsdGeom,UsdShade,Sdf,Gf


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
        if geom['type']!=5:raise NotImplementedError('Runtime visual extra must be a source liquid cylinder')
        if i not in geometry_cache:
            shape=UsdGeom.Cylinder.Define(stage,f'/World/RuntimeVisuals/liquid{i}')
            shape.CreateAxisAttr('Z')
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
        shape.CreateRadiusAttr(float(geom['size'][0]));shape.CreateHeightAttr(2*float(geom['size'][1]))
        matrix=np.eye(4);matrix[:3,:3]=np.asarray(geom['mat']).reshape(3,3).T;matrix[3,:3]=geom['pos']
        if not np.isfinite(matrix).all():raise ValueError('Non-finite liquid surface transform')
        transform.Set(Gf.Matrix4d(*matrix.ravel().tolist()))
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*map(float,geom['rgba'][:3])))
        shader.GetInput('opacity').Set(float(geom['rgba'][3]))
        UsdGeom.Imageable(shape.GetPrim()).MakeVisible()
    for i,(shape,_,_) in geometry_cache.items():
        if i>=len(geometry):UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
    return {'texture_updates':len(textures),'liquid_surfaces':len(geometry)}
