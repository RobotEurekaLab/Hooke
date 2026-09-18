"""Reuse the source runtime display and liquid drawing in either renderer.

These are visual extras only. They never create colliders or alter task checks.
"""
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def serialize_liquid_geom(geom):
    if geom.type!=mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise NotImplementedError('Unsupported source runtime visual geometry')
    # mjv_initGeom expands the model's [radius, halfheight, 0] into the
    # renderer's [radius, radius, halfheight]. Send canonical model sizes.
    return {'type':int(geom.type),'size':[float(geom.size[0]),float(geom.size[2]),0.],
            'pos':geom.pos.tolist(),'mat':geom.mat.tolist(),'rgba':geom.rgba.tolist()}


def ellipse_frame(points):
    """Real conic fit for nearly circular boundaries that upset legacy fitting.

    Fit x.T Q x + linear.T x = 1 in normalized coordinates, then obtain
    the center and principal radii from the positive quadratic form.
    """
    points=np.asarray(points,dtype=float);origin=points.mean(axis=0)
    scale=float(np.max(np.linalg.norm(points-origin,axis=1)))
    if len(points)<5 or not np.isfinite(scale) or scale<1e-12:
        raise ValueError('Liquid boundary does not define a finite ellipse')
    x,y=((points-origin)/scale).T
    design=np.column_stack((x*x,x*y,y*y,x,y))
    coefficients,_,rank,_=np.linalg.lstsq(design,np.ones(len(points)),rcond=None)
    if rank<5:raise ValueError('Degenerate liquid boundary')
    a,b,c,d,e=coefficients;quadratic=np.array([[a,b/2],[b/2,c]])
    center=-.5*np.linalg.solve(quadratic,[d,e])
    normalized=quadratic/(1+center@quadratic@center)
    values,axes=np.linalg.eigh(normalized)
    if np.any(values<=0):raise ValueError('Liquid boundary fit is not an ellipse')
    if np.linalg.det(axes)<0:axes[:,1]*=-1
    return origin+scale*center,axes*(scale/np.sqrt(values))


class LiveVisuals:
    def __init__(self, task, output):
        from instrument import Thermal_mixer_eppendorf_c
        from evaluator import make_liquid_extra
        from liquid import ContainerSystem

        self.task=task;self.output=Path(output)
        self.displays=[];self.images={};self.updates=0
        for mixer in task.manager.systems_by_type.get(Thermal_mixer_eppendorf_c,[]):
            fig,ax=mixer.ui_state.make_canvas()
            self.displays.append({'mixer':mixer,'fig':fig,'ax':ax,'last':None,'revision':0})
        self.liquid_extra=make_liquid_extra(task) if task.manager.systems_by_type.get(ContainerSystem,[]) else None
        self.containers=task.manager.systems_by_type.get(ContainerSystem,[])
        self.liquid_fit_fallbacks=0
        self.extra_scene=mujoco.MjvScene(task.model,maxgeom=1024) if self.liquid_extra else None

    def snapshot(self):
        textures=[]
        for display in self.displays:
            mixer=display['mixer'];ui=mixer.ui_state
            if display['last']==ui:continue
            ui.draw(display['ax']);pixels=ui.render_canvas(display['fig'])
            target=int(mixer.display)
            expected=self.task.model.tex(target).data.shape
            if pixels.shape!=expected:raise ValueError(f'Display texture {target}: {pixels.shape} != {expected}')
            self.output.mkdir(parents=True,exist_ok=True)
            filename=self.output/f't{target}-{display["revision"]:05d}.png'
            Image.fromarray(pixels).save(filename)
            textures.append({'id':target,'file':str(filename.resolve())})
            self.images[target]=pixels
            display['last']=deepcopy(ui);display['revision']+=1;self.updates+=1
        geometry=[]
        if self.liquid_extra:
            self.extra_scene.ngeom=0
            try:self.liquid_extra[0](self.extra_scene,None)
            except (TypeError,ValueError,np.linalg.LinAlgError):
                # Legacy EllipseModel can produce an exactly-real complex
                # eigenvector and then fail at angle modulo. Keep drawing the
                # source liquid boundary without changing its physical state.
                self.extra_scene.ngeom=0;self.liquid_fit_fallbacks+=1
                for system in self.containers:
                    container=system.container;liquid=container.liquid
                    if liquid is None or not liquid.surface.valid:continue
                    mesh=liquid.meshplane.calculate_mesh(liquid.surface.distance)
                    plane=liquid.surface.frame
                    points=np.asarray(mesh.vertices)[mesh.boundary]@plane
                    center,axes=ellipse_frame(points[:,:2])
                    local=np.eye(3);local[:2,:2]=axes
                    rotation=np.asarray(container.rotation_matrix)@plane
                    position=np.asarray(container.position)+rotation@np.r_[center,liquid.surface.distance]
                    mujoco.mjv_initGeom(self.extra_scene.geoms[self.extra_scene.ngeom],mujoco.mjtGeom.mjGEOM_CYLINDER,
                                       np.array([1.,1e-4,0.]),position,(rotation@local).ravel(),np.array([0.,0.,1.,1.],dtype=np.float32))
                    self.extra_scene.ngeom+=1
            for i in range(self.extra_scene.ngeom):
                geom=self.extra_scene.geoms[i]
                geometry.append(serialize_liquid_geom(geom))
        if callable(getattr(self.task, 'runtime_visuals', None)):
            geometry.extend(self.task.runtime_visuals())
        return {'textures':textures,'geometry':geometry}

    def apply_mujoco(self, renderer, state):
        for texture in state['textures']:
            target=texture['id'];self.task.model.tex(target).data[:]=self.images[target]
            mujoco.mjr_uploadTexture(self.task.model,renderer._mjr_context,target)
        scene=renderer.scene
        for geom in state['geometry']:
            if scene.ngeom>=scene.maxgeom:raise RuntimeError('Runtime visual geometry exceeds renderer capacity')
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],geom['type'],np.asarray(geom['size']),
                               np.asarray(geom['pos']),np.asarray(geom['mat']).ravel(),np.asarray(geom['rgba'],dtype=np.float32))
            if 'surface' in geom:
                scene.geoms[scene.ngeom].specular=float(np.mean(geom['surface']['specular_color']))
                scene.geoms[scene.ngeom].shininess=1-float(geom['surface']['roughness'])
            scene.ngeom+=1

    def close(self):
        if self.displays:
            import matplotlib.pyplot as plt
            for display in self.displays:plt.close(display['fig'])
        if self.liquid_extra:self.liquid_extra[1]()
