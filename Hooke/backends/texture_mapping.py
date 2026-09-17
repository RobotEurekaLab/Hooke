"""Source OpenGL texture coordinates expressed for USD image sampling."""
import numpy as np


def source_uv_to_usd(coordinates):
    # Source tex_data's first row is uploaded at OpenGL v=0. PNG/USD places
    # that row at v=1. Keep the original pixels and invert the sampling axis.
    uv=np.asarray(coordinates,dtype=np.float32).copy()
    uv[...,1]=1-uv[...,1]
    return uv


def plane_uv(points, size, repeat, uniform):
    # MuJoCo 3.3 render_gl3.c: source S=.5*sx*x-.5,
    # T=-.5*sy*y-.5. Adding one is equivalent for repeat wrapping.
    scale=np.asarray(repeat,dtype=float).copy()
    if not uniform:scale/=np.asarray(size[:2])
    uv=np.asarray(points)[:,:2]*scale*.5
    uv[:,1]*=-1
    return source_uv_to_usd(uv+.5)
