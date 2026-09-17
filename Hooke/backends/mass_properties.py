"""Exact mass/inertia composition for bodies welded to a moving joint owner."""
import numpy as np


def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def quaternion(matrix):
    """Rotation matrix to a unit wxyz quaternion, including half turns."""
    r=np.asarray(matrix,dtype=float);trace=np.trace(r)
    if trace>0:
        s=np.sqrt(trace+1.)*2
        q=np.array([s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s])
    else:
        i=int(np.argmax(np.diag(r)));j=(i+1)%3;k=(i+2)%3
        s=np.sqrt(max(0.,1+r[i,i]-r[j,j]-r[k,k]))*2
        q=np.zeros(4);q[0]=(r[k,j]-r[j,k])/s;q[i+1]=s/4
        q[j+1]=(r[j,i]+r[i,j])/s;q[k+1]=(r[k,i]+r[i,k])/s
    return q/np.linalg.norm(q)


def compose_welded_mass(m,root,members):
    """Return mass properties in root coordinates, without adding mass."""
    rr=rotation(m['reference_xquat'][root]);origin=m['reference_xpos'][root]
    parts=[]
    for body in members:
        mass=float(m['body_mass'][body])
        if mass<=0:continue
        rb=rotation(m['reference_xquat'][body])
        center=rr.T@(m['reference_xpos'][body]+rb@m['body_ipos'][body]-origin)
        axes=rr.T@rb@rotation(m['body_iquat'][body])
        tensor=axes@np.diag(m['body_inertia'][body])@axes.T
        parts.append((mass,center,tensor,float(m['body_gravcomp'][body])))
    mass=sum(p[0] for p in parts)
    if mass<=0:raise ValueError(f'Moving welded group {root} has no physical mass')
    center=sum(weight*position for weight,position,_,_ in parts)/mass
    tensor=np.zeros((3,3))
    for weight,position,inertia,_ in parts:
        delta=position-center
        tensor+=inertia+weight*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    diagonal,axes=np.linalg.eigh(tensor)
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    if np.any(diagonal<=0):raise ValueError(f'Invalid welded inertia for body {root}')
    return {'mass':mass,'center':center,'diagonal':diagonal,'axes':quaternion(axes),
            'tensor':tensor,'gravcomp':sum(p[0]*p[3] for p in parts)/mass,'members':list(members)}
