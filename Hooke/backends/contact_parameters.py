"""Experimental constant-impedance approximation for native compliant contacts.

The mapping matches isolated static penetration at a chosen source impedance.
It does not reproduce varying impedance, the coupled solver, or friction cone.
The runtime's calibrated default uses source impedance at small residuals.
"""
import numpy as np


def constraint_parameters(reference,impedance,dt,impedance_fraction=1.):
    if np.any(np.asarray(reference)<=0):raise NotImplementedError('Direct-format constraint reference is not calibrated')
    dmax=float(np.clip(impedance[1],.0001,.9999))
    d=float(np.clip(impedance[0]+impedance_fraction*(impedance[1]-impedance[0]),.0001,.9999))
    time_constant=max(float(reference[0]),2*dt);ratio=float(reference[1])
    stiffness=d**2/((1-d)*dmax**2*time_constant**2*ratio**2)
    damping=2*d/((1-d)*dmax*time_constant)
    return stiffness,damping


def compliant_parameters(model,geom,dt,impedance_fraction=1.):
    reference=np.asarray(model['geom_solref'][geom],dtype=float)
    impedance=np.asarray(model['geom_solimp'][geom],dtype=float)
    # Dedicated zero-mask colliders participate only in explicit pairs. When
    # all such pairs agree, their contact parameters can live on that material.
    if not (model['geom_contype'][geom] or model['geom_conaffinity'][geom]):
        indices=np.flatnonzero((model['pair_geom1']==geom)|(model['pair_geom2']==geom))
        if len(indices):
            refs=model['pair_solref'][indices];imps=model['pair_solimp'][indices]
            if not (np.allclose(refs,refs[0]) and np.allclose(imps,imps[0])):
                raise NotImplementedError('Different explicit-pair compliances require pair-specific materials')
            reference=refs[0];impedance=imps[0]
    # At rest: (1-d)*a_free - d*k*r = 0; native acceleration spring
    # gives a_free - stiffness*r = 0. This equates their static depths.
    return constraint_parameters(reference,impedance,dt,impedance_fraction)
