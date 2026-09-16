"""Read contact membership from the active simulation state."""
import numpy as np

def body_geoms(model, root):
    bodies = {int(root)}
    for i in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    return {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies}


def touching(data, first, second):
    pairs = getattr(data.contact, 'geom', None)
    if pairs is not None:
        return bool(np.any(
            (np.isin(pairs[:,0], tuple(first)) & np.isin(pairs[:,1], tuple(second)))
            | (np.isin(pairs[:,1], tuple(first)) & np.isin(pairs[:,0], tuple(second)))))
    return any((int(c.geom[0]) in first and int(c.geom[1]) in second)
               or (int(c.geom[1]) in first and int(c.geom[0]) in second)
               for c in data.contact)
