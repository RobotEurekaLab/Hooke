"""Read contact membership from the active simulation state."""

def body_geoms(model, root):
    bodies = {int(root)}
    for i in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    return {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies}


def touching(data, first, second):
    return any((int(c.geom[0]) in first and int(c.geom[1]) in second)
               or (int(c.geom[1]) in first and int(c.geom[0]) in second)
               for c in data.contact)

