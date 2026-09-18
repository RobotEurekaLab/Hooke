"""Coordinate transforms for independently mounted electric tool axes."""

import numpy as np
import mujoco


def tool_axes(model, tool):
    """World directions of the tool's three slides (unrotated MJCF bodies)."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tool + "_platform_free") >= 0:
        return np.eye(3)  # Logical parallel controls use world Cartesian displacement.
    return np.column_stack([model.joint(f"{tool}_{axis}").axis for axis in "xyz"])


def tool_commands(model, tool, displacement):
    values = np.linalg.solve(tool_axes(model, tool), displacement)
    return {f"{tool}_{axis}": float(value) for axis, value in zip("xyz", values)}


def site_dofs(model, site):
    """Generalized coordinates influenced by a site; includes a free platform's six DOFs."""
    node, indices = int(model.site_bodyid[site]), []
    while node:
        start, count = int(model.body_dofadr[node]), int(model.body_dofnum[node])
        indices.extend(range(start, start + count))
        node = int(model.body_parentid[node])
    return sorted(indices)


def tool_dofs(model, tool):
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tool + "_x") >= 0:
        return [int(model.joint(f"{tool}_{axis}").dofadr[0]) for axis in "xyz"]
    return site_dofs(model, model.site(tool + "_tcp").id)


def site_force(model, data, site, force):
    """Map a world force at the actual tip to translation and torque with its Jacobian."""
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian, None, site)
    return jacobian.T @ force
