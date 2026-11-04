"""Photo-derived covers and services preserve supported motion apertures."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


def context_bounds(m):
    """Relative affine XYZ bounds plus independent finite-angle body bounds.

    The free retained vial uses its initial position with a 2 mm margin and
    spheres bounding every orientation. This does not cover unbounded travel.
    """
    d = mujoco.MjData(m)
    ids = [
        m.joint("uw_cartesian_v3__joint_" + k).id
        for k in ("gantry_x", "gantry_y", "injection_z")
    ]
    new = np.array(
        [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("uw_source_")]
    )
    all_ids = np.arange(m.ngeom)
    weld = m.body_weldid[m.geom_bodyid]
    pairmask = weld[new, None] != weld[None, :]
    # Include visible shapes as well as enabled surfaces; weld-equal pieces
    # may intentionally connect as one rigid supported solid assembly.
    centres, halves = [], []
    for pose in itertools.product(*(m.jnt_range[j] for j in ids)):
        d.qpos[:] = m.qpos0
        for j, value in zip(ids, pose):
            d.qpos[m.jnt_qposadr[j]] = value
        mujoco.mj_forward(m, d)
        r = d.geom_xmat.reshape(-1, 3, 3)
        centres.append(d.geom_xpos + np.einsum("nij,nj->ni", r, m.geom_aabb[:, :3]))
        halves.append(np.einsum("nij,nj->ni", abs(r), m.geom_aabb[:, 3:]))
    centres, halves = np.array(centres), np.array(halves)
    low = np.full((len(new), m.ngeom, 3), np.inf)
    high = np.full_like(low, -np.inf)
    for centre, half in zip(centres, halves):
        relative = centre[None, :, :] - centre[new, None, :]
        halfsum = half[None, :, :] + half[new, None, :]
        low = np.minimum(low, relative - halfsum)
        high = np.maximum(high, relative + halfsum)
    # For other devices, use independent world envelopes to cover combinations
    # with all XYZ poses. Hinge samples get r*half-angle conservative padding.
    primary = set(ids)
    for j in range(m.njnt):
        if j in primary:
            continue
        body = m.jnt_bodyid[j]
        affected = np.flatnonzero(m.body_weldid[m.geom_bodyid] == m.body_weldid[body])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qa = m.jnt_qposadr[j]
            centre = m.qpos0[qa : qa + 3]
            radius = (
                np.linalg.norm(m.geom_pos[affected], axis=1)
                + np.linalg.norm(m.geom_aabb[affected, :3], axis=1)
                + np.linalg.norm(m.geom_aabb[affected, 3:], axis=1)
                + 0.002
            )
            omin = centre - radius[:, None]
            omax = centre + radius[:, None]
        else:
            lo, hi = m.jnt_range[j]
            count = max(2, int(np.ceil((hi - lo) / 0.05)) + 1)
            obounds = []
            for value in np.linspace(lo, hi, count):
                d.qpos[:] = m.qpos0
                d.qpos[m.jnt_qposadr[j]] = value
                mujoco.mj_forward(m, d)
                r = d.geom_xmat.reshape(-1, 3, 3)
                c = d.geom_xpos + np.einsum("nij,nj->ni", r, m.geom_aabb[:, :3])
                h = np.einsum("nij,nj->ni", abs(r), m.geom_aabb[:, 3:])
                obounds.append((c[affected] - h[affected], c[affected] + h[affected]))
            omin = np.min([a for a, b in obounds], axis=0)
            omax = np.max([b for a, b in obounds], axis=0)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                radius = np.linalg.norm(
                    np.maximum(abs(omin - d.xanchor[j]), abs(omax - d.xanchor[j])),
                    axis=1,
                )
                pad = radius[:, None] * (hi - lo) / (count - 1) / 2
                omin -= pad
                omax += pad
        nmin = np.min(centres[:, new] - halves[:, new], axis=0)
        nmax = np.max(centres[:, new] + halves[:, new], axis=0)
        low[:, affected] = omin[None, :] - nmax[:, None]
        high[:, affected] = omax[None, :] - nmin[:, None]
    gaps = np.linalg.norm(np.maximum(np.maximum(low, -high), 0), axis=2)
    gaps[~pairmask] = np.inf
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    return {
        "minimum_clearance_lower_bound_m": float(gaps[i, j]),
        "witness": [m.geom(new[i]).name, m.geom(all_ids[j]).name],
        "affected_cross_weld_pairs": int(pairmask.sum()),
        "new_geom_count": len(new),
        "moving_new_geom_count": int(sum(weld[new] != 0)),
    }


class UWSourceServicesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, _ = build_scene("uw_isolab")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def test_all_added_solids_have_support_on_their_own_rigid_frame(self):
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        new = {
            g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("uw_source_")
        }
        self.assertTrue(new)
        weld = m.body_weldid[m.geom_bodyid]
        reached = {g for g in range(m.ngeom) if g not in new and m.geom_contype[g]}
        pending = set(new)
        for g in new:
            self.assertTrue(m.geom_contype[g] and m.geom_conaffinity[g])
        while pending:
            found = {
                g
                for g in pending
                if any(
                    weld[g] == weld[a]
                    and mujoco.mj_geomDistance(m, d, g, a, 1e-7, None) <= 1e-8
                    for a in reached
                )
            }
            if not found:
                break
            pending -= found
            reached |= found
        self.assertFalse(pending, [m.geom(g).name for g in pending])

    def test_new_surfaces_clear_all_existing_motion_and_open_tool_outlet(self):
        report = context_bounds(self.model)
        self.assertGreater(report["minimum_clearance_lower_bound_m"], 0.0015, report)
        self.assertEqual(report["moving_new_geom_count"], 7)


if __name__ == "__main__":
    unittest.main()
