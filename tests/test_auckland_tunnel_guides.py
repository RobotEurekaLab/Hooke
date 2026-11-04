"""Actual open-guide support, preserved interfaces and loaded dry positioning."""
import unittest
import xml.etree.ElementTree as ET
from collections import Counter

import mujoco
import numpy as np

from real_labs.auckland_tunnel_geometry import PREFIX
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.layout_audit import audit_layout

SCENE = "auckland_aerodynamics_tunnel"
EQUIPMENT = "primary_apparatus"


def session():
    root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def enabled_pairs(model):
    ids = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    groups = model.body_weldid[model.geom_bodyid]
    # All enabled pairs crossing either actuated weld group, including direct
    # parent/child pairs that native contact generation normally filters out.
    return np.array([(a, b) for i, a in enumerate(ids) for b in ids[i+1:]
                     if groups[a] != groups[b] and (groups[a] or groups[b])], dtype=int)


def minimum_distance(model, data, pairs):
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
    a, b = pairs.T
    lower = np.linalg.norm(np.maximum(abs(center[a]-center[b])-half[a]-half[b], 0), axis=1)
    checked = pairs[lower < .002 + 1e-10]
    return min([.002]+[float(mujoco.mj_geomDistance(model, data, int(a), int(b), .002, None))
                       for a, b in checked]), len(checked)


def distance(model, data, first, second):
    return float(mujoco.mj_geomDistance(model, data, model.geom(first).id,
                                      model.geom(second).id, 1, None))


def run_loaded(session):
    """Actual actuator route; all physics steps checked and no state reset."""
    s = session
    m, d = s.model, s.data
    pairs = enabled_pairs(m)
    cx = s.controls[EQUIPMENT, "specimen_x"]
    cy = s.controls[EQUIPMENT, "specimen_yaw"]
    samples, minimum, queries, legs = 0, .002, 0, []
    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
    def actual_state():
        state = np.empty(mujoco.mj_stateSize(m, state_kind))
        mujoco.mj_getState(m, d, state, state_kind)
        return {"state_kind": int(state_kind), "integration_state": state.tolist(),
                "qacc_warmstart": d.qacc_warmstart.tolist(),
                "qfrc_applied": d.qfrc_applied.tolist()}
    for load in (-.8, .8):
        d.qfrc_applied[m.jnt_dofadr[cx.joint]] = load
        d.qfrc_applied[m.jnt_dofadr[cy.joint]] = .04 * np.sign(load)
        for fraction in (.15, .85, .25):
            for c in (cx, cy):
                d.ctrl[c.actuator] = c.lower + fraction*(c.upper-c.lower)
            for _ in range(round(2/m.opt.timestep)):
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                gap, count = minimum_distance(m, d, pairs)
                if gap < .0005998:
                    raise AssertionError(("direct clearance", gap, d.qpos.copy()))
                if any(c.dist < -1e-7 for c in d.contact):
                    raise AssertionError("native penetration")
                minimum, queries, samples = min(minimum, gap), queries+count, samples+1
            errors = [abs(s._measured(c)-d.ctrl[c.actuator]) for c in (cx, cy)]
            if any(e > .025*(c.upper-c.lower) for e, c in zip(errors, (cx, cy))):
                raise AssertionError(("tracking", errors))
            legs.append(dict(load_N=load, torque_Nm=.04*np.sign(load), fraction=fraction,
                             qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(),
                             time_s=float(d.time), errors=errors, actual_state=actual_state()))
    d.qfrc_applied[:] = 0
    d.ctrl[:] = 0
    for _ in range(round(2/m.opt.timestep)):
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        gap, count = minimum_distance(m, d, pairs)
        if gap < .0005998:
            raise AssertionError(("return clearance", gap))
        minimum, queries, samples = min(minimum, gap), queries+count, samples+1
    return dict(passed=True, checked_steps=samples, executed_steps=samples,
                direct_queries=queries, pair_count=len(pairs),
                clearance_lower_bound_m=minimum, bound_cap_m=.002,
                legs=legs, final=dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(),
                                     ctrl=d.ctrl.tolist(), time_s=float(d.time),
                                     actual_state=actual_state()),
                no_reset=True, external_load_scope="Authored small generalized axial force/torque; ideal guides retain prescribed constraints, not physical bearing load calibration")


class AucklandGuideTests(unittest.TestCase):
    def test_original_control_scale_and_site_contract(self):
        s = session();m=s.model
        for key, pos, axis, limits, mass, gain, force in (
            ("specimen_x", (0,0,.53), (1,0,0), (-.22,.22), 4,1200,1600),
            ("specimen_yaw", (0,0,.075), (0,0,1), (-.3,.3), 2,400,500),
        ):
            c=s.controls[EQUIPMENT,key];body=m.jnt_bodyid[c.joint]
            np.testing.assert_allclose(m.body_pos[body],pos)
            np.testing.assert_allclose(m.jnt_axis[c.joint],axis)
            np.testing.assert_allclose(m.jnt_range[c.joint],limits)
            self.assertEqual(m.body_mass[body],mass)
            np.testing.assert_allclose(m.body_inertia[body],[mass*.006]*3)
            self.assertEqual(m.actuator_gainprm[c.actuator,0],gain)
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator],[-force,force])
        np.testing.assert_allclose(m.geom("primary_apparatus__clear_foil_30").size,[.045,.28,.74])
        np.testing.assert_allclose(m.site("primary_apparatus__site_clear_specimen").pos,[0,0,.92])
        self.assertEqual(m.ncam,3)
        self.assertTrue(audit_layout(s.manifest)["passed"])

    def test_connected_mounts_and_open_watertight_sleeves(self):
        s=session();m,d=s.model,s.data
        for i in range(2):
            for j in range(2):
                for a,b in ((f"foot_{i}_{j}","base"),
                            (f"rail_mount_{i}_{j}","base"),
                            (f"rail_mount_{i}_{j}",f"rail_{i}")):
                    self.assertLessEqual(distance(m,d,PREFIX+a,PREFIX+b),1e-8)
                self.assertLessEqual(distance(m,d,PREFIX+f"foot_{i}_{j}","primary_apparatus__test_floor_2"),1e-8)
                self.assertLessEqual(distance(m,d,PREFIX+f"bearing_crown_{i}_{j}",PREFIX+"carriage_plate"),1e-8)
                contacts=[m.geom(PREFIX+f"bearing_{i}_{j}_contact_{k}").id for k in range(48)]
                crown=m.geom(PREFIX+f"bearing_crown_{i}_{j}").id
                self.assertLessEqual(min(mujoco.mj_geomDistance(m,d,crown,g,1,None) for g in contacts),1e-8)
        for a,b in (("carriage_plate","fixed_rotary_shaft"),("carriage_plate","thrust_support"),
                    ("upper_foil_clamp","upper_plate")):
            self.assertLessEqual(distance(m,d,PREFIX+a,PREFIX+b),1e-8)
        self.assertLessEqual(distance(m,d,PREFIX+"upper_foil_clamp","primary_apparatus__clear_foil_30"),1e-8)
        # Mesh topology is inspected in author coordinates, independently of
        # the convex collision pieces; a watertight torus still has a real bore.
        root,_=build_scene(SCENE)
        for mesh in root.findall('./asset/mesh'):
            if not mesh.get('name','').startswith(PREFIX):continue
            v=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3)
            f=np.fromstring(mesh.get('face'),sep=' ',dtype=int).reshape(-1,3)
            edges=Counter(tuple(sorted(e)) for face in f for e in zip(face,np.roll(face,-1)))
            self.assertEqual(set(edges.values()),{2})
            volume=np.sum(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])))/6
            self.assertGreater(volume,0)

    def test_full_coupled_travel_parent_pairs_and_running_gap(self):
        s=session();m,d=s.model,s.data;pairs=enabled_pairs(m)
        for x in np.linspace(-.22,.22,13):
            for angle in np.linspace(-.3,.3,11):
                d.qpos[:]=[x,angle];mujoco.mj_forward(m,d)
                gap,_=minimum_distance(m,d,pairs)
                self.assertGreaterEqual(gap,.0005998,(x,angle,gap))
                for rail in range(2):
                    for bearing in range(2):
                        # Full 120 mm sleeves remain between fixed end mounts.
                        c=(-.30,.30)[bearing]+x
                        self.assertGreater(c-.06,-.70+.026)
                        self.assertLess(c+.06,.70-.026)
        # An enabled rail shifted into the open bearing wall must be detected;
        # this guards against an empty pair set or masks hiding the interface.
        g=m.geom(PREFIX+'rail_0').id;m.geom_pos[g,2]+=.002
        mujoco.mj_forward(m,d)
        self.assertLess(minimum_distance(m,d,pairs)[0],0)

    def test_loaded_bidirectional_route_and_actual_return(self):
        s=session();report=run_loaded(s)
        self.assertEqual(report['checked_steps'],round(14/s.model.opt.timestep))
        self.assertLess(np.max(abs(s.data.qpos)),1e-6)
        self.assertAlmostEqual(s.data.time,14,places=8)


if __name__ == '__main__':
    unittest.main()
