"""Explicit touching support interfaces for newly authored dry inspections.

Listed pairs remain in the direct geometry observer. They may meet within
0.1 micrometre numerical tolerance; every other pair retains positive clearance.
The older RigidAccess operation contracts are unchanged.
"""

import json
from pathlib import Path

import numpy as np

from real_labs.rigid_access import RigidAccess, near_geometry_pairs
from real_labs.motion import JointMotion
from real_labs.inspection_distance import geometry_separation


CONTACT_NUMERICAL_TOLERANCE_M = 1e-7


def _near_pairs(model, data, pairs, cap):
    if not np.isfinite(data.geom_xpos).all() or not np.isfinite(data.geom_xmat).all():
        raise RuntimeError("inspection_nonfinite_geometry_state")
    return near_geometry_pairs(model, data, pairs, cap)


class InspectionJointMotion(JointMotion):
    """Plan with the same finite interface contract as the live observer."""

    def __init__(self, session, equipment, controls, *, clearance_m, pairs, interfaces):
        self.direct_pairs = pairs.copy()
        self.support_pairs = frozenset(interfaces)
        self.geometry_query_discrepancies = []
        super().__init__(session, equipment, controls, clearance_m=clearance_m)

    def configuration_contacts(self, values):
        # Keep the ordinary enabled-contact rules for all unregistered pairs.
        ordinary = super().configuration_contacts(values)
        violations = {}
        for contact in ordinary:
            ids = tuple(sorted(self.model.geom(name).id for name in contact["geoms"]))
            if (ids not in self.support_pairs
                    or contact["distance_m"] < -CONTACT_NUMERICAL_TOLERANCE_M):
                violations[ids] = contact
        cap = max(.002, self.clearance)
        for first, second in _near_pairs(self.model, self.data, self.direct_pairs, cap):
            first, second = int(first), int(second)
            key = tuple(sorted((first, second)))
            distance, raw, method = geometry_separation(self.model, self.data, first, second, cap)
            bound = -CONTACT_NUMERICAL_TOLERANCE_M if key in self.support_pairs else self.clearance
            if (raw < -CONTACT_NUMERICAL_TOLERANCE_M) != (distance < -CONTACT_NUMERICAL_TOLERANCE_M):
                self.geometry_query_discrepancies.append(dict(
                    geoms=[self.model.geom(first).name, self.model.geom(second).name],
                    configuration=self.data.qpos[self.qadr].tolist(), raw_distance_m=raw,
                    conservative_distance_m=distance, method=method,
                ))
            if distance < bound:
                violations[key] = dict(
                    geoms=[self.model.geom(first).name, self.model.geom(second).name],
                    distance_m=distance, raw_distance_m=raw, method=method,
                )
        return list(violations.values())


def registered_interfaces(scene, equipment):
    """Return authored named interfaces, never allowances inferred from a run."""
    path = Path(__file__).with_name("inspection_interfaces.json")
    record = json.loads(path.read_text())["scenes"].get(scene)
    if record is None or equipment not in record["equipment"]:
        return {"pairs": []}
    return {"scene_sha256": record["scene_sha256"], **record["equipment"][equipment]}


class InspectionAccess(RigidAccess):
    """A source-bound rigid observer with a finite named support contract."""

    def __init__(self, session, scene, equipment, controls, *, minimum_m=0.00025,
                 interfaces=None):
        super().__init__(session, scene, equipment, controls, minimum_m=minimum_m)
        record = registered_interfaces(scene, equipment) if interfaces is None else interfaces
        if not isinstance(record, dict) or not isinstance(record.get("pairs"), list):
            raise ValueError("Inspection interfaces require an explicit pair record")
        if record["pairs"] and record.get("scene_sha256") != self.binding["scene_sha256"]:
            raise ValueError("Registered inspection interface geometry changed")
        checked_pairs = {tuple(sorted(map(int, pair))) for pair in self.pairs}
        self.interfaces = {}
        for row in record["pairs"]:
            names = row.get("geoms", [])
            if len(names) != 2 or len(set(names)) != 2:
                raise ValueError("Inspection interfaces require two distinct geometry names")
            ids = tuple(int(self.model.geom(name).id) for name in names)
            key = tuple(sorted(ids))
            if list(ids) != row.get("geom_ids") or key not in checked_pairs:
                raise ValueError("Registered interface is not an observed moving geometry pair")
            if key in self.interfaces:
                raise ValueError("Duplicate inspection support interface")
            self.interfaces[key] = tuple(names)
        self.interface_rationale = record.get("rationale")
        if self.interfaces and not self.interface_rationale:
            raise ValueError("Registered support interfaces need a mechanical rationale")
        self.motion = InspectionJointMotion(
            session, equipment, controls, clearance_m=minimum_m,
            pairs=self.pairs, interfaces=self.interfaces,
        )
        self.reset_counts()

    def reset_counts(self):
        super().reset_counts()
        self.geometry_query_discrepancies = []
        self.interface_observations = {
            pair: {"geoms": list(names), "checked_observations": 0,
                   "touching_observations": 0, "minimum_distance_m": None}
            for pair, names in getattr(self, "interfaces", {}).items()
        }

    def check(self, model, data):
        # Preserve RigidAccess ownership/contact invariants and AABB culling.
        # The only changed rule is the bound for the finite registered pairs.
        self.entries += 1
        if (self.session.model is not self.model or self.session.data is not self.data
                or model is not self.model or data is not self.data or data.model is not model):
            raise RuntimeError("inspection_model_state_binding_changed")
        if ((int(model.opt.disableflags), int(model.opt.enableflags)) != self.flags
                or not np.array_equal(model.geom_contype, self.masks[0])
                or not np.array_equal(model.geom_conaffinity, self.masks[1])):
            raise RuntimeError("inspection_contact_configuration_changed")
        current = self.cap_m
        for first, second in _near_pairs(model, data, self.pairs, self.cap_m):
            first, second = int(first), int(second)
            gap, raw, method = geometry_separation(model, data, first, second, self.cap_m)
            if (raw < -CONTACT_NUMERICAL_TOLERANCE_M) != (gap < -CONTACT_NUMERICAL_TOLERANCE_M):
                self.geometry_query_discrepancies.append(dict(
                    time_s=float(data.time), checked_entry=self.entries,
                    geoms=[model.geom(first).name, model.geom(second).name],
                    raw_distance_m=raw, conservative_distance_m=gap, method=method,
                ))
            self.queries += 1
            current = min(current, gap)
            key = tuple(sorted((first, second)))
            registered = key in self.interfaces
            bound = -CONTACT_NUMERICAL_TOLERANCE_M if registered else self.minimum_m
            if not np.isfinite(gap) or gap < bound:
                raise RuntimeError("inspection_clearance_lost:" + model.geom(first).name
                                   + ":" + model.geom(second).name)
            if registered:
                observation = self.interface_observations[key]
                observation["checked_observations"] += 1
                observation["touching_observations"] += int(abs(gap) <= CONTACT_NUMERICAL_TOLERANCE_M)
                previous = observation["minimum_distance_m"]
                observation["minimum_distance_m"] = gap if previous is None else min(previous, gap)
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def interface_report(self):
        return dict(
            numerical_contact_tolerance_m=CONTACT_NUMERICAL_TOLERANCE_M,
            other_pairs_minimum_clearance_m=self.minimum_m,
            rationale=self.interface_rationale,
            pairs=list(self.interface_observations.values()),
            scope="Listed faces may meet; all observed pairs remain checked for penetration.",
            distance_contract="Padded OBB SAT lower bound for boxes; raw MuJoCo query for other shapes",
            raw_geometry_query_discrepancies=self.geometry_query_discrepancies,
        )

    def move(self, target, *, speed_scale=1.):
        self.motion.geometry_query_discrepancies = []
        report = super().move(target, speed_scale=speed_scale)
        report["direct_clearance"]["support_interfaces"] = self.interface_report()
        report["direct_clearance"]["planning_geometry_query_discrepancies"] = list(self.motion.geometry_query_discrepancies)
        return report
