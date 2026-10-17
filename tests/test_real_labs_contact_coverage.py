"""Read-only zero-contact screen, including rigid-group ownership boundaries."""

import unittest

import mujoco

from real_labs.contact_coverage import (
    screen_contact_coverage,
    ZERO_CONTACT_ISSUE,
    TRANSMISSION_ISSUE,
)
from real_labs.motion import _model_signature
from real_labs.usability import audit_model


def fixture(
    *,
    proxy=False,
    proxy_on_fixed_child=False,
    child=False,
    head_enabled=False,
    child_enabled=True,
    explicit=False,
    material_alpha=1,
    visual_group=0,
    tendon=False,
):
    contact = (
        'contype="1" conaffinity="1"' if head_enabled else 'contype="0" conaffinity="0"'
    )
    proxy_xml = (
        '<geom name="proxy" type="box" size=".02 .02 .02" pos=".6 0 0" group="3"/>'
        if proxy
        else ""
    )
    if proxy_on_fixed_child:
        proxy_xml = '<body name="fixed_mount">' + proxy_xml + "</body>"
    child_xml = ""
    if child:
        masks = "1" if child_enabled else "0"
        child_xml = f"""<body name="child" pos="0 0 .15"><joint name="child_joint" type="slide" range="-.1 .1"/>
          <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
          <geom name="child_shape" type="box" size=".03 .03 .03" contype="{masks}" conaffinity="{masks}"/>
        </body>"""
    actuator = (
        '<position name="head_drive" joint="head_joint" kp="10" ctrlrange="-.1 .1"/>'
    )
    tendon_xml = ""
    if tendon:
        tendon_xml = '<tendon><fixed name="coupler"><joint joint="head_joint" coef="1"/></fixed></tendon>'
        actuator = '<motor name="tendon_drive" tendon="coupler"/>'
    xml = f"""<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
      <asset><material name="paint" rgba=".4 .5 .6 {material_alpha}"/>
        <mesh name="head_mesh" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"/>
      </asset><worldbody>
        <geom name="ground" type="plane" size="2 2 .1" pos="0 0 -.5"/>
        <body name="stationary_housing"><geom name="fixed_cover" type="box" size=".1 .1 .1" pos="1 0 0"/></body>
        <body name="head"><joint name="head_joint" type="slide" axis="1 0 0" range="-.1 .1"/>
          <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
          <geom name="head_shell" type="mesh" mesh="head_mesh" material="paint" group="{visual_group}" {contact}/>
          {proxy_xml}{child_xml}
        </body>
      </worldbody>
      {tendon_xml}
      <actuator>{actuator}{'<position name="child_drive" joint="child_joint" kp="10" ctrlrange="-.1 .1"/>' if child else ''}</actuator>
      {'<contact><pair geom1="head_shell" geom2="ground"/></contact>' if explicit else ''}
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


class ContactCoverageTests(unittest.TestCase):
    def test_all_visual_actuated_head_is_not_covered_by_ground(self):
        report = screen_contact_coverage(fixture())
        self.assertEqual(report["issues"], [ZERO_CONTACT_ISSUE])
        self.assertEqual(report["zero_contact_visible_group_count"], 1)
        group = report["groups"][0]
        self.assertEqual(group["visible_count"], 1)
        self.assertEqual(group["contact_disabled_count"], 1)
        self.assertEqual(group["contact_enabled_count"], 0)
        self.assertNotIn("ground", group["contact_enabled_geoms"])
        self.assertNotIn("stationary_housing", group["bodies"])

    def test_visible_mesh_with_hidden_proxy_in_same_weld_group_is_not_flagged(self):
        report = screen_contact_coverage(fixture(proxy=True))
        self.assertEqual(report["issues"], [])
        group = report["groups"][0]
        self.assertEqual(group["visible_count"], 1)
        self.assertEqual(group["contact_disabled_count"], 1)
        self.assertEqual(group["contact_enabled_count"], 1)
        self.assertEqual(group["hidden_contact_enabled_count"], 1)
        # The proxy is deliberately remote: the screen must not claim that
        # counting it validates coverage of the visible mesh's actual surface.
        self.assertEqual(group["status"], "has_contact_shapes_unverified_coverage")
        self.assertIn("do not establish complete", report["scope"]["limitations"])

    def test_fixed_descendant_proxy_belongs_to_parent_rigid_group(self):
        report = screen_contact_coverage(fixture(proxy=True, proxy_on_fixed_child=True))
        self.assertEqual(report["issues"], [])
        self.assertIn("fixed_mount", report["groups"][0]["bodies"])
        self.assertEqual(report["groups"][0]["contact_enabled_geoms"], ["proxy"])

    def test_child_joint_collision_geometry_does_not_cover_visual_parent(self):
        report = screen_contact_coverage(fixture(child=True))
        self.assertEqual(report["zero_contact_visible_group_count"], 1)
        groups = {g["weld_body"]: g for g in report["groups"]}
        self.assertEqual(groups["head"]["issue"], ZERO_CONTACT_ISSUE)
        self.assertEqual(groups["head"]["contact_enabled_count"], 0)
        self.assertNotIn("child", groups["head"]["bodies"])
        self.assertEqual(groups["child"]["contact_enabled_count"], 1)
        self.assertIsNone(groups["child"]["issue"])

    def test_parent_collision_geometry_does_not_cover_visual_child_joint(self):
        report = screen_contact_coverage(
            fixture(head_enabled=True, child=True, child_enabled=False)
        )
        groups = {g["weld_body"]: g for g in report["groups"]}
        self.assertIsNone(groups["head"]["issue"])
        self.assertEqual(groups["child"]["issue"], ZERO_CONTACT_ISSUE)
        self.assertEqual(groups["child"]["contact_enabled_count"], 0)

    def test_explicit_pair_is_configured_contact_even_with_zero_masks(self):
        report = screen_contact_coverage(fixture(explicit=True))
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["groups"][0]["explicit_pair_geoms"], ["head_shell"])
        self.assertEqual(report["groups"][0]["contact_enabled_count"], 1)

    def test_render_visibility_matches_groups_and_inherited_material_alpha(self):
        for model in (fixture(visual_group=3), fixture(material_alpha=0)):
            report = screen_contact_coverage(model)
            self.assertEqual(report["issues"], [])
            self.assertEqual(report["visible_actuated_weld_group_count"], 0)
        self.assertEqual(
            screen_contact_coverage(fixture(material_alpha=0.05))["issues"],
            [ZERO_CONTACT_ISSUE],
        )

    def test_unsupported_transmission_is_unexamined_not_claimed_covered(self):
        report = screen_contact_coverage(fixture(tendon=True))
        self.assertEqual(report["issues"], [TRANSMISSION_ISSUE])
        self.assertEqual(report["actuated_weld_group_count"], 0)
        self.assertEqual(report["unexamined_actuators"][0]["name"], "tendon_drive")

    def test_model_bytes_remain_unchanged_and_global_flags_are_reported(self):
        model = fixture(proxy=True, child=True)
        model.opt.disableflags = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        before = _model_signature(model)
        for _ in range(2):
            report = screen_contact_coverage(model)
            self.assertTrue(report["global_physics"]["contact_disabled"])
        self.assertEqual(_model_signature(model), before)

    def test_usability_appends_issue_and_preserves_existing_raw_issue(self):
        model = fixture(child=True)
        # A child contact box penetrates a real ground plane at home; the
        # parent's all-visual shell is an additional, independent problem.
        model.geom_pos[model.geom("ground").id, 2] = 0.16
        manifest = dict(
            id="fixture",
            room=dict(size=[4, 4, 2]),
            equipment=[
                dict(
                    id="head",
                    kind="fixture",
                    joints={"head": "head_joint", "child": "child_joint"},
                    actuators={"head": "head_drive", "child": "child_drive"},
                )
            ],
        )
        before = _model_signature(model)
        result = audit_model(
            model, manifest, samples=3, pair_samples=2, run_commands=False
        )
        self.assertIn("home_pose_interpenetration", result["issues"])
        self.assertIn("requires_geometry_or_motion_review", result["issues"])
        self.assertIn(ZERO_CONTACT_ISSUE, result["issues"])
        self.assertTrue(result["home_pose_penetrations"])
        self.assertEqual(
            result["moving_contact_coverage"]["zero_contact_visible_group_count"], 1
        )
        self.assertEqual(_model_signature(model), before)


if __name__ == "__main__":
    unittest.main()
