"""Private CAD appearance and explicit rigid slide assemblies.

Vendor meshes remain outside Git. The model separates their geometry from
contact proxies and uncalibrated drive dynamics. CAD geometry alone does not
establish real device accuracy, mass properties, optical fidelity or rights.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, knurled_knob, numbers, rounded_box, slide

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "temp/microscopy_research/cad_meshes"
ZABER_SHA256 = "cec4b2683dfbb979e1a6ae4a631578a8df0615f077f7c490e9e5462c2f543077"
# Original tool extensions place the unchanged OEM housings outside the
# stage and illuminator pillar. Their stiffness/precision is not calibrated.
TOOL_EXTENSIONS = {"probe": .38, "gripper": .23, "injector": .33, "holder": .38}


class CadAssets:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("HOOKE_MICROSCOPY_ASSET_ROOT", DEFAULT_ROOT)).resolve()
        self._loaded = {}
        self._registered = set()
        self.evidence = dict(profile="cad", assemblies=[],
                             dynamics="Uncalibrated position-servo and rigid-slide model",
                             collisions="Tool and sample contact proxies; CAD housings are visual meshes")

    def read(self, name, expected_hash=None):
        if name not in self._loaded:
            directory = self.root / name
            path = directory / "manifest.json"
            if not path.is_file():
                raise FileNotFoundError(f"Requested microscopy CAD is unavailable: {path}")
            report = json.loads(path.read_text())
            if report.get("mesh_length_unit") != "m" or not report.get("parts"):
                raise ValueError(f"CAD {name} has no validated metre meshes")
            for part in report["parts"]:
                mesh_path = (directory / part["file"]).resolve()
                if mesh_path.parent != directory.resolve() or not mesh_path.is_file():
                    raise ValueError(f"CAD {name} contains an unavailable part mesh")
                if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != part.get("mesh_sha256"):
                    raise ValueError(f"CAD {name} part mesh differs from the inspected artifact")
            self._loaded[name] = report
        report = self._loaded[name]
        if expected_hash is not None and report.get("sha256") != expected_hash:
            raise ValueError(f"CAD {name} does not match the inspected STEP source")
        return report

    def interfaces(self, name):
        """Read offline metrology bound to the same STEP as the runtime mesh."""
        report = self.read(name)
        path = self.root/name/"interfaces.json"
        interfaces = json.loads(path.read_text())
        if interfaces.get("source_sha256") != report["sha256"] or interfaces.get("units") != "mm":
            raise ValueError(f"CAD {name} mounting metrology differs from its STEP source")
        return interfaces

    def attach(self, asset, node, name, index, position, rotation=np.eye(3), material="metal", prefix=None):
        report = self.read(name)
        key = f"cad_{name.replace('/', '_').replace('-', '_')}_{index}"
        if key not in self._registered:
            ET.SubElement(asset, "mesh", name=key,
                          file=str(self.root / name / report["parts"][index]["file"]))
            self._registered.add(key)
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        ET.SubElement(node, "geom", name=f"{prefix or node.get('name')}_{key}", type="mesh",
                      mesh=key, pos=numbers(position), quat=numbers(quaternion), material=material,
                      contype="0", conaffinity="0", mass="0")

    def zaber(self, asset, world, actuators, tool, home, origin, bench_top, direction, *, compact=False,
              back=None):
        """Build the M-LSM RHF chain using factory names and mounting drawing.

        CAD Y is vertical on the flat-base configuration. Assembly 1 translates
        along CAD Z, assembly 3 along CAD Y, and assembly 2 along CAD X.
        Each slide is centred from its base/carriage block CAD bounds; this
        changes the nominal CAD pose and is recorded as an assembly transform.
        """
        report = self.read("zaber-rhf", ZABER_SHA256)
        if len(report["parts"]) != 184:
            raise ValueError("Unexpected Zaber assembly part count")
        yaw = (dict(probe=np.pi/2, holder=np.pi/2, gripper=-np.pi/2, injector=-np.pi/4)[tool]
               if compact else float(np.arctan2(direction[1], direction[0])))
        rotation = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("x", 90, degrees=True).as_matrix()
        root = body(world, tool+"_mount", origin+home)
        # The near-field tool is supported by a declared original extension;
        # no OEM mass/inertia or instrument positioning precision is inferred.
        back = (np.r_[direction[:2]/2**.5, 1/2**.5] if back is None else np.asarray(back))
        extension = .180 if compact else TOOL_EXTENSIONS[tool]
        holder_world = back*extension
        holder_cad = np.array([.00892, .05680, .0612058])
        shifts = self._zaber_centres(report)
        translation = holder_world-rotation@(holder_cad+shifts["x"])
        fixed_base = rotation@np.array([0., -.05624675, -.00148])+translation
        foot = fixed_base.copy()
        radial = fixed_base[:2]+home[:2]
        if not compact:
            foot[:2] = radial/np.linalg.norm(radial)*.255-home[:2]
        else:
            # Keep the left stand camera clear of the support pole. The short
            # lateral support arm is an independent, uncalibrated accessory.
            foot[0] += .045 if direction[0] > 0 else -.045
        foot[2] = bench_top+.012-(origin[2]+home[2])
        geom(root, tool+"_foot", "box", (.046, .039, .012), foot, "graphite")
        support = np.r_[foot[:2], fixed_base[2]-.023]
        cylinder(root, tool+"_post", foot+[0, 0, .012], support, .016)
        cylinder(root, tool+"_support_arm", support, fixed_base-[0, 0, .009], .012)
        geom(root, tool+"_post_platform", "box", (.036, .036, .004), fixed_base-[0, 0, .004], "metal")
        # Body frames stay unrotated; slide axes and geometry use the mounting
        # rotation explicitly so both adapters read the same physical axes.
        y = slide(root, actuators, tool+"_y", -rotation[:, 2], (-.0125, .0125))
        z = slide(y, actuators, tool+"_z", rotation[:, 1], (-.0125, .0125))
        x = slide(z, actuators, tool+"_x", rotation[:, 0], (-.0125, .0125))
        links = {"fixed": root, "y": y, "z": z, "x": x}
        part_links = []
        for index, part in enumerate(report["parts"]):
            link = self._zaber_link(part.get("source_name", ""))
            part_links.append(link)
            self.attach(asset, links[link], "zaber-rhf", index,
                        translation+rotation@shifts[link], rotation,
                        self._zaber_material(part["source_name"]), prefix=tool)
        needle_holder = None
        if compact and tool in ("injector", "holder"):
            from microscopy.needle_holder import CAPILLARY_EXPOSED_M, HOLDER_LENGTH_M, mount_holder
            cylinder(x, tool+"_holder_adapter", holder_world,
                     back*(CAPILLARY_EXPOSED_M+HOLDER_LENGTH_M), .002)
            needle_holder = mount_holder(x, tool, back)
        elif tool in ("probe", "holder") and not compact:
            route = (holder_world, np.array([-.230, -.120, .200]),
                     np.array([-.075, -.070, .075]), back*.025)
            for index, (start, end) in enumerate(zip(route, route[1:])):
                cylinder(x, tool+"_boom"+str(index), start, end, .004)
        else:
            cylinder(x, tool+"_boom", holder_world, back*.025, .004)
        if needle_holder is None:
            cylinder(x, tool+"_collet", back*.025, back*.009, .002)
        ET.SubElement(x, "site", name=tool+"_tcp", size=".00005", group="5")
        self.evidence["assemblies"].append(dict(tool=tool, source_sha256=report["sha256"],
            parts=184, part_links=part_links, chain=[tool+"_y", tool+"_z", tool+"_x"],
            cad_to_world_rotation=rotation.tolist(), cad_to_tcp_translation_m=translation.tolist(),
            centred_link_offsets_m={k: v.tolist() for k, v in shifts.items()},
            travel_m=.025, tool_extension_m=extension,
            tool_back_direction=back.tolist(), tool_elevation_deg=float(np.rad2deg(np.arcsin(back[2]))),
            tool_layout="compact" if compact else "extended", needle_holder=needle_holder,
            mounting=("Independent close-mounted post and short holder adapter; OEM fixed/moving parts retained"
                      if compact else "Original elevated cantilever post and tool extension; CAD fixed and moving components retained"),
            rights=report["rights"], articulation="Rigid slides; lead screw rotation and bearing roll omitted"))
        return x

    def openframe(self, asset, world, origin, bench_top):
        """Stack the open hardware frame; peripherals are declared originals.

        Layer zero planes coincide with the preceding layer's top. The 5 mm
        male dovetails consequently enter the preceding layer rather than
        increasing the stack height. No eyepiece is added to this camera scope.
        """
        scope = body(world, "microscope", [*origin[:2], bench_top])
        from microscopy.cad_optics import camera_profile, mount_camera
        optical_profile = camera_profile()
        transforms = []

        def part(name, position, rotation=np.eye(3), prefix=None):
            report = self.read("openframe/"+name)
            if report.get("upstream_commit") != "19c312931f4fdcbfaf3725fbef3db075a32413c6":
                raise ValueError("openFrame component differs from the inspected revision")
            self.attach(asset, scope, "openframe/"+name, 0, position, rotation,
                        "graphite", prefix=prefix)
            transforms.append(dict(component=name, sha256=report["sha256"],
                                   position_m=np.asarray(position).tolist(), rotation=rotation.tolist()))

        z = 0.
        layers = (("OF-LL-BP", .052), ("OF-LL-TL", .030),
                  ("OF-LL-CORE", .072), ("OF-LL-FL-MOT", .076), ("OF-LL-SP", .011))
        for name, height in layers:
            part(name, [0., 0., z])
            z += height
        stage_zero = z-.011
        # Four pinned CAD centres are 10 x 40 mm, and match the corresponding
        # countersunk stage-plate holes after this translation exactly. The
        # older fabrication PDF uses a different horizontal dimension.
        adapter = np.array([-.156, -.009, stage_zero-.012])
        source_holes = self.interface_centres("OF-AD-SP-TI-PILLAR",
                                             [[.011, .029], [.021, .029], [.011, -.011], [.021, -.011]])
        stage_holes = self.interface_centres("OF-LL-SP",
                                            [[-.145, .020], [-.135, .020], [-.145, -.020], [-.135, -.020]])
        alignment_error = float(np.max(np.linalg.norm(source_holes+adapter[:2]-stage_holes, axis=1)))
        if alignment_error > 50e-6:
            raise ValueError("openFrame pillar mount holes do not match the stage plate")
        part("OF-AD-SP-TI-PILLAR", adapter)
        pillar = adapter+[-.010, -.016, .007]
        part("OF-TI-PILLAR", pillar)
        # Rotate the two-bore clamp so its horizontal arm crosses the optical
        # axis; this follows CAD bore centres rather than the part bbox.
        illumination = None
        if optical_profile == 'assembled':
            from microscopy.illumination_assembly import support_pose
            illumination = support_pose(self, pillar)
            clamp, rotation = illumination['clamp_position'], illumination['clamp_rotation']
            arm_start, arm_rotation = illumination['arm_position'], illumination['arm_rotation']
        else:
            delta = np.array([-.025061, .015, .010])
            phi = float(np.arctan2(pillar[1], pillar[0]))
            theta = phi+float(np.arccos(-delta[0]/np.linalg.norm(pillar[:2])))
            rotation = Rotation.from_euler("z", theta).as_matrix()
            clamp = pillar+[0, 0, .210]-rotation@[.010, 0, 0]
            arm_start = pillar+[0, 0, .210]+rotation@delta
            arm_direction = rotation@np.array([0., -1., 0.])
            arm_rotation = np.column_stack((rotation[:, 0], np.array([0., 0., 1.]), arm_direction))
        part("OF-AD-TI-PILLAR-TI-ARM", clamp, rotation)
        part("OF-TI-ARM", arm_start, arm_rotation)
        if illumination is not None:
            part('OF-AD-TI-ARM-LED-CAIRN', illumination['led_position'], illumination['led_rotation'])
        # Original mounting screws: appearance only, not screw/bolt mechanics.
        for x in (-.145, -.135):
            for y in (-.020, .020):
                cylinder(scope, f"scope_pillar_mount_screw_{x}_{y}",
                         (x, y, stage_zero-.009), (x, y, stage_zero+.012), .002)
        for side in (-1, 1):
            for layer_z in (.052, .082, .154, .230):
                cylinder(scope, f"scope_layer_set_screw_{side}_{layer_z}",
                         (side*.071, 0, layer_z-.0025),
                         (side*.078, 0, layer_z-.0025), .002)
        # A camera and illumination path are explicit original peripherals.
        # Their dimensions/poses are layout estimates, not Nikon/Cairn OEM CAD.
        camera_assembly = None
        if optical_profile in ("mechanical", "assembled"):
            camera_assembly = mount_camera(self, asset, scope)
        else:
            port_z = .118
            cylinder(scope, "scope_camera_port", (0, -.060, port_z), (0, -.103, port_z), .025)
            for i in range(4):
                cylinder(scope, f"scope_camera_ring_{i}", (0, -.090-i*.006, port_z),
                         (0, -.093-i*.006, port_z), .027, "graphite")
            rounded_box(asset, scope, "scope_camera", (.036, .032, .036),
                        (0, -.143, port_z), .004, "graphite")
            for x in np.linspace(-.028, .028, 8):
                geom(scope, f"scope_camera_heat_rib_{x}", "box", (.001, .031, .001),
                     (x, -.143, port_z+.037), "metal")
        illumination_assembly = None
        if illumination is not None:
            from microscopy.illumination_assembly import mount_illuminator
            illumination_assembly = mount_illuminator(self, asset, scope, illumination, origin[2]-bench_top)
        else:
            light_z = float(arm_start[2])
            cylinder(scope, "scope_led_head", (0, 0, light_z-.007), (0, 0, light_z+.048), .022, "ivory")
            cylinder(scope, "scope_led_dissipator", (0, 0, light_z+.035), (0, 0, light_z+.049), .027, "graphite")
            for i in range(5):
                cylinder(scope, f"scope_led_heat_ring_{i}", (0, 0, light_z+.028+i*.004),
                         (0, 0, light_z+.030+i*.004), .029, "metal")
            cylinder(scope, "scope_condenser", (0, 0, light_z-.061), (0, 0, light_z-.012), .021, "graphite")
            cylinder(scope, "scope_condenser_trim", (0, 0, light_z-.018), (0, 0, light_z-.010), .025, "bronze")
            cylinder(scope, "scope_condenser_lens", (0, 0, light_z-.062), (0, 0, light_z-.061), .017, "lens")
            knurled_knob(scope, "scope_condenser_adjustment", (.033, 0, light_z-.030),
                         (1, 0, 0), .012, .014)
        self.evidence["microscope"] = dict(name="openFrame inverted microscopy layout", components=transforms,
            rights="CERN-OHL-P-2.0; notices and modification record retained with private derived assets",
            layer_heights_m=[height for _, height in layers], dovetail_depth_m=.005,
            stage_pillar_hole_pattern_m=[.010, .040], stage_pillar_hole_alignment_error_m=alignment_error,
            original_peripherals=["Camera, objective/turret, electric stage and focus", "LED, condenser, screws and optical connections"],
            estimates=["Peripheral optical/mechanical calibration", "Fine-stage risers",
                       "Original illuminator connector without verified threads" if illumination is not None else "Illuminator mounting"],
            drawing_discrepancy="Older pillar adapter PDF dimension 44 mm; pinned STEP mounting-hole spacing 40 mm",
            scope="CAD core geometry and explicit assembly layout; not a complete commercial microscope CAD or optical calibration")
        self.evidence["microscope"]["optical_profile"] = optical_profile
        if camera_assembly is not None:
            self.evidence["microscope"]["camera_assembly"] = camera_assembly
        if illumination_assembly is not None:
            self.evidence['microscope']['illumination_assembly'] = illumination_assembly
        return scope

    def interface_centres(self, name, nominal):
        report = self.read("openframe/"+name)
        interfaces = json.loads((self.root/"openframe"/name/"interfaces.json").read_text())
        if interfaces.get("source_sha256") != report["sha256"] or interfaces.get("units") != "mm":
            raise ValueError("CAD mounting interfaces do not match the inspected STEP")
        centres = np.asarray([surface["point_mm"][:2] for surface in interfaces["cylinders"]
                              if 1.5 < surface["radius_mm"] < 2.1 and abs(surface["axis"][2]) > .999])*1e-3
        if centres.ndim != 2 or not len(centres):
            raise ValueError("CAD has no matching vertical mounting cylinders")
        actual = []
        for target in nominal:
            nearest = centres[np.argmin(np.linalg.norm(centres-target, axis=1))]
            if np.linalg.norm(nearest-target) > 50e-6:
                raise ValueError("Expected mounting-hole centre is absent from the CAD")
            actual.append(nearest)
        return np.asarray(actual)

    @staticmethod
    def _zaber_centres(report):
        offsets = {}
        for key, base_index, carriage_index, axis in (("y", 0, 18, 2), ("z", 84, 102, 1), ("x", 42, 60, 0)):
            base = np.asarray(report["parts"][base_index]["bounds_m"]).mean(axis=0)
            carriage = np.asarray(report["parts"][carriage_index]["bounds_m"]).mean(axis=0)
            offset = np.zeros(3)
            offset[axis] = base[axis]-carriage[axis]
            offsets[key] = offset
        return {"fixed": np.zeros(3), "y": offsets["y"],
                "z": offsets["y"]+offsets["z"], "x": offsets["y"]+offsets["z"]+offsets["x"]}

    @staticmethod
    def _zaber_link(name):
        match = re.match(r"LSM Recirculating Bearing Assembly-([123])/", name)
        if match:
            carriage = "/LSM Recirculating Bearing Carriage Assembly-" in name
            return {1: ("fixed", "y"), 3: ("y", "z"), 2: ("z", "x")}[int(match[1])][carriage]
        if name.startswith("AB103B-"):
            return "y"
        if name.startswith(("AB117, Probe holder-", "AP116, 360° bracket-1/")):
            return "x"
        if name.startswith(("base bracket-", "AP116, 360° bracket-2/")):
            return "fixed"
        screw = re.fullmatch(r"Screw M3 x 5 SC Low Profile-(\d+)-solid1", name)
        if screw:
            number = int(screw[1])
            if number in range(1, 5):
                return "fixed"
            if number in range(9, 13):
                return "z"
            if number in range(13, 21):
                return "y"
        raise ValueError(f"Unmapped Zaber component: {name!r}")

    @staticmethod
    def _zaber_material(name):
        return "graphite" if any(s in name for s in ("Stepper motor", "Cable", "X-HS05")) else "metal"

    def record(self, root):
        custom = root.find("custom")
        if custom is None:
            custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", name="microscopy_assets", data=json.dumps(self.evidence))


def requested_assets():
    profile = os.environ.get("HOOKE_MICROSCOPY_ASSETS", "reference")
    if profile not in ("reference", "cad"):
        raise ValueError("HOOKE_MICROSCOPY_ASSETS must be reference or cad")
    from microscopy.cad_optics import camera_profile
    if camera_profile() != "estimated" and profile != "cad":
        raise ValueError("Mechanical microscopy optics require the explicit CAD asset profile")
    return CadAssets() if profile == "cad" else None


def asset_evidence(model):
    import mujoco
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "microscopy_assets")
    if index < 0:
        return dict(profile="reference", fidelity="Original photo-reference geometry, not dimensionally reproduced OEM CAD")
    start, length = int(model.text_adr[index]), int(model.text_size[index])
    return json.loads(bytes(model.text_data[start:start+length]).rstrip(b"\0"))
