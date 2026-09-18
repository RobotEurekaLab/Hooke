"""Acquire the pinned open hardware CAD without executing FreeCAD features."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import ssl
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile

from microscopy.parallel_cad import LICENSE_SHA256
from microscopy.parallel_kinematics import SOURCE_REVISION

SOURCE = "https://github.com/0x23/MicroManipulatorStepper"
DOCUMENTS = (
    "Assembly_Actuator", "Assembly_BallJointPlate", "Assembly_EndEffector",
    "Assembly_LinkageUnit", "Assembly_MicroManipulator", "BallJointPlate",
    "BaseBlock", "CrimpBead", "EncoderMagnet", "EncoderMagnetArray",
    "EndEffector", "JointBall", "LinkageRod", "MT6835", "MotorHorn", "MotorMount",
    "RubberBand", "RubberBandCollet", "StepperMotorNema17",
)


def acquire(output, ca_file=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(success=False, revision=SOURCE_REVISION, source=SOURCE, documents={},
                    scope="Pinned source acquisition and saved ZIP/XML integrity; no feature regeneration or hardware calibration")

    def save():
        (output / "acquisition.json").write_text(json.dumps(manifest, indent=2))

    save()
    try:
        context = ssl.create_default_context(cafile=ca_file)
        paths = ["LICENSE"] + [f"construction/micro_manipulator/{name}.FCStd" for name in DOCUMENTS]
        for path in paths:
            url = f"https://raw.githubusercontent.com/0x23/MicroManipulatorStepper/{SOURCE_REVISION}/{path}"
            with urlopen(Request(url, headers={"User-Agent": "Hooke-CAD-acquisition"}), timeout=60, context=context) as response:
                payload = response.read()
            digest = hashlib.sha256(payload).hexdigest()
            name = Path(path).name
            if name == "LICENSE" and digest != LICENSE_SHA256:
                raise ValueError("Upstream license differs from the pinned hardware notice")
            file = output / name
            file.write_bytes(payload)
            record = dict(source_path=path, bytes=len(payload), sha256=digest)
            if name.endswith(".FCStd"):
                with zipfile.ZipFile(file) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f"CAD ZIP integrity failure: {name}")
                    xml = archive.read("Document.xml")
                document = ET.fromstring(xml)
                record["external_cad_references"] = sorted(set(
                    re.findall(r"([A-Za-z0-9_]+\.FCStd)", xml.decode())))
                license_property = document.find("./Properties/Property[@name='License']/String")
                record["document_license_metadata"] = license_property.get("value") if license_property is not None else None
            manifest["documents"][name] = record
            save()
        missing = sorted({reference for record in manifest["documents"].values()
                          for reference in record.get("external_cad_references", [])}
                         - manifest["documents"].keys())
        manifest["unresolved_external_cad_filenames"] = missing
        if missing:
            raise ValueError(f"CAD acquisition has unresolved references: {missing}")
        manifest.update(success=True,
                        license="Exact upstream author-modified MIT text explicitly covers hardware designs; preserved in LICENSE",
                        license_metadata_note="FCStd document properties may retain All rights reserved; the pinned repository notice explicitly covers hardware design. Metadata is preserved separately.")
        save()
        return manifest
    except Exception as error:
        manifest["error"] = str(error)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New ignored directory; existing directories are preserved")
    parser.add_argument("--ca-file", type=Path, help="Explicit trusted CA bundle for environments with missing default certificates")
    args = parser.parse_args()
    report = acquire(args.output, args.ca_file)
    print(json.dumps(dict(success=report["success"], revision=report["revision"], documents=len(report["documents"]))))


if __name__ == "__main__":
    main()
