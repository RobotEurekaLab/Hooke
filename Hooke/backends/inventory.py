"""Compile catalogue scenes and record features that need backend qualification.

Run from Hooke/: ../.venv/bin/python -m backends.inventory --output ../temp/backend_parity/inventory.json
No task is marked compatible from import/compile success alone.
"""
from __future__ import annotations
import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap
import time


def inspect_catalog():
    import mujoco
    from archetypes.task_catalog import CATALOG
    rows = []
    for name, entry in CATALOG.items():
        row = {"task": name, "module": entry.module, "class": entry.cls,
               "robot": entry.robot, "isaac_status": "unqualified"}
        started = time.monotonic()
        try:
            cls, _ = entry.load_classes()
            spec = cls.load()
            model = spec.compile()
            check = ast.parse(textwrap.dedent(inspect.getsource(cls.check))).body[0].body
            trivial = (len(check) == 1 and isinstance(check[0], ast.Return)
                       and isinstance(check[0].value, ast.Constant) and check[0].value.value is True)
            always_false = (len(check) == 1 and isinstance(check[0], ast.Return)
                            and isinstance(check[0].value, ast.Constant) and check[0].value.value is False)
            row.update(scene=str(cls.default_scene), mujoco_compiles=True,
                       display_only=cls.__name__ == "StaticDisplayTask", check_constant_true=trivial,
                       check_constant_false=always_false,
                       bodies=model.nbody, geoms=model.ngeom, meshes=model.nmesh,
                       joints=model.njnt, actuators=model.nu, cameras=model.ncam,
                       equalities=model.neq, tendons=model.ntendon, plugins=model.nplugin,
                       sensors=model.nsensor, timestep_s=model.opt.timestep,
                       equality_types=sorted(set(int(x) for x in model.eq_type)),
                       actuator_transmissions=sorted(set(int(x) for x in model.actuator_trntype)))
        except Exception as exc:
            row.update(mujoco_compiles=False, error=f"{type(exc).__name__}: {exc}")
        row["inspect_wall_s"] = time.monotonic() - started
        rows.append(row)
        print(json.dumps({"task":name,"compiled":row['mujoco_compiles']},ensure_ascii=False),flush=True)
        yield rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    for rows in inspect_catalog():
        document={"schema_version":1,"scope":"Catalogue scenes including generated variants; compilation is not functional qualification", "entries":rows}
        args.output.write_text(json.dumps(document,indent=2,ensure_ascii=False))

if __name__ == '__main__': main()
