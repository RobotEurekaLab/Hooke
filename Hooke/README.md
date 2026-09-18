# Hooke simulator

The MuJoCo-based lab-automation simulator, task definitions, and
demonstration-generation/rendering pipeline. Originally the `autobio/`
directory from [AutoBio](https://arxiv.org/abs/2505.14030); renamed to
match the overall [Hooke](../README.md) project as it diverges from the
original release (see `archetypes/` below and `private/technical-log.md`
for what's changed and why).

## Requirements
### System requirement
- Ubuntu >= 20.04 (24.04 recommanded)
- WSL supported

### Environment setup with conda
```bash
conda create -n autobio python=3.12
conda activate autobio
pip install 'mujoco==3.3.0' numpy scipy jax[cpu] toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image
# Optional: only needed for archetypes/compose_protocol.py (Phase I step 4,
# LLM-driven protocol -> task-sequence composition)
pip install anthropic openai
# Optional: only needed for webui/ (Phase I step 5, interactive task-scene picker)
pip install flask
```

Isaac runs in its own standalone installation. For automatic discovery,
account permissions and server deployment, see the
[Isaac installation guide](../docs/isaac_setup.md).

## File structure
### folders
- assets/ *3D models for lab assets*
- grasp/ *forward kinematics, quaternion*
- logs/ *storing generated simulation data*
- model/ *simulation-ready lab assets/scene in MJCF language*
- packages/ *websocket for inference using remote server*

### files
#### task definition
- load_centrifuge_5430.py *load centrifuge 5430 rotor*
- mani_centrifuge_5430.py *close/open lid of centrifuge 5430*
- mani_centrifuge_5910.py *close/open lid of centrifuge 5910*
- mani_centrifuge_mini.py *close/open lid of cenrtrifuge mini (desktop)*
- mani_pipette.py *aspirate liquid with pipette*
- mani_thermal_cycler.py *close/open lid of thermal cycler*
- mani_thermal_mixer.py  *set parameters (time, temperature, frequency) on a mixer panel*
- mani_vortex_mixer.py *close/open lid of vortex mixer*
- pickup_centrifuge_tube *pick up a centrifuge tube from its rack*
- screw_loosen.py *unscrew centrifuge tube cap*
- screw_tighten.py *screw on centrifuge tube cap*
- transfer_centrifuge_tube.py *transfer a tube to a specified rack slot*

#### noteble utilities
- task.py *abstract class for AutoBio tasks*
- render.py *get visual output for each camera from offline simulation data*
- kinematics.py *provide inverse kinematics for ur5e and analytical IK for aloha arm*
- evaluate.py *for policy evaluation*
- instrument.py *define functionalities or behaviors for lab instruments*

## Data generation
Run python file of *task definition*
```bash
python ./[task_definition].py
```
Each simulation trajectory will be saved into `./logs/[task_name]/[timestamp]/`. All trajectory samples share a common `.mjb` file which stores the scene information in binary.

After simulation data acquired, you may run 
```bash
bash render.bash "[task_name]"
```
to get the visual output for each camera in the scenario.

## `archetypes/`

Added on top of the original release to make the task/asset library
scale rather than requiring a new hand-written file per instrument
variant. See `private/technical-log.md` (not published; ask the repo owner)
for the full write-up, in brief:

- `expert_common.py` — shared UR5e/motion-primitive code every task file
  used to duplicate.
- `lever_lock_centrifuge.py` + `centrifuge_specs.py` — a generic archetype
  for the "grip a lever, close it, engage a lock" instrument family
  (`mani_centrifuge_5430.py`/`5910.py` are now ~20-line instantiations of it).
- `rotor_variants.py` — generates alternate rotor-capacity variants of the
  centrifuge 5430 asset by editing its MJCF `<replicate>` block.
- `validate_task.py` / `validate_rotor_variants.py` / `validate_catalog.py` —
  a generic harness that runs a task's scripted expert and reports
  compile/stability/success-rate stats.
- `task_catalog.py` + `compose_protocol.py` — a catalog of verified atomic
  tasks and an LLM-driven pipeline that composes a free-text protocol
  description into an ordered sequence of them.

## `webui/`

An interactive scene picker: pick a category, then a task, then a robot to
place in the scene, then (for tasks with generated asset variants, e.g.
rotor slot count) a variant, and see a rendered preview. Run:
```bash
export MUJOCO_GL=egl
python -m webui.server
```
then open `http://localhost:8080/`.

**Generate a custom task scene**: describe an instrument/asset (optionally
with a reference photo), paste your own OpenAI API key (needs GPT-6 Astra
access), and it writes a Blender (`bpy`) script, statically rejects it if
it tries to touch the filesystem/network/subprocesses, runs the (approved)
script in a real headless Blender, and renders the exported mesh. Your key
is used only for that one request -- never written to disk, logged, or
kept afterwards (see `webui/custom_gen.py`'s module docstring for the full
security model). This is single-asset generation, not yet a full
task+scene generator -- see `private/TODO.md`.

**Robot picker**: every task offers its native robot plus, for
UR5e-native tasks, arm-mount alternatives (Franka Panda, UFACTORY xArm7)
that swap in at the same tabletop mount point; every task also offers
floor-mount bystanders (Unitree G1 humanoid, PAL Tiago Dual mobile
manipulator) that stand beside the table instead. These are placed for
visualization only -- swapping the robot does not adapt IK/motion
primitives to it, so the preview shows the scene's default pose, not the
task's actual reset() state (see `webui/robot_scene.py` and
`private/technical-log.md` for why, and `private/TODO.md` for the
functional-execution version of this that's deliberately not attempted
yet). Robot MJCF models beyond the original UR5e/Aloha assets are vendored
from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
under `model/robot_menagerie/` (each subdirectory keeps its own LICENSE).
