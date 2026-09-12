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
conda create -n autobio python=3.11
conda activate autobio
pip install 'mujoco==3.3.0' numpy scipy jax[cpu] toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image
# Optional: only needed for archetypes/compose_protocol.py (Phase I step 4,
# LLM-driven protocol -> task-sequence composition)
pip install anthropic
```

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
