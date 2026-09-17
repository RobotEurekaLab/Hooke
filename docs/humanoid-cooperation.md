# Humanoid and rover cooperation demo

Open `/surface-missions` and select **G1 人形机器人 + 采样车**. The Moon and Mars retain their original local gravity and outdoor terrain. The G1 and the rover each have a floating base, with no root weld, prescribed root trajectory, or external stabilizing force.

The G1 walks beside the rover. At the sample location it reaches with its original left hand and presses a spring-return confirmation button on the rover. Collection starts only after actual left-hand contact, at least 4 mm of button travel and 50 ms of sustained contact. Contact detection includes the original palm and finger collision meshes and the added fingertip pad, excluding the wrist mesh and arm. The hand retracts as soon as confirmation is established. The G1 then steps away while the rover's UR5e / Robotiq collects the rock into its physical bin. Both robots return to the landing zone. Active cooperation aborts immediately if the continuous upright criterion fails.

The 15 cm square confirmation paddle sits about 0.98 m above the level sampling corridor, with a 20 N/m return spring and 1 N·s/m damping. It provides a reachable, light contact target while retaining physical slide motion and contact-based confirmation.

The G1 does not grasp the rock. This demo separates locomotion and contact authorization from the existing rover collection workflow; it does not implement perception-based survey, humanoid rock grasp or quadruped manipulation.

## Verified demo episodes

The deterministic seed-0 layout completes the full mission in both engines. Each episode passes all seventeen rover and humanoid checks. These are simulation durations, not wall-clock runtimes.

| World | Engine | Simulation duration | G1 recorded travel | Maximum pelvis tilt |
| --- | --- | ---: | ---: | ---: |
| Moon | MuJoCo | 163.262 s | 15.742 m | 17.45° |
| Moon | Isaac / PhysX | 160.196 s | 16.715 m | 18.95° |
| Mars | MuJoCo | 123.718 s | 15.557 m | 8.78° |
| Mars | Isaac / PhysX | 118.416 s | 16.276 m | 9.13° |

Original episodes and derived gallery files remain in ignored local storage. The measurements establish this demo layout's completion; they do not establish general walking robustness on arbitrary terrain.

No-action controls use the expert's duration as their requested horizon. Controls can finish earlier when G1's continuous upright history fails: a past excessive tilt or insufficient pelvis height cannot become valid later in the same episode. Such results use `CONTROL_REJECTED_EARLY` and retain the requested horizon, actual duration and failed check. Qualification requires the original controls to remain held, the requested horizon to match, and the recorded upright check to be false. Full-duration controls remain separately identified as `CONTROL_COMPLETE`.

## Prepare the gait

The robot geometry and dexterous hands reuse the repository's Menagerie G1. The recurrent gait reuses the BSD-3-Clause [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) G1 policy at revision `276801e46c5d433564f24658bac64f254b7d2d4b`. The preparation command checks the original file hashes, retains the upstream license, and validates 200 recurrent inference steps against TorchScript before publishing the NumPy arrays.

From `Hooke/`:

```bash
python -m surface.gait_assets --torch-python /path/to/python-with-torch
```

Torch is required once for export. Simulation uses NumPy / SciPy, without Torch or a training service. Downloaded weights, the derived arrays and their provenance stay under ignored `temp/surface_missions/gait/`. The upstream license is also retained in [third-party/unitree-g1-gait.LICENSE](third-party/unitree-g1-gait.LICENSE). The original model's license remains in its model directory.

This project adapts the policy for low gravity: for `r = g / 9.81`, servo stiffness and joint dry friction scale with `r`; damping and the policy clock scale with `sqrt(r)`; joint/angular velocity observations and velocity commands use the corresponding time scale. This is our simulation adaptation, not an upstream claim of Moon or Mars validation. Arm motion uses numerical kinematics on scratch state and writes actuator targets. Reaching and smooth joint-space retraction also use the gravity-adjusted clock. Only reset sets robot poses.

## Run and qualify

Choose an idle GPU and configure the local Isaac installation as for other tasks:

```bash
python -m backends.run \
  --task space_lunar_humanoid_rover --backend isaac \
  --mode expert --seed 0 --gpu 6 --max-sim-seconds 180 \
  --output ../temp/humanoid_demo/my-lunar-team
```

Use `space_martian_humanoid_rover` for Mars or `--backend mujoco`. Preview mode does not require prepared gait weights and does not enable locomotion or arm authorization.

```bash
HOOKE_ISAAC_TASK_THREADS=16 \
HOOKE_RENDER_FPS=1 HOOKE_RENDER_WIDTH=1280 HOOKE_RENDER_HEIGHT=720 \
HOOKE_ISAAC_COLOR_PIPELINE=source_display \
python -m surface.qualify --scenario team \
  --output ../temp/humanoid_demo/campaign --gpu 6

python -m surface.publish \
  ../temp/humanoid_demo/campaign/lunar-mujoco \
  ../temp/humanoid_demo/campaign/lunar-mujoco-no-action \
  ../temp/humanoid_demo/campaign/martian-mujoco \
  ../temp/humanoid_demo/campaign/martian-mujoco-no-action \
  ../temp/humanoid_demo/campaign/lunar-isaac \
  ../temp/humanoid_demo/campaign/lunar-isaac-no-action \
  ../temp/humanoid_demo/campaign/martian-isaac \
  ../temp/humanoid_demo/campaign/martian-isaac-no-action
```

The team publisher uses a separate summary and file prefix, preserving the solo-rover gallery. Videos synchronize each robot's actual recorded route and distance. Success combines the rover's ten physical checks with G1 travel, support contacts, continuous upright posture, contact authorization, return and low final base speed. Qualification is deterministic and seed 0; it does not establish randomized terrain robustness, pixel/physics equivalence or hardware readiness.

## Render a recorded native episode

For expensive native runs, qualification can use `--no-render`. After the native expert succeeds, its recorded states can be rendered separately:

```bash
HOOKE_ISAAC_TASK_THREADS=16 \
HOOKE_RENDER_FPS=1 HOOKE_RENDER_WIDTH=1280 HOOKE_RENDER_HEIGHT=720 \
HOOKE_ISAAC_COLOR_PIPELINE=source_display \
python -m surface.replay_render ../temp/humanoid_demo/campaign/lunar-isaac \
  --gpu 6 --output ../temp/humanoid_demo/rendered/lunar-isaac
```

Publish the rendered expert directory together with the original matched no-action directory. This viewer sets poses only to replay states recorded during the completed native physics episode. It never reruns or stabilizes that episode. Rendering rejects failed experts, checks reconstructed positions and velocities, and verifies that rendering advances no physics events. The original physics result and trajectory remain untouched; the derived result and gallery retain replay provenance and their hashes.
