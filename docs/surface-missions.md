# Planetary surface sampling

Open `/surface-missions` in the Web UI to explore the lunar and Martian scenes, select a camera, or run a sampling mission.

Select **G1 人形机器人 + 采样车** for the [humanoid cooperation demo](humanoid-cooperation.md). Its gait preparation and qualification are documented separately.

Each 80 × 80 m scene contains generated landforms, a landing zone, a lander, survey markers and a six-wheel rover with the existing UR5e / Robotiq assets. The rover drives to a sample, grips and lifts it through contact, releases it into a physical onboard bin, and returns. No pose teleportation or specimen attachment is used.

## Run a mission

From `Hooke/`, using the project environment:

```bash
python -m backends.run \
  --task space_lunar_surface_sampling --backend isaac \
  --mode expert --seed 0 --gpu 6 \
  --output ../temp/surface_missions/my-lunar-run
```

Use `space_martian_surface_sampling` for Mars and `--backend mujoco` for MuJoCo. Add `--no-render` for controller and physics checks. Prefer fresh output directories to preserve earlier evidence. Configure `HOOKE_ISAAC_PATH` for the installed Isaac Sim and choose an idle GPU.

For recorded media, set `HOOKE_RENDER_WIDTH=1280`, `HOOKE_RENDER_HEIGHT=720`, and `HOOKE_RENDER_FPS=2`. Isaac can use `HOOKE_ISAAC_COLOR_PIPELINE=source_display` to retain the source sky. Each mission supplies overview, landing-zone, follow and collection cameras.

```bash
python -m surface.publish ../temp/surface_missions/my-lunar-run
```

The publisher checks mission success, the declared time limit, frame continuity and complete video decoding. It writes media and a summary under ignored `temp/surface_missions/`, where the gallery finds them. A fresh clone can generate its own media by running a mission; the page handles missing saved media explicitly.

## Repeatable qualification

Run both worlds and both engines into a fresh evidence directory:

```bash
HOOKE_RENDER_WIDTH=1280 HOOKE_RENDER_HEIGHT=720 HOOKE_RENDER_FPS=1 \
HOOKE_ISAAC_COLOR_PIPELINE=source_display \
python -m surface.qualify \
  --output ../temp/surface_missions/campaign --gpu 6

python -m surface.publish \
  ../temp/surface_missions/campaign/lunar-mujoco \
  ../temp/surface_missions/campaign/lunar-mujoco-no-action \
  ../temp/surface_missions/campaign/martian-mujoco \
  ../temp/surface_missions/campaign/martian-mujoco-no-action \
  ../temp/surface_missions/campaign/lunar-isaac \
  ../temp/surface_missions/campaign/lunar-isaac-no-action \
  ../temp/surface_missions/campaign/martian-isaac \
  ../temp/surface_missions/campaign/martian-isaac-no-action
```

The qualification command reuses one Isaac worker, checks each complete mission, then runs a control with the same duration and unchanged reset commands. Its exit status requires all selected experts and controls to pass. The publisher also compares the exact initial scene and compiled model values between engines. MP4 and WebM recordings preserve every sampled frame; the page uses the browser-supported format and synchronizes the route with recorded rover positions. This campaign uses seed 0 and deterministic layouts.

## Local qualification

The 2026-09-17 deterministic, seed-0 campaign completed collection and return in all four world/engine combinations:

| World | Engine | Simulation time (s) | Actual travel (m) |
| --- | --- | ---: | ---: |
| Moon | MuJoCo | 65.20 | 10.81 |
| Moon | Isaac / PhysX | 65.78 | 10.81 |
| Mars | MuJoCo | 65.18 | 10.79 |
| Mars | Isaac / PhysX | 66.27 | 10.84 |

Each expert passed all ten contact, storage, travel and return checks. Four duration-matched no-action controls held their recorded reset commands and produced zero false successes. Both worlds passed exact initial-input comparisons across 412 compiled fields and the source XML. Media, trajectories and qualification summaries remain in ignored local storage.

## Evidence and scope

Success requires reaching the sampling area, at least 9 m of travel, bilateral gripper contact, at least 10 cm of sample lift, release into the bin, stable bin contact, return within 20 cm, and a stopped rover. Reading success never advances the mission. A `--mode no_action` run holds reset controls and must fail this predicate; use the expert's duration with `--seconds` for the comparison.

Gravity uses the existing lunar and Martian environment profiles. Vacuum and atmospheric pressure are declared boundaries; gas flow, dust and deformable soil are not simulated. The navigation strip and landing zone are generated scene layout, not measured terrain or simulated earthworks. Navigation uses observed state, not camera perception.

The embedded LROC / HiRISE DEM patches retain their original horizontal scale, relative elevations, source spacing and attribution. Their elevation origin is translated into the local world; a generated transition joins the patch to the landscape. A denser display grid does not improve the source measurement resolution. Per-scene provenance is cached alongside the generated XML.

Sample and background-rock geometry reuse the attributed NASA JSC Apollo asset already imported through SRB. On Mars this is an illustrative analog, not a measured Martian specimen. Existing asset notices still apply. Mass, inertia and friction are illustrative. Background boulders are visual scenery; this version does not validate obstacle avoidance. Native wheel cylinders use PhysX convex approximations. Cross-engine physics and pixels are not qualified as equivalent.

Layouts and specimen placement are deterministic. Initial qualification covers these scenes, not randomized terrain or arbitrary robot substitutions. The [G1 cooperation demo](humanoid-cooperation.md) adds walking and contact authorization; humanoid rock grasp and quadruped manipulation are not implemented.

## Landscape appearance

The surface renderer uses separate lunar and Martian palettes, generated 2048-pixel mineral/grain textures, angular gravel, and a distant landscape joined to the original map border. Lunar scenery retains a black sky and sharp sunlit shadows. Martian scenery uses ochre soil, a pale dusty sky and distant erosional remnants, guided by NASA's [natural-colour Falbreen panorama](https://www.jpl.nasa.gov/images/pia26644-nasas-perseverance-rover-at-falbreen/). The lunar reference is NASA's [surface photograph gallery](https://science.nasa.gov/moon/image-galleries/views-from-the-lunar-surface/).

These textures and distant landforms are generated illustrations. No new third-party photographs are redistributed as textures. Existing DEM and Apollo asset attribution remains applicable. The 80 × 80 m collision map, prepared access strip, gravity, robot masses, friction and task controller remain unchanged. Distant scenery and gravel have no collision. The distant visual landscape is not a navigable terrain extension. A smooth visual mesh follows the collision heightfield's original float samples; the collision-only heightfield is hidden from rendering.

Appearance generation lives in `surface.appearance`, independent of task control. Both renderers use the revised cameras. Isaac retains the source sun's direction and colour, uses the archived camera clipping range and actual render aspect ratio, and hides geometry groups excluded by MuJoCo's default renderer. Its first capture refreshes non-colliding display meshes to avoid stale terrain appearance after initialization, preserving authored visibility and physical state. MuJoCo centres its finite directional shadow view on the scene's reference centre, without modifying the model or recorded state. Illumination remains an illustrative rendering model; atmospheric scattering and measured lunar reflectance are not qualified.

To refresh a successful saved surface episode without rerunning its physics:

```bash
MUJOCO_GL=egl HOOKE_RENDER_FPS=1 HOOKE_RENDER_WIDTH=1280 HOOKE_RENDER_HEIGHT=720 \
HOOKE_ISAAC_COLOR_PIPELINE=source_display \
python -m surface.replay_render /path/to/qualified-episode \
  --refresh-appearance --gpu 6 --output ../temp/surface_missions/new-appearance
```

Appearance refresh accepts either backend. It rejects changed gravity, solver options, mass, inertia, joints, actuators and collision geometry before rendering. Original results and trajectories remain unchanged. The derived result retains the original physics result hash, the appearance comparison, reconstructed-state errors and a declaration that replay advances no physics. Refreshed media validate rendering of the previously qualified episode; they are not a new dynamics campaign or a claim of pixel equivalence.
