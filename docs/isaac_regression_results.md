# 双后端当前回归结果

种子 0；原始资产、控制器和成功判据。22 项均执行结束，无运行异常或达到运行器的 120 秒仿真上限。
原 MuJoCo 成功的 18 项，在 Isaac 中全部通过。此结论不代表跨种子或物理、画面、性能等价。

165/165 场景通过加载、100 步真实物理推进与 RGB 检查；每场景只覆盖 0.2 秒。
18 项原判据通过中，两边各有 15 项同时满足原声明时限；5430 关盖及其复合任务、通风柜的成功回合超过原声明时限。

| 任务 | MuJoCo 原判据 | Isaac 原判据 | MuJoCo 仿真秒 | Isaac 仿真秒 |
| --- | --- | --- | ---: | ---: |
| `pickup_centrifuge_tube` | 通过 | 通过 | 9.352 | 9.354 |
| `thermal_cycler_close` | 通过 | 通过 | 21.052 | 21.052 |
| `thermal_cycler_open` | 通过 | 通过 | 17.036 | 17.028 |
| `centrifuge_5430_close_lid` | 通过 | 通过 | 16.922 | 16.932 |
| `centrifuge_5910_lid_close` | 通过 | 通过 | 22.132 | 22.132 |
| `insert_centrifuge_5430` | 失败 | 失败 | 10.328 | 10.322 |
| `thermal_mixer` | 通过 | 通过 | 16.188 | 16.336 |
| `pipette` | 失败 | 失败 | 14.440 | 14.440 |
| `vortex_mixer` | 失败 | 失败 | 85.638 | 54.584 |
| `centrifuge_mini_close_lid` | 通过 | 通过 | 10.902 | 10.902 |
| `pickup_reagent_bottle` | 通过 | 通过 | 13.546 | 13.530 |
| `close_fume_hood` | 通过 | 通过 | 18.550 | 17.162 |
| `push_filling_nozzle_down` | 通过 | 通过 | 9.292 | 9.290 |
| `composite_push_vial_nozzle` | 通过 | 通过 | 8.374 | 8.374 |
| `composite_push_hplc_plunger` | 通过 | 通过 | 9.264 | 9.264 |
| `composite_filter_cartridge_housing` | 通过 | 通过 | 8.440 | 8.438 |
| `composite_microplate_stacker_eject` | 通过 | 通过 | 9.060 | 9.060 |
| `composite_insert_centrifuge_5430` | 失败 | 失败 | 10.328 | 10.322 |
| `composite_centrifuge_5430_close_lid` | 通过 | 通过 | 16.274 | 16.278 |
| `hplc_injector_plunger` | 通过 | 通过 | 9.244 | 9.244 |
| `microplate_stacker_eject` | 通过 | 通过 | 9.018 | 9.018 |
| `filter_cartridge_housing` | 通过 | 通过 | 9.380 | 9.380 |

源通风柜判据在空动作下也成功，不能作为机器人操作有效性的证明。移液器采用与源后端相同的判据调用次数；此前额外调用改变历史状态的结果已撤销。
涡旋混匀仪的原 check 固定返回 False，任务未实现成功判据。适配层已按 MuJoCo 的行为归一化非单位重置四元数；原始资产和控制器未修改。

涡旋混匀仪的源专家已超过原声明的 30 秒时限；本次采用 120 秒仿真上限以验证完整执行。
逐项机器可读记录：`temp/backend_parity/final_summary.json`。完整实现和限制见 [迁移说明](isaac_scene_parity.md)。
