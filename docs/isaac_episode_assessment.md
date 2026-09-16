# 双后端回合判据与多种子回归

## 两种结果分别保存

`backends.run` 继续在回合结束时调用一次原 `task.check()`，保存 `source_check`、`source_success` 和原状态码。新增 `assessment` 使用 `hooke-manipulation-v2` 判据，在每次 `Manager.step()` 完成物理和仪器系统更新后采样。读取报告不会更新历史，也不会再次调用原判据。

这次改变验收逻辑，没有改变资产、控制器、动力学参数或专家默认动作。旧 seed-0 结果保留；新版通过率不能作为旧版后端性能提升的证据。当前仅覆盖下面五个目录任务，其余任务的新版 `success` 为 `null`，表示尚未审查。所有结果仍为 `parity_qualified=false`。

## 判据与解释范围

| 任务 | v2 判据 | 限制 |
| --- | --- | --- |
| `pipette` | 管内径向偏差 <6.5 mm；尖端在管底上方；实际拇指关节 >0.70 rad 时浸入液面 >5 mm，随后在液面下释放到 <0.45 rad，最后离开液面 >50 mm | 液面使用容器局部坐标与实际法向；离开管内后释放不计分。只证明动作顺序，没有实现吸入体积/流量模型 |
| `close_fume_hood` | 最终窗位置距目标 <20 mm；夹爪与把手接触至少 50 ms；接触期间沿关闭方向累计净位移至少 50 mm | 仅重力关窗不通过；正反抖动不会累计有效位移。接触不能单独证明完整的受力因果关系 |
| `vortex_mixer` | 试管抬起 >30 mm；与实际角速度 ≥1 rad/s 的平台连续接触 ≥500 ms；归还至 `origin1` 30 mm 内；最终无机器人接触且平台停止 | 不证明化学混合均匀度；默认专家 `gear=0` 未改动。无机器人接触是释放的代理指标 |
| `insert_centrifuge_5430`、`composite_insert_centrifuge_5430` | 保持原高度 0.955–0.961 m、目标距离 <5 mm，报告各项误差 | 几何判据，未新增姿态、释放或机械稳定性验收；不通过放宽公差刷成功率 |

新增接触时间/位移等门限是本项目的初始工程验收定义，尚未经实机标定。`success` 与 `success_within_time_limit` 分开保存；超时完成不能作为声明时限内成功。`scientific_process_validated=false` 表示尚无科学过程验收。

## 空动作对照

`--mode no_action` 保持 reset 后控制量，继续推进真实物理与仪器系统。不是把控制量清零，也不是暂停仿真。默认对照 2 秒，仅用于检查短时被动运动导致的假成功，不能替代完整回合时长的对照。

```bash
# 在源码 Hooke/ 目录，使用已配置依赖的 Python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m backends.matrix \
  --task close_fume_hood --seeds 0 1 2 3 4 5 6 7 8 9 \
  --mode expert --mode no_action --control-seconds 2 \
  --gpu 6 --wall-seconds 600 --output ../temp/backend_parity/fume_multiseed
```

默认运行 MuJoCo 和 Isaac，可用 `--backend mujoco` 或 `--backend isaac` 单独运行。每个回合使用独立进程；Isaac 仍遵守 GPU 空闲检查和独占锁。生成 `report.json`、`report.csv`、`report.md`，并保存每个回合的日志、轨迹和 `result.json`。

- `manifest.json` 保存任务、种子、时限、模式、Git HEAD、Python 版本，以及 Python/XML/共享库/JSON 输入文件的 SHA-256 指纹；这不是全部纹理/网格资产的重新哈希验收。
- 新运行拒绝覆盖非空目录。追加 `--resume` 仅恢复相同参数和输入指纹的运行；已完成失败回合也保留，不自动重试以改变分母。中断未完成的证据会先归档。
- `WALL_TIMEOUT`、`PROCESS_ERROR`、`ERROR`、`TIME_LIMIT` 单列，仍占请求分母。默认单回合实际时间上限 600 秒；涡旋完整 Isaac 回合此前需约 70 分钟，需显式增加预算，例如 `--wall-seconds 7200`。
- 缺失或崩溃回合不算双后端一致；共同失败也可能产生“一致”，不能当作成功。恒真/恒假原判据单独计数，不计原判据有效一致性。
- 跨种子统计仅代表实际请求的任务集合；不能外推为全部 165 个目录项或所有 22 个专家任务。
- 返回码 1 表示运行异常、专家原/新判据失败，或空动作被新版误判成功；判据失败与引擎崩溃应查看报告区分。

## Isaac 启动配置

独立 worker 关闭扩展文件热重载监听，避免在共享服务器上消耗大量 inotify 配额。该开关见 [NVIDIA Kit 扩展文档](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.2.0/guide/extensions_advanced.html)。默认设置 8 个 Carb tasking 工作线程；可通过 `HOOKE_ISAAC_TASK_THREADS` 覆盖，`0` 恢复 Kit 自动选择。线程设置也进入续跑指纹检查。

本机测试中，关闭监听消除了文件监视资源不足日志，但没有单独消除全部启动异常；限制线程后 3 次启动探针均成功。两项改动不能据此宣称已定位所有底层崩溃原因。后续完整回合重复启动结果应与探针分别报告。没有修改驱动、系统配额或其他用户进程。

## 本轮实测

运行结果另行归档；旧数据见 [seed-0 报告](isaac_regression_results.md)。
