# 双引擎共享控制与移液体积模型

本页说明共享实现和验证边界。逐回合指标另行保存；上一轮基线见 [流程验收报告](isaac_process_manipulation.md)。

## 两臂协作

`control_streams.run_control_streams` 接收多个动作生成器。生成器每次产生一个执行器编号到目标值的映射，空映射表示保持当前目标。调度器合并目标后只推进一次物理仿真，使两臂使用同一个时钟。

一个执行器在一次调度中只属于一个动作流；跨时刻交替写入也会报错。所有目标必须有限且编号有效，验证完成后才写入控制数组。异常退出时关闭动作流。

涡旋任务利用该调度器重叠执行抓取与调档、停机与归位。两臂依然通过执行器和真实接触操纵试管、旋钮及开关。放置和归位使用当前测得的抓取变换；归位只对齐试管轴，保留旋转对称试管自由的轴向角度。

抬高后的转移使用关节轨迹，避免笛卡尔路径在腕部奇异位形附近产生过大的关节绕行。新增 `Topp.joint_traj` 使用分段恒加速度参数化；原有笛卡尔轨迹继续使用原来的样条参数化。两臂的关节速度和加速度限制仍为 0.8，任务声明时限仍为 30 秒。

控制流程要求实际开关与请求档位到位、试管与旋转平台连续接触达到混匀时长，以及最终开关回到关闭位置。独立动作验收读取实际仿真状态；实际混匀效果与物理等价性需要另外验证。

## 移液体积

体积统一使用立方米：`200 µL = 200e-9 m³`。三个层次的职责分别是：

| 模块 | 职责 |
| --- | --- |
| `liquid_transfer` | 不依赖引擎的体积状态、容量限制、守恒记账和理想活塞模型 |
| `pipetting.PipetteTransferSystem` | 读取实际按钮关节行程和容器几何，判断浸入状态及排液目的地 |
| `backends.volume_assessment` | 独立报告目标吸液量、守恒、容量及源容器液面对应的体积 |

按钮按下比例增大时排液，比例减小时吸液。仅当吸液时吸头实际处于容器内部且低于液面，才从该容器转移体积。空中释放记录为空气行程；空中排液进入环境储量。转移量受供液量及接收容量限制。错误不会消耗尚未成功处理的按钮行程。

吸头容量取自场景中的 `tip_200ul` 标称容量。现有专家场景包含一个源容器，因此整机回合验证的是吸液并抬起。共享系统可通过 `destinations` 配置多个接收容器；向空接收容器排液已通过几何集成测试，完整机器人双容器转移配方仍待实现。

`ContainerSystem.set_volume` 更新体积和液面位置，不额外推进物理时间或液面动力学。液体日志同时记录是否存在液体及体积，两个回放渲染器据此显示空容器。任务控制、液面显示和体积报告使用相同状态。

## 验证解释

动作验收使用 `hooke-manipulation-v3`；理想体积验收使用 `hooke-ideal-volume-v1`，在网页和矩阵中分别显示。空动作对照应在两类验收中均失败。矩阵保留错误、超时和失败种子，未记录回合仍属于计划分母。

体积验收要求吸头终量距标称容量不超过 5%、最大总体积误差及源液面体积误差不超过 `1e-12 m³`、储量满足容量边界、没有液体排入环境。这些是理想模型的记账条件，不能作为真实移液器精度指标。模型未模拟压力、气体压缩性、气泡、滴落、残液或流量标定。

源任务的涡旋 `check()` 仍是恒假占位；移液旧谓词具有历史流程语义。因此独立验收通过的专家回合仍可能报告 `TASK_FAILED`，结果保留两种结论。复合离心机目录入口的当前回归范围仍是插入步骤，后续关盖、锁盖、启动和安全结束需单独验收。

`parity_qualified=false` 和 `scientific_process_validated=false` 继续保留。相同源资产和配方、成功动作以及理想体积守恒，都不等于两种引擎的动力学与科学过程已经等价。

## 运行

从 `Hooke/` 源码目录使用已配置的 Python：

```bash
python -m unittest discover -s ../tests -v
python -m backends.matrix --task vortex_mixer --backend mujoco \
  --mode expert --mode no_action --control-seconds 30 \
  --output ../temp/backend_parity/vortex_shared_recheck
python -m backends.matrix --task pipette --backend isaac \
  --mode expert --wall-seconds 600 \
  --output ../temp/backend_parity/pipette_volume_recheck
```

每次复测使用新的输出目录。矩阵会记录代码、资产及 Isaac 环境选项指纹；恢复运行仅接受相同输入和参数。`HOOKE_ISAAC_SIMULATION_THREADS` 是可选的原生线程诊断设置，默认仍为 1，不代表已取得任务加速效果。

保存后的证据可通过独立命令核对，无需启动 Isaac：

```bash
python -m backends.evidence source /path/to/mujoco/source /path/to/isaac/source
python -m backends.evidence trajectory /path/to/baseline/trajectory.npz /path/to/new/trajectory.npz
```

源快照核对全部数值字段、数据类型、名称、导出 XML、资产清单及实际复制文件的 SHA-256；轨迹核对字段完整性和数组。差异使命令以非零状态退出。证据一致仍不构成物理等价性认证。
