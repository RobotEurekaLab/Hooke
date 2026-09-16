# MuJoCo / Isaac 场景复用与验证

## 当前可以使用的入口

- 双后端运行界面：`http://<server-ip>:8080/backends`
- 实际截图、视频对比：`http://<server-ip>:8081/`
- 原场景选择界面保留在 `/`，新增双后端页面链接。

界面可以选择同一个目录任务和随机种子，分别打开场景、执行原控制器、停止任务，并查看主相机和腕部相机的实时帧及完成后的回放。Isaac 是独立进程中的真实 PhysX 步进，状态、接触和触觉数据反馈给原任务代码。没有使用录制的 MuJoCo 物体位姿驱动 Isaac 视频。

**165/165 项已在第 6 版默认配置下重新通过加载、100 步真实物理推进与 RGB 出图；此前遗漏自由关节的 7 项也已重新通过。完整功能、物理、视觉和性能等价尚未验收。** 全库任务控制器的执行结果单独记录，不能把加载成功、有限数值或恒真 `check()` 当作操作成功。

## 已有实测证据

| 检查 | 结果 | 证据目录（仓库根目录下） |
| --- | --- | --- |
| 原目录编译、资产完整导出 | 165/165 | `temp/backend_parity/catalog/export_results.json` |
| 原文件与归档资产逐文件 SHA-256 核对 | 5,056 处资产引用、495 个编译快照文件，全部一致 | `temp/backend_parity/asset_integrity.json` |
| 全部场景加载、100 步物理推进、RGB，第 6 版 | 165/165；最大实际刚体位置与源 FK 差约 0.54 μm | `temp/backend_parity/catalog_native_v6_final/batch_results.json` |
| 图片完整性 | 165 个场景、189 路相机，均有非空 640×480 RGB | `temp/backend_parity/catalog_native_v6_final/render_audit.json` |
| 原任务重置状态与归档一致性 | 22/22；位置、速度、控制和约束开关一致 | `temp/backend_parity/reset_archive_audit.json` |
| HPLC 原任务，MuJoCo seed 0 | 任务判据通过，4,622 步，9.244 秒仿真 | `temp/backend_parity/ui_mujoco_hplc/result.json` |
| HPLC 原任务，经网页启动 Isaac seed 0 | 任务判据通过，4,622 步，9.244 秒仿真 | `temp/backend_parity/ui-v6-isaac.json` |
| 网页任务创建、真实图片路由、重复运行拦截、停止进程组 | 通过 | `temp/backend_parity/ui-api-check.json` |
| 全部 22 个非展示专家，MuJoCo/Isaac | 两边均完成 22 项：18 项原判据真、4 项失败；其中 15 项同时满足原声明时限 | `temp/backend_parity/final_summary.json` |

HPLC 两边均在原声明的 15 秒内完成，判据包含物体位移和实际夹爪接触。Isaac 网页回合中，PhysX 连杆位姿与返回关节状态计算的原模型正运动学，最大位置差约 `5.16e-07 m`、旋转差约 `7.47e-07 rad`。这是坐标映射一致性的证据，不是两套引擎轨迹相同的证据。夹爪受力后的连杆姿态、接触模型与画面仍有差异。

全库加载测试每项只推进 0.2 秒，不能证明长时间稳定、原专家成功、螺纹旋合精度或科学测量准确。143 项本来就是展示任务，146 项源 `check()` 恒为真。新运行器会把展示完成、任务成功、失败和异常分开；归一化评分不足 1 不再被转换成布尔真而误报成功。

## 实现方式

- `baseline.py` / `export.py`：保存编译模型 NPZ/MJB、重置状态、名称映射和可独立重新编译的 XML，复制全部引用资产并记录 SHA-256。修复 MjSpec 序列化遗漏网格惯量模式、空默认类重名、frame 展平后几何顺序变化等问题。
- `usd_scene.py`：从编译后的原数据生成 USD，保留网格、惯量、关节轴和参考位置、限制、相机、sites、碰撞位掩码及排除关系。
- `isaac_runtime.py`：原生 PhysX 多关节系统、自由刚体、关节驱动、固定腱力分配、armature、接触报告、渲染和场景清理。
- `isaac_worker.py` / `worker_client.py`：通过权限为 0600 的本地 Unix socket 通信，隔离 Python 3.12/MuJoCo 与 Python 3.10/Isaac。目标 GPU 繁忙时拒绝另起实例；同一 GPU 使用进程锁。网页对连续 5 分钟无状态进度的任务超时退出，持续推进的慢速 SDF 任务由仿真时长限制约束。
- `closed_loop.py`：原控制器和任务系统继续使用原模型名称、IK、正运动学和任务逻辑；只有 PhysX 积分位置、速度。替换源接触列表时清除无效的 MuJoCo 求解器索引，避免错误接触力及原生崩溃。
- `source_forces.py`：计算未被原生驱动覆盖的被动力，如固定腱弹性和原 detent 插件。只计算运动学和被动力，不再每步额外进行一次 MuJoCo 碰撞求解。
- `run.py` / `validate_tasks.py`：统一运行入口、原序列化器、时间限制、进程异常检测和逐任务报告。
- 日志中标记实际 `simulation_backend` 和 `physics_engine`。原日志格式中的位置、速度、控制可复用；MuJoCo 的 `qacc_warmstart` 字段不代表 PhysX 求解器内部状态，不能据此声称支持跨引擎的精确断点续算。
- `webui/backend_api.py`：异步运行和停止、真实进程状态检测、有限的图片/结果路由。不会把整个仓库作为静态目录公开。

### 特殊场景

- Free joint：先按固定连接关系精确合并质量、质心与惯量，再生成浮动刚体，避免遗漏零质量的关节外层节点。转换必须核对全部源关节数，质心速度与原坐标原点速度也要正确换算。
- Vortex mixer：两个不同锚点的同体转轴转换成串联关节，并保留反向耦合与速度电机。辅助连杆每个增加 `1e-6 kg` 和 `1e-9 kg·m²`，应纳入后续误差验收。
- Thread SDF：保留原网格，在 GPU PhysX 中使用 256 分辨率的网格 SDF。它与原解析螺纹插件不是完全相同的碰撞函数，仍需实际旋合测试。
- Detent：复用原插件被动力计算，向 PhysX 施加力，不用 MuJoCo 积分。
- 仪器锁扣：预先建立不改变主系统自由度数量的外部锁定约束，根据原任务的 `eq_active` 启闭；需在完整开盖、关盖回合中持续验证。
- Touch sensor：原按钮的区域与射线判定使用实际 PhysX 接触法向力，避免读取另一次 MuJoCo 碰撞预测产生的按键状态。

## 深度检查中修复的漏项

1. 零质量自由关节外层节点：按固定连接关系精确合并子节点质量、质心和惯量，保留原外层坐标系与全部自由度。没有为这些外层节点添加虚构质量。旧的 7 项通过记录已撤销，并已由第 6 版全目录复测覆盖。
2. 显式接触对：原试管螺纹和部分抓取碰撞体的 contype/conaffinity 都为零，通过 XML `<pair>` 单独启用。现在这些几何体会建立碰撞，并覆盖普通位掩码过滤。
3. 锁扣事件：只修改发生变化的约束，避免重建未变化的夹爪 mimic 关系导致机器人关节状态与连杆位姿不同步。原先热循环仪“判据通过”但 FK 偏差 0.372 m 的结果已撤销。
4. 直线 mimic：PhysX 4.5 对单自由度直线关节同样要求 `rotX/rotY/rotZ` API 实例名；`transX` 会被忽略。实际双夹指从 8.4 mm 打开至 29.7 mm 的联动已验证。连续旋转 mimic 使用官方建议的极大有限范围（±1e10 度），避免联动被解析器拒绝。
5. 新增运行门禁：源关节数量必须全部进入 PhysX；实际连杆位姿与源 FK 相差超过 5 mm 时立即报错，不能继续返回任务成功。

对应实测：`native_v3_diagnostics/free-fall-result.json` 中试管整体质心的自由下落偏差约 0.43 mm，且没有外部接触；`thermal_constraint_update/result.json` 中热循环仪关盖原判据通过，21.052 秒仿真，最大 FK 位置差约 5.88e-7 m。均位于 `temp/backend_parity/`。

## 当前接触与驱动诊断

### 第 6 版修正与实测

- 旧接触缓冲区漏写 `geom1/geom2` 的问题已修复；第 5 版完整回归中，试管与试剂瓶抓取均通过原判据。
- 闭环 connect 的柔顺性用原生 D6 线性力驱动近似，保留原连接锚点、参考逆质量及 solref/solimp。独立悬挂质量测试的下沉量误差为 1.2 μm。USD 的 `low > high` 表示锁定，不能用来表示自由轴。
- 热混匀仪在柔性连接和接触参数使用源低残差阻抗时，通过原始专家：16.336 s，设定结果为 600 RPM、33°C、30 s；源基线为 16.188 s。证据：`native_v6_regression_r3/thermal_mixer/result.json`。33°C 是设定值，原任务没有在该回合模拟升温到 33°C。
- 微孔板按钮的旧显式摩擦在零速附近振荡，滑块在抓取前已滑到底。对无执行器、无弹簧的标量关节，现在用原生限力零速驱动实现摩擦，黏性阻尼另外保留。源 noslip 开启时消除爬行；关闭时按源参考逆质量与阻尼保留正则化爬行。未覆盖的摩擦拓扑仍保留显式近似。
- 摩擦实测：源 noslip=2 时 2 s 静止，原生位移约 0.18 μm；超过摩擦上限后，两者均滑动，0.2 s 后位移差约 0.13 mm。165 项原目录均使用 noslip=2。额外 noslip=0 诊断中，源模型本身会爬行，原生近似 2 s 后偏差 0.58 mm，未通过 0.1 mm 门限；该配置仍未验收，可用 `native_contracts --include-experimental` 复现，不能声称所有摩擦模型已等价。
- 微孔板单任务第 6 版通过原判据，9.018 s；复合微孔板也已通过，9.060 s。当前回归目录：`native_v6_regression_r3/`。六项回归（两类微孔板、热混匀仪、HPLC、试管抓取、试剂瓶抓取）均通过；网页默认运行已采用这组配置。

这些检查仍不构成完整物理等价验收。当前默认配置使用 `HOOKE_ISAAC_COMPLIANT_SCALE=1`、`HOOKE_ISAAC_SOFT_CONNECT=1`、`HOOKE_ISAAC_IMPEDANCE_FRACTION=0`；接触阻抗是常数近似，未实现完整非线性残差依赖。

`native_v4_tasks/pickup_centrifuge_tube/result.json` 是修复双指联动后的完整回合：试管提起至 0.92963 m（原 MuJoCo 为 0.92945 m），旧接触适配器漏写了独立的 `geom1/geom2` 兼容字段，导致原判据看不到接触。12 项旧失败记录已撤销并重新验证。独立保持抓取实验中，真实 PhysX 同时返回了外侧凸纹和指定圆柱的接触；不能把旧失败归因于只有外侧接触。实际双后端视频已发布在对比页。

接触数据传输改为直接读取原生 Float3 分量，保留全部点和冲量；100 步诊断中原生回调从约 5.22 s 降至 0.49 s。对含螺纹的场景，默认接触范围缩小至 50 μm，避免 1 mm 接触范围提前跨越细小间隙。原正的 geom margin 仍保留。

固定腱连接的等式耦合关节已转为原生隐式驱动：在 `L = W*q + C` 的约束坐标中使用总广义力 `W*F`，保持虚功和执行器力限，避免显式计算高增益阻尼的数值振荡。`native_contracts.py` 包含真实物理的直线、转动、混合单位及固定腱测试，4 项均通过稳态联动检查；混合单位案例的初始瞬态峰值误差 0.368 mm 单独记录。

`contact_parameters.py` / `contact_probe.py` 提供可选的接触标定。它只近似匹配指定阻抗处的单接触静态压入深度，不能代表完整 solimp 或摩擦模型；默认启用经过原任务回归的低残差阻抗配置。`HOOKE_ISAAC_COMPLIANT_SCALE=0` 可关闭此近似用于诊断。圆柱凸包近似也仅用于性能诊断，默认关闭，保留解析圆柱。

## 最终复核补充

- 四元数：涡旋混匀仪的原任务把两个关键帧相加，导致两个自由关节的重置四元数长度为 2。MuJoCo 自动归一化，PhysX 拒绝非单位姿态；适配层现按相同姿态方向归一化后再提交，原资产和归档值不改。全库扫描仅这一场景受影响，见 `reset_quaternion_audit.json`。
- 检查时机：物理适配层不再自行逐步调用 `task.check()`。移液器的检查会写入吸液历史，额外调用会改变行为。其旧 Isaac 成功结果已撤销，统一运行器只在专家执行结束后按相同规则检查两个后端。其他 18 项源成功任务的检查没有这类历史写入。
- 涡旋混匀仪原 `check()` 固定返回 `False`，页面单独提示尚未实现成功判据。不能把该任务运行结束后的失败标签直接理解为仿真后端无法运行。
- 完整专家：22 项均运行结束，未发生异常或达到运行器的 120 秒仿真上限；两边原判据结果均为 18 项成功、4 项失败。逐项结果见 [回归结果](isaac_regression_results.md)。移液器修正检查调用频率后，在两边均未达成原任务判据，旧成功记录不再计入。
- 加速度缓存：替换源接触数据前同时清除 MuJoCo 的约束行计数（`nefc/ne/nf/nl`），避免空间加速度换算读取已经失效的约束内存。闭环等式加接触分配的回归测试已覆盖此路径。
- 性能诊断：螺纹接触的开销主要在 PhysX 内部求解。`HOOKE_ISAAC_TASK_THREADS=8` 可用于工作线程诊断，但实测未显著加速，未修改默认线程设置或碰撞几何来换取通过结果。

## 运行时仪器屏幕与液面

`visual_state.py` 在任务的仪器/液体更新完成后采样显示状态。两边复用原热混匀仪 UI 绘制函数；Isaac 通过更新 USD 材质纹理显示实际设定值。液面来自原 `ContainerSystem`，传给两个渲染器的椭圆薄面使用同一位置、方向和厚度。`usd_visuals.py` 只创建图形，不添加碰撞体；Isaac 出图前后还会核对位置和速度，避免绘制意外推进物理。

原液面绘制依赖 `scikit-image`，已补装 README 中要求、当前虚拟环境原先缺失的依赖。近圆形边界会触发旧椭圆拟合的复数角度异常；适配层保留原绘制路径，并在该异常下使用实数二次曲线拟合，次数记录在 `runtime_visuals.liquid_fit_fallbacks`。这不是流体精度或逐像素等价证明。

液体系统读取的空间加速度现在随实际 PhysX 速度变化更新，源 MuJoCo 只进行运动学/空间加速度换算，不进行第二次积分。关节加速度是离散速度差分估计，不宣称和另一套求解器的连续加速度完全一致。移液器最新完整回合已在实时 PhysX 状态和加速度反馈下执行 7,220 步、14.44 秒仿真。涡旋混匀仪的 27,292 步原生轨迹另外通过原液体系统的状态传播检查；这种记录回放只用于检查状态链路，不能替代完整实时液体或接触验收。

CPU 显示状态构造检查覆盖 165/165 个场景；安装版本的 USD 坐标变换检查确认液面厚度 0.2 mm、变换点误差为 0，且没有碰撞 API。相关记录在 `visual_state_audit/`。网页实际运行已分别验证移液器和涡旋混匀仪两边的 0.2 秒液面出图，不能据此证明长回合流体精度。

热混匀仪的实际截图检查发现，纹理文件更新次数不能证明屏幕可读：原转换对小平面错误地使用米制 UV，并遗漏 OpenGL 像素行与 USD 的方向差异。`texture_mapping.py` 现在保留源像素，通过 UV 转换复用网格贴图；平面按原 `texuniform` 和 `texrepeat` 区分物体尺寸与空间重复。修复不改变碰撞和物理参数。最新网页完整回合已通过：MuJoCo 16.188 秒、Isaac 16.336 秒仿真，两边均更新屏幕纹理 21 次；最后一帧腕部相机均能读到 0:30、33 / 25°C、600 RPM。实际图像与六项网页检查记录见 `live_visual_web_results.json`，对比页展示原始 RGB，光照和材质仍不同。

验证补充：19 项单元测试通过；7 项原生物理小场景测试通过（直线/转动/混合单位 mimic、固定腱、柔性 connect、静摩擦、非单位自由关节重置）。对应记录为 `native_v6_pipette_final/contracts/result.json`。

## 原项目本身的问题

`close_fume_hood` 的空动作对照 10/10 也成功：门从 0.18 m 在重力下滑落，约 0.266 秒进入成功范围，原专家直到约 12 秒才关闭夹爪。原专家 seed 0 未产生夹爪—门接触。隔离的门重力补偿实验让空动作失败，但原专家也失败；没有把该实验写回源资产。

证据在 `temp/backend_parity/close_fume_hood/no_action_10.json` 和 `temp/backend_parity/counterbalanced/`。不能用这个任务的最终门位置确认抓取迁移成功。

原专家还存在声明时限不一致：5430 离心机关盖、其复合任务、通风柜和涡旋混匀仪，在源 MuJoCo 中均超过原声明时限。验证保留 `within_declared_time_limit`，页面也会标注超时；18 项原判据通过中，15 项同时满足声明时限。

全专家基线还会记录原 MuJoCo 的失败；迁移时必须区分源任务缺陷和后端适配问题。目录描述也可能比实际实现更宽，例如部分“仪器任务”只有几何展示，没有科学过程模型。

## 尚未等价的部分

1. 接触刚度、阻尼、静摩擦、椭圆摩擦和闭环柔顺性；夹爪受力姿态需继续校准。
2. 跨种子稳定性、螺纹旋合精度；当前通过原判据的种子 0 回合不代表普遍成功。
3. 视觉：原相机位姿/FOV 已保留，但天空盒、光照、玻璃、材质和曝光尚未匹配。
4. 性能：HPLC 最新网页 Isaac 回合全流程约 73.64 秒，包括启动、IK、记录和双路 20 Hz RGB；这不是实时性能达标。GPU SDF 和跨进程状态传输还需优化。涡旋混匀仪完整专家推进 54.584 秒仿真，实际耗时约 70.6 分钟（不渲染）；原 MuJoCo 专家的仿真时长为 85.638 秒，两边控制过程也并非轨迹相同。
5. 科学模型：只有原项目已有的功能可以复用；不会从仪器外观推断称量、液体、热学等过程已经实现。
6. 更广泛 MJCF：ball、柔性体、空间腱、未知插件、未知执行器模型等不能静默丢弃，应明确拒绝或继续实现。

当前所有结果的 `parity_qualified` 均为 false。`qualification.py` 仅校验评审证据清单是否齐全，不自动证明引用文件内容正确；不能代替数值测试和任务验收。网页中的 Isaac 是可运行的实验选项。

## 复现命令

公开文档中的 `<server-ip>` 和 `<ISAAC_SIM_INSTALLATION>` 是占位符。用环境变量 `HOOKE_ISAAC_PATH` 指定 Isaac 安装目录；也可在被 Git 忽略的 `temp/isaac_local.json` 中设置 `{"isaac_path": "/your/isaacsim"}`。环境变量优先，未配置时使用 `/opt/isaacsim`。服务器实际路径和内网地址不随分支发布。

使用现有 NVIDIA 535.230.02 驱动，本次验证使用独立空闲 GPU，Isaac 4.5 安装在 `<ISAAC_SIM_INSTALLATION>`。没有修改驱动或原场景资产。

从仓库根目录：

```bash
cd Hooke
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m webui.server
```

同一工作目录中运行同一个任务：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m backends.run --backend mujoco \
  --task hplc_injector_plunger --output ../temp/backend_parity/my_mujoco_run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m backends.run --backend isaac \
  --task hplc_injector_plunger --output ../temp/backend_parity/my_isaac_run
```

使用新的输出目录。`--mode preview --seconds 2` 保持重置控制推进物理；`--no-render` 用于不生成 RGB 的控制与物理检查。`--gpu` 和 `HOOKE_ISAAC_PATH` 可配置空闲 GPU 和完整 Isaac 安装路径。

```bash
../.venv/bin/python -m backends.export --all --output ../temp/backend_parity/new_catalog
../.venv/bin/python -m backends.batch --catalog ../temp/backend_parity/new_catalog \
  --output ../temp/backend_parity/new_batch --steps 100 --render
../.venv/bin/python -m backends.validate_tasks --backend isaac \
  --output ../temp/backend_parity/new_expert_batch
```

从仓库根目录执行回归检查：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_backend*.py' -v
```

## 关键语义依据

- [PhysX 接触点定义](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_contact_pair_point.html)：法向从 shape 1 指向 shape 0，冲量除以时间步得到力。
- [MuJoCo 接触与等式语义](https://mujoco.readthedocs.io/en/3.3.0/computation/)：接触力从 geom1 指向 geom2；关节耦合以初始参考关节位置为零点。
- [MuJoCo 3.3 传感器实现](https://github.com/google-deepmind/mujoco/blob/3.3.0/src/engine/engine_sensor.c)：touch sensor 的所属刚体、接触法向力、区域射线与 cutoff 规则。
- [Isaac 4.5 仿真基础](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/simulation_fundamentals.html)。

- [PhysX mimic 关节与连续转轴的有限范围要求](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/articulations.html)。
- [MuJoCo 接触参数与静态压入关系](https://mujoco.readthedocs.io/en/3.3.0/modeling.html#solver-parameters)。
- [OpenUSD 关节限位语义](https://openusd.org/release/api/class_usd_physics_limit_a_p_i.html)：下界大于上界表示该轴锁定；柔性连接的自由轴使用无限范围。

接触标定实测：0.1 kg 和 10 kg 球的静态压入深度，源引擎均为 0.1962 mm，原生 PhysX 均为 0.2028 mm（误差 6.6 μm）。证据为 `temp/backend_parity/contact_calibration/result.json`。这只验证孤立静态接触，不代表全场景动力学等价。

贴图坐标依据：[MuJoCo 3.3 渲染器的纹理映射实现](https://github.com/google-deepmind/mujoco/blob/3.3.0/src/render/render_gl3.c)。
