# 显微实验平台 demo

**Current tasks:** pushing, grasp/transfer and volume-controlled injection now
use micrometre-scale simulated cells, with phase contrast and fluorescence.
The dose-controlled injection target is 0.5 pL. See
[cell manipulation](cell_manipulation.md) for the current operations, mechanics
and limits. Bead and 100 nL results below are historical calibration evidence.

The cell workstation now offers **Phase contrast** and **Fluorescence** views.
See [cell imaging](microscopy_imaging.md) for the current model and scale;
historical validation and recordings below describe their original versions.

一套原创参数化倒置显微工作站，包含三套电动 XYZ 微操手、电动双指微夹爪、实心微针、玻璃毛细注射针、压力控制器、开孔电动 XY 载物台、透明样本腔、电动调焦、聚光器和下置物镜。镜体外观参考 [Nikon Ti2-E 官方产品照片](https://www.microscope.healthcare.nikon.com/products/inverted-microscopes/eclipse-ti2-series)，补齐双目头、目镜调节环、渐变镜体、照明支架、聚光器、物镜、调焦旋钮与侧置相机。它是独立编写的参考重建模型，**非官方 CAD，也非经过尺寸核验的 Ti2-E 复制品**；未导入商用整机网格、品牌标识或官方照片纹理。

默认参数化场景保留上述参考模型。当前局域网服务使用 TE2000-S 独立尺寸／照片参考镜体和许可明确的 MicroManipulatorStepper 真实电动并联 CAD，参考载物台、针座和夹爪附件为独立建模；无需私有品牌 CAD。准备与启动见[并联机构](microscopy_parallel.md)。

此前校准版本的两后端各五项任务曾在实际网页通过，共十条流程、98 项判据；
这些录像与验收记录保留为历史证据。当前细胞操作见上面的更新说明。参考载物台 X/Y 操作范围各 ±4 mm，
调焦 ±1 mm。整机一比一和实机性能未验收，量化结果及使用边界见
[当前指标](microscopy_goal.md)。历史 `cad` 配置保留 openFrame／Zaber／
SmarAct 集成，私有资产准备及未确认的品牌用途许可见[CAD 集成](microscopy_cad_integration.md)。

细胞任务的左吸持管／holder 与水平夹角为 30°，右注射针／holder
为 35°。注射针尖外径 1.2 µm、内径 0.5 µm，带微米级细颈及渐变
拉制段；三维、合成显微轮廓与分段流阻共用米制参数。新版实时画面
和两后端录像使用 `parallel_v4_production_v1/` 中的当前记录。细胞任务另有
按官方尺寸图独立重建的 40× 物镜外形与下置物镜视角，仍非官方 CAD
或实机光学标定，详见[物镜装配](microscopy_optical_assembly.md)。

细胞注射工具、流程、开源控制参考与后续实现方案见 [细胞微注射调研](cell_microinjection.md)。`microscopy_injection` 现复用贴壁细胞的 pL 剂量控制，另保留 `microscopy_cell_injection` 与双侧悬浮细胞 `microscopy_suction_injection`，见 [双侧吸持注射](suction_microinjection.md)。当前公开任务均使用细胞；切换场景会重建模型。

## 运行网页

最新 TE2000-S 参考配置的双环境控制入口：`http://10.5.174.93:8084/microscopy?experiment=suction_injection`。页面可选择 Isaac，支持实际手动控制和对应环境的双格式录像。部署、会话同步、尺寸和播放格式见[双环境网页说明](microscopy_live_backends.md)。8082／8083 历史单环境服务已关闭，相应试验记录仍保留在忽略目录。

从仓库根目录执行；GPU 和端口按本机可用资源选择：

```bash
cd Hooke
MUJOCO_GL=egl HOOKE_MICROSCOPY_GPU=3 HOOKE_ISAAC_GPU=6 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m webui.server --host 0.0.0.0 --port 8084
```

打开 `http://服务器局域网地址:8084/microscopy`。当前部署地址为 `http://10.5.174.93:8084/microscopy`；此通用示例启动默认参数化模型，当前真实并联机构使用[网页部署说明](microscopy_live_backends.md)中的完整环境变量。

选择实验，点击“运行实验”。流程先自动对焦，再完成相应操作。全景、仪器近景与显微图像均随实际状态更新。页面还支持微操手各轴位移、双爪开合、载物台 XY、调焦 Z、注液压力和目标体积的手动控制。手动操作只在请求期间推进指定的模拟时间；没有后台实时时钟，录像可连续回放。

网页可选择 MuJoCo 或原生 Isaac／PhysX，使用独立工作进程拥有模拟状态与渲染器。实验执行期间可读取图像与状态，其他控制请求返回忙碌；控制输入先整体校验再写入。切换后端会关闭原会话，再启动所选环境；原生启动与场景转换需要等待，不会自动回退。重置会重新建立对应种子的场景，以同步静态细胞几何和关节初值。

## 历史微珠与开放腔校准实验

| 实验 | 控制与功能 | 成功判据 |
| --- | --- | --- |
| `microscopy_push` | 根据橙色微珠的图像位置调整推针方向，接触推动后退针 | 自动对焦、实际接触累计 ≥50 ms、最终位置误差 <350 µm |
| `microscopy_pick_place` | 蓝色微珠定位、闭爪、接触夹持抬升、搬运、落下、开爪释放 | 双侧实际接触累计 ≥50 ms、接触期间抬升 ≥800 µm、最终位置误差 <350 µm、已释放 |
| `microscopy_injection` | 毛细针定位到开放微腔、施压、按目标体积停止转移、关压退针 | 注液 100 nL，误差 <0.05 nL，压力关闭，体积账本守恒 |

全部定位机构为电动滑动关节，夹爪开合也是电动关节。微珠使用自由刚体和实际接触，无夹持焊接、附着约束或成功时的对象位姿写入。控制器观察颜色标记显微图像与轴编码器；对象真值用于成像与评估。

## 贴壁细胞 phantom 注射

局域网页面：`http://10.5.174.93:8084/microscopy?experiment=cell_injection`。也可在现有页面选择“贴壁细胞模拟注射”。切换实验会关闭原渲染上下文并加载对应工作站，避免把微珠模型的 nL 控制套用到细胞模型的 pL 控制。

该场景保留倒置镜、电动 XY 载物台、调焦和电动 XYZ 注射微操。针具为原创空心锥形玻璃管，尖端内半径 0.25 µm、外半径 0.60 µm；显微视场 160 µm，768 × 768，像素间距约 0.208 µm。这些是模型参数，不是已标定的实机精度。

中心目标是半轴 18/14/4 µm、完整尺寸 36 × 28 × 8 µm 的贴壁 phantom，背景细胞仅作视觉上下文。蓝色合成核标记用于图像定位，控制器使用已声明的名义尺寸和高度规划进针；绿色来自胞内示踪液账本。实际器械编码器、针尖位置和模拟时间驱动膜响应及流量。形变和示踪在合成显微图像中显示，三维近景保留名义细胞几何。`paper_scale_v1` 之前的记录保留旧 14 µm 厚度，不作为新尺寸的验证。样本类别、针尖尺寸、视野与原始论文依据见[论文与尺度](microscopy_paper_references.md)。

流程为：图像自动对焦 → 目标定位 → 针尖细接近 → 膜接触与压凹 → 阈值穿刺 → 0.5 pL 胞质注液 → 受控退针 → 示踪及剂量检查。针座倾角是固定配置，尚未提供倾角电动调整或完整工具热插拔。

膜模型为标量黏弹松弛和阈值穿刺，接触反作用力施加到针具，载物台承受相反的 XY 力；Z 反力由固定支撑承担。液体采用线性锥形圆管的层流流阻、入口压力的一阶响应和理想剂量关闭阀。供液、胞内注入量与环境流出量分别记账。未穿刺或针尖在外部时胞内转移为零；堵塞时可以有上游压力但没有转移。

```bash
cd Hooke
MUJOCO_GL=egl ../.venv/bin/python -m microscopy.demo \
  --operation cell_injection --gpu 7 --seed 0 --fps 8 \
  --output ../temp/microscopy_demo
../.venv/bin/python -m backends.run --task microscopy_cell_injection \
  --backend isaac --gpu 9 --seed 0 --output ../temp/cell_microinjection/native
```

种子 0 结果：MuJoCo 7.037 s 模拟时间、0.5 pL、最大名义进针深度约 3.001 µm、穿刺前最大反力约 30.05 nN；原生 Isaac 7.038 s、7,038 次 PhysX 步进、0.5 pL、最大深度约 2.930 µm。两者全部 12 项判据通过。原生 PhysX 运行器械关节，共同简化模型计算细胞力和液体；没有原生软体膜求解。结果保持 `parity_qualified: false`，科学流程标定状态为 false。

7 项细胞测试包含三个随机布局的完整流程，以及细胞外施压、未穿刺施压、退针后的穿刺状态不能解锁注射、过深与核接触、堵针、体积单位和整批输入校验。原始结果、浏览器截图和日志在 `temp/cell_microinjection/`；过程录像与帧在 `temp/microscopy_demo/cell_injection/`。

本版本未模拟膜网格、细胞存活、膜修复、真实生物反应、液体 CFD、真实相衬或荧光光学。自动对焦是已声明的合成离焦响应，剂量依赖声明的简化阀模型，没有压力／体积实机标定。当前目标验收与模型边界见[目标报告](microscopy_goal.md)。

## 双侧吸持注射

入口：`http://10.5.174.93:8084/microscopy?experiment=suction_injection`。左右分别为独立电动空心吸持管与注射针，配合负压吸持、膜穿刺、0.5 pL 注射、退针和解除吸持。两后端均通过 19 项检查；实际浏览器与失败对照通过。目标为密度匹配的悬浮球形 phantom，细胞平移使用明确标注的简化受力模型，不是原生软体细胞解算。操作、参数与边界见 [双侧吸持注射](suction_microinjection.md)。

## 资产、截图与录像

```bash
cd Hooke
MUJOCO_GL=egl ../.venv/bin/python -m microscopy.demo \
  --gpu 7 --seed 0 --fps 8 --output ../temp/microscopy_demo
```

导出内容：

- `workstation.xml`：无需外部网格或纹理文件的完整 MJCF 资产，可用 MuJoCo 加载。
- `initial/overview.png`、`initial/closeup.png`、`initial/microscope.png`：真实初始渲染。
- `push/`、`pick_place/`、`injection/`：每项实验的视频 `demo.mp4`、过程关键帧、最终状态与判据。
- `summary.json`：执行结果。网页回放读取这些本地视频；全新部署需先生成录像。

原始截图、录像、下载资产与完整日志均写入已忽略的 `temp/`。

## 共用任务入口

加入双侧吸持后，CAD 页面的五项完整实验已通过真实浏览器操作；堵针后上游压力升至约 4,751 Pa、胞内剂量仍为 0 pL，复位后注射通过。SGP 单侧手动开合、CAD 轴输入行程、实验切换、独立吸持轴、吸持管堵塞及五段录像 Range 均通过；脚本错误为 0。最新曲面法线/逐对象离焦版本的截图在 `temp/microscopy_demo/cad_integration/browser_optical_depth/`，录像在 `recordings_depth/`。手动离焦 6 µm 后自动对焦恢复至参考焦平面，实际浏览器截图的目标区域锐度提高约 17 倍；该指标验证合成图像响应，不代表实机光学精度。

五项任务也接入场景目录和后端执行器。例如：

```bash
cd Hooke
../.venv/bin/python -m backends.run --task microscopy_pick_place \
  --backend mujoco --no-render --output ../temp/microscopy_demo/check_mujoco
HOOKE_RENDER_FPS=2 ../.venv/bin/python -m backends.run \
  --task microscopy_pick_place --backend isaac --gpu 9 \
  --output ../temp/microscopy_demo/check_isaac
```

Isaac 使用原生 PhysX 的关节、接触和刚体状态驱动同一个控制器；显微图像由共同成像模块从各自的实际状态生成。原生接触偏移设为 1 µm，速度求解迭代 16 次。显微网页可直接选择 Isaac；也可通过共用后端页或上述命令启动。页面明确区分三维原生渲染和合成显微成像，后者不是物镜光路的物理仿真。

## 验证与模型边界

2026-09-18 验证：MuJoCo 三项任务在种子 0–4 的 15 次运行中全部通过。网页三项自动实验、自动对焦、微针轴移动、双爪开合、载物台移动以及偏离微腔时阻止注液均通过实际浏览器操作；浏览器脚本错误为 0。

Isaac Sim 4.5 的三项原生任务在种子 0 下也全部通过，夹取与注液同时完成原生渲染。未修改本机 535 驱动。外观更新后的共同工作站包含 22 个 body、783 个几何体、21 个内联网格、14 个电动驱动、2 个自由微珠；独立导出的 MJCF 加载成功。新外观的三项 MuJoCo 实验与录像重新生成并全部通过。

新外观还重新运行了原生 Isaac 注液：10.21 s 模拟时间、10,210 次实际 PhysX 步进、100 nL，任务通过并输出两路原生图像。证据位于 `temp/microscopy_demo/native_appearance/`。新网页三项实验重新检查通过，截图位于 `temp/microscopy_demo/browser/appearance_*.png`；11 项显微测试再次通过。

| 种子 0 的结果 | MuJoCo | 原生 Isaac / PhysX |
| --- | --- | --- |
| 推移流程 | 通过，9.59 s 模拟时间 | 通过，16.23 s 模拟时间；最终目标误差约 88 µm |
| 夹取放置 | 通过，8.07 s；双侧接触 3.053 s，抬升约 1.888 mm | 通过，8.07 s；双侧接触 3.055 s，抬升约 1.892 mm，放置误差约 49 µm |
| 注液 | 通过，10.21 s；100 nL | 通过，10.21 s；100 nL |

这些是特定 demo 的功能验证，原生结果保持 `parity_qualified: false`，不表示两引擎接触力、性能、像素或任意微尺度任务等价。首次未加微小自由旋转阻尼的 Isaac 推移未通过，原始失败证据保存在 `native_push/`；最终通过版本在 `native_push_damped/`，其余原生结果在 `native_pick_place/`、`native_injection/`。

11 项显微功能测试覆盖真实接触、双侧夹持、对焦及图像定位、载物台成像关系、体积守恒、无动作与错误对位、输入整体校验和 API 边界。相关既有任务评估 9 项、表面任务 9 项和后端适配 17 项测试也通过，合计 46 项。最终网页验证还确认并发控制返回 409、实验期间状态仍可读取、三段录像支持浏览器 Range 请求。

边界：

- 采用直径 **1.2 mm** 的玻璃密度标定微珠，长度为米、质量为千克、时间为秒；轴命令以 µm 显示，不表示真实设备达到 µm 或 nm 定位精度。微小自由旋转阻尼为未标定 demo 参数。
- 显微图像为 14 mm 视场、512 × 512 像素，像素标定约 27.34 µm/px，使用颜色标记和简化 Gaussian 离焦响应；不是光学 PSF、相衬或荧光模型，也未模拟随样本高度变化的焦平面。
- 微珠操作为干式刚体接触；没有黏附、液体拖曳、细胞形变、损伤和膜穿刺模型。载物台、微腔壁与工具尖端参与碰撞；整机支架和装饰件没有全面碰撞验收。
- 注液复用守恒体积账本和已填充圆管的层流模型，圆管等效内半径 20 µm、长度 40 mm、水样黏度 1 mPa·s；不是实际锥形针内流 CFD、自由液面、细胞注射或压力实机标定。
- 当前整机为倒置镜。体视、正置和 SEM 配置，以及取得许可的品牌整机 CAD、工具热插拔和显微子场景的完整科学模型仍需后续扩展。

源代码按场景、机构与液体状态、成像、任务控制、证据导出和网页工作进程分离，均位于 `Hooke/microscopy/`；后端沿用已有适配层。

历史 openFrame 网页配置使用 `assembled` 装配，包含真实镜架与安装件、尺寸参考相机／物镜／电动调焦及原创八实体聚光照明。此前照明／相机修正配置在两后端及实际浏览器各完成五项任务，五条实际原生器械轨迹表面间距检查通过。`paper_scale_v1` 将贴壁细胞厚度改为 8 µm，使用独立的新结果：MuJoCo 五项任务通过，原生 Isaac 两项细胞任务通过。原 8082 网页控制运行 MuJoCo，原生 Isaac 独立执行 PhysX 并保留 RGB；当时页面提供私有 `recordings_paper_scale_v1/` 五段录像及运行状态读取的样本尺寸、针尖尺寸和视野。各阶段证据、实际事件、成像与一比一边界见 [照明／相机修正阶段](microscopy_cad_integration.md#照明与相机夹具内部配合修正阶段)、[光学装配](microscopy_optical_assembly.md)、[持续 goal](microscopy_goal.md)。

## 商用倒置镜参考外观

当前[双环境局域网页面](http://10.5.174.93:8084/microscopy?experiment=suction_injection)采用独立 TE2000-S 镜体、MicroManipulatorStepper 真实电动并联 CAD、140 mm 参考针座及跟随实际位姿的双通道气管；样本保持真实微米量级，网页可手动控制原生 Isaac，当前指标见[目标验收](microscopy_goal.md)。历史 `te2000_reference_v4` 使用 Zaber CAD，两后端与实际浏览器各完成五项任务，原记录保留在私有目录，不能当作当前模型的截图。镜体有厂家尺寸和论文实拍依据，未取得官方整机 CAD，细部和一比一没有验证；尺寸与边界见[商用参考场景](microscopy_commercial_reference.md)。openFrame 内部装配验收不移作 TE2000-S 验收。

相同输入的光学卷积已实现有界复用，稳定层重复渲染约提速 2.9 倍；首次与输入变化时的开销单列。先前相机／缓存阶段的数据保留原目录与配置，详见 [相机与缓存阶段](microscopy_cad_integration.md#相机安装与光学缓存阶段)。
