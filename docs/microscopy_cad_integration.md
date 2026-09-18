# 真机 CAD 显微工作站

2026-09-18。本文保留历史品牌 CAD 集成方案及其许可边界。当前主网页已改用许可明确的真实并联 CAD，见[并联机构](microscopy_parallel.md)与[目标验收](microscopy_goal.md)。整机一比一、全工作空间和生物物理标定没有验证。

## 真实部件与原创附件

| 模块 | 已采用的真实几何 | 运行配置与边界 |
| --- | --- | --- |
| openFrame | 固定提交 `19c312931f4fdcbfaf3725fbef3db075a32413c6`，35 份 STEP；当前装配 9 个不同部件 | 5 层镜体高 241 mm；层间 5 mm 公燕尾进入前层。立柱安装孔间距 10 × 40 mm，从 STEP 圆柱面提取核验。相机、光学附件、电动载物台及升高件原创 |
| Zaber M-LSM RHF | 原厂右手平底 STEP，184 个实体 | 原厂名称映射固定板和三组滑台；CAD Y 为竖直轴。移动件居中，25 mm 行程，OEM 外形尺寸不缩放。丝杠转动、滚动轴承及实机驱动精度未标定 |
| SmarAct SGP-17F | 官方参考配置 STEP，56 个实体 | 一个真实平行滑台，一侧移动另一侧固定。保留原厂夹指，加装原创精细接触延长指；演示力限制 5 mN，不是厂家力传感器模拟 |
| 针具与支架 | 原创米制参数化几何 | 单细胞针内半径 0.25 µm、外半径 0.60 µm，水平夹角 35°；吸持管水平夹角 30°。拉制针细颈与后端管身分段建模；悬臂、针夹连接与延长指为布局估算，没有实机刚度或精度证明 |

[openFrame 上游](https://github.com/ImperialCollegeLondon/openFrame)、[Zaber 文件入口](https://www.zaber.com/products/micromanipulators/M-LSM/documents)、[SGP-17F 上游](https://www.smaract.com/en/micro-grippers/product/sgp-17f)。CAD 由 OpenCascade 检查拓扑并按 0.04 mm 容差转换；运行加载校验源文件及逐网格 SHA-256。网格容差不足以验证亚微米尺寸，更不代表设备定位误差。

openFrame 核心设计的 CERN-OHL-P-2.0、上游声明和衍生修改记录保留在私有资产目录。两家品牌 CAD 的本地转换、研究仿真、图像与公开再分发权限未逐项确认；不将其网格纳入 Git。本地原型与几何验证不构成授权证明，具体声明见 [资产许可核查](microscopy_assets.md)。未联系厂商、提交申请表或购买模型。

## 可选运行配置

默认 `HOOKE_MICROSCOPY_ASSETS=reference` 使用无需外部文件的原创参考模型。显式选择 `cad` 时使用本机私有衍生资产；缺文件、错哈希或未知配置会在加载前报错。

```bash
cd Hooke
HOOKE_MICROSCOPY_ASSETS=cad MUJOCO_GL=egl \
  HOOKE_MICROSCOPY_GPU=8 HOOKE_ISAAC_GPU=9 \
  HOOKE_MICROSCOPY_MEDIA_ROOT=/绝对路径/temp/microscopy_demo/cad_integration/recordings_final \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m webui.server --host 0.0.0.0 --port 8082
```

默认私有资产位置为 `temp/microscopy_research/cad_meshes/`，可用 `HOOKE_MICROSCOPY_ASSET_ROOT` 更改。该环境已安装 NumPy、SciPy、trimesh、MuJoCo；CAD 内核 `cadquery-ocp` 和干涉检查 `python-fcl` 仅用于离线研究环境。新的机器必须先取得和转换对应资产，Git 克隆本身不包含品牌 CAD。

贴壁细胞模型只建立注射微操手；双侧悬浮细胞模型按需建立吸持与注射两侧机构，均不加载微夹爪 CAD。器械运动、接触代理、细胞力学、流量、成像和界面分开；厂商网格为外观，不自动启用壳体接触。

## 装配核验

早期 95 mm 工具伸出配置检出了 CAD 支架与载物台相交，以及相邻微操手相交。修正为原创外置支架和更长工具连接后，又检出连接杆触及照明臂与载物台；改用探针折线路径和更高夹爪安装。失败证据保留，不覆盖。

离线工具读取 **MuJoCo 编译后的网格与实际世界姿态**，保留网格重心变换，用 FCL 检查分组表面相交：

```bash
HOOKE_MICROSCOPY_ASSETS=cad PYTHONPATH=Hooke .venv/bin/python \
  -m microscopy.assembly_audit export --output temp/microscopy_clearance
PYTHONPATH=Hooke temp/microscopy_research/cad_env/bin/python \
  -m microscopy.assembly_audit check --output temp/microscopy_clearance
```

修正后的初始姿态覆盖探针 192、夹爪侧 248、注射侧 190、仪器侧 178 个几何体，6 组比较均无表面相交，无跳过几何。该检查排除任务接触尖端与样本；**尚未验证包含关系、同一总成配合、全行程及任务扫掠空间**。

进一步记录显微任务 MuJoCo 种子 0 的每个实际物理步。先用保守包围盒排除远离零件，再用“参考表面距离 − 两零件最大平移量”证明距离下界；整段过宽的候选按每个物理步核验。每个刚性滑台只存一条位置轨迹，其他网格使用固定偏移；旋转机构会明确拒绝此方法。解析压缩数组只做一次。

| 完整任务 | 实际记录步数 | 表面间距保守证明 |
| --- | --- | --- |
| 推移 | 9,590 | 通过 |
| SGP 夹取放置 | 8,070 | 通过；最小距离下界约 0.297 mm |
| 开放腔注液 | 10,210 | 通过 |
| 贴壁胞质注射 | 7,037 | 通过 |
| 双侧吸持注射 | 9,901 | 通过 |

采用额外 0.1 mm 几何余量；重叠的包围盒不会自动判为碰撞，无法证明的候选也不会自动通过。证据在 `clearance/tasks/` 和 `clearance/pick-every-step/`。这是指定 MuJoCo 轨迹的器械 **网格表面间距** 证明，不包含体内包裹、同一总成装配配合、碰撞尖端或任意全行程。原生轨迹另按以下归档方法核验。三项离线距离回归包含会撞障碍的反例及刚体代表位置偏移核验；另三项原生归档测试拒绝改动模型、缺失物理步和不匹配的任务。

openFrame 原柱夹旧 PDF 标 44 mm，与固定版本 STEP 的 40 mm 孔距不同；运行依据当前 CAD 孔中心，不声称旧图纸完全一致。相机型 openFrame 没有加虚构双目头。

## 证据与后续

`temp/microscopy_demo/cad_integration/` 保留各阶段视频、浏览器证据、源模型、原生 USD、进程日志及失败记录。当前 SGP、连接杆避让、单针与双侧吸持配置的五项任务在两个引擎均通过：

| 任务 | MuJoCo 模拟时间 | 原生 Isaac 模拟时间 / 实际 PhysX 步进 |
| --- | --- | --- |
| 推移 | 9.590 s | 16.230 s / 16,230 |
| SGP 夹取放置 | 8.070 s | 8.070 s / 8,070 |
| 定量注液 | 10.210 s | 10.210 s / 10,210 |
| 贴壁胞质注射 | 7.037 s | 7.038 s / 7,038 |
| 双侧吸持注射 | 9.901 s | 9.901 s / 9,901 |

上一轮四项原生记录在 `native_verified/`，最新五项原生记录在 `native_suction/`，1 FPS 采样输出实际 RGB，任务逐项实际事件数与步进数一致。最新精简汇总为 `suction_verification_summary.json`；上一阶段汇总保留为 `verification_summary.json`。连续运行中因编辑后进程缓存旧函数造成的细胞加载错误另留在 `native_parallel_gripper/`，新进程完整重跑四项通过；未将加载失败当作科学模型通过证据。

6 项机构测试覆盖旋转轴定位与虚功、Zaber 零件随动、膜反力投影、配置完整性、SGP 移动/固定归属及真实接触夹取释放。11 项原有显微、7 项贴壁细胞、6 项 CAD 机构、7 项吸持、3 项原生归档、3 项离线距离证明和 1 项 STEP 曲面法线测试均通过，共 38 项。相关后端适配 17 项及配置 8 项通过，合计 63 项；可选研究依赖按独立环境运行，跳过项不计作通过。

最新实际浏览器独立完成五项自动实验，SGP 单侧手动控制、±12,500 µm 行程、堵针、独立吸持轴、吸持管堵塞及五段视频 Range 回放通过，脚本错误为 0。证据在 `browser_suction/`，录像在 `recordings_final/`。双侧操作和解析模型边界见 [吸持注射](suction_microinjection.md)。

最新五项原生任务已按各自归档的模型与实际 PhysX 关节轨迹核验器械表面间距。核验源模型 SHA-256、连续时间戳、物理事件数与轨迹长度后，只执行源运动学；不积分 MuJoCo。每对表面另计入原生位置误差与姿态误差造成的最大点位移。证据在 `latest_native_clearance/`；先前轨迹的证明保留在 `native_clearance/`。

后续包括光学附件与工作距离核验、最终 CAD/原生配置的吸持失效对照及渲染外观修正。独立 `cad_meshes_surface_normals/` 保存曲面/UV 法线版本，保留面边界以避免硬边平滑；厂商来源与刚体参数核验通过，原生每帧 16 次采样的预览明显改善了低采样时的表面失真；该修正版五项 MuJoCo、五项原生与实际浏览器流程均通过，详见下节；源码默认 CAD 根目录保留，局域网服务显式选择法线目录。曲面法线方法参考 [OpenCascade 官方文档](https://dev.opencascade.org/doc/occt-7.9.0/refman/html/class_b_rep_lib___tool_triangulated_shape.html)。可用 `HOOKE_ISAAC_SAMPLES_PER_FRAME=1..64` 与 `HOOKE_ISAAC_DENOISER=0/1` 单独控制路径追踪采样/去噪；默认仍为 1 与 1。该参数含义见 [Isaac Sim 4.5 官方 API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)。保持 `parity_qualified: false` 和 `scientific_process_validated: false`，不以任务成功率证明实机或生物准确性。

## 曲面法线与逐对象离焦回归

后续冷启动已完成上述其余任务：最新五项原生流程在 `native_optical_depth/`，全部 49 项任务检查通过，每项实际事件数与报告步数一致；逐步器械间距另在 `optical_depth_native_clearance/` 全部通过。更新的 MuJoCo 五段录像在 `recordings_depth/`，实际浏览器五项任务及手动离焦/自动对焦检查在 `browser_optical_depth/`，脚本错误为 0。精简汇总是 `optical_depth_verification_summary.json`。局域网服务已显式加载独立法线目录与新版录像，源码的默认根目录没有更改。

逐对象离焦四项新增测试，以及原生部分剂量颜色的二进制往返回归通过；相关测试总计 68 项（显微/模型/CAD/距离 43、后端 25）。一次离线脚本误用了既有结果目录，任务在原生加载前被拒绝；原始 stdout 恢复该次索引，错误输出及恢复说明单独保留，原始关节轨迹和 RGB 未改动。不能以恢复索引替代当前冷启动的五项新流程证据。

真实光学安装件、Teledyne 相机 CAD 和 Nikon/PI 图纸已进一步取得与检查，仍未装配进工作站，见 [光学装配核验](microscopy_optical_assembly.md)。三维细胞偏灰、光学附件估算和生物模型未标定仍是明确的剩余问题。

部分剂量颜色的通信修复随后通过五类原生 CAD 失败对照，全部实际 PhysX 步进与 RGB 完成，事件数与观察轨迹一致。对应五类 MuJoCo CAD 对照也通过。对照范围与被保留的失败证据见 [双侧对照说明](suction_microinjection.md#最终-cad-失败对照与颜色修复)；其真值定位不能当作自动视觉控制或生物成功率。

## 相机安装与光学缓存阶段

`HOOKE_MICROSCOPY_OPTICS=mechanical` 按真实安装面与三孔对应接入 openFrame 的 SM2 夹具及 C-mount 转接件，镜体累计装配 11 个不同 CAD 部件。默认 `estimated` 继续保留旧布局供对照。厂商 Teledyne 模型包含尚未确认的复制/使用权限，当前相机是独立尺寸参考模型；来源与测量边界见 [光学装配](microscopy_optical_assembly.md)。

相机装配后的五项 MuJoCo 演示、五项原生流程、五项实际浏览器实验及五条实际原生轨迹的器械表面间距证明均通过。原生结果来自三个验证进程：高采样进程保存细胞注射、吸持与推珠三个成功任务；剩余夹取和注液由稀疏采样进程完成；有界光学缓存改动后另验证吸持任务。最初夹取的中断记录不计通过。细胞注射和推珠保留每仿真秒 20 次采样，其余当前验收路径使用每仿真秒 1 次；这些是采样设置，不是实时性能指标。

最新精简汇总为私有 `camera_mount_verification_summary.json`，逐任务事件数与步数一致，所有间距检查为零跳过。源录像在 `recordings_camera_mount/`，缓存版两项源演示在 `recordings_cached_cells/`，原生轨迹与 RGB 目录在汇总中逐项列出。先前失败对照保留其原配置和来源，不冒充此次相机配置的新对照。

本阶段 29 项测试通过；跨阶段相关测试合计 77 项，分别在应用与可选 CAD 环境核验，不宣称是同一个依赖环境的一次完整测试运行。光学缓存的逐像素等价、首次开销与稳定层速度见 [成像缓存](cell_microinjection.md#8-完全相同光学层的有界复用)。这一阶段完成相机机械连接与缓存；后续装配进展见下节，Goal 保持 active。

## 物镜与电动调焦装配阶段

最新局域网配置为 `HOOKE_MICROSCOPY_OPTICS=assembled`，使用同一私有曲面法线资产根目录。镜架包含 13 个未缩放的 openFrame 原始部件、1 个有明确修改记录的 M25 名义螺纹包络衍生盘；独立尺寸参考物镜和电动调焦滑台分别经真实 STEP 转换，滑台的固定底板与移动端是两个实体。移动端、物镜座、转接盘、原创延长件和物镜跟随实际 focus 关节；相机保持固定。三类样本的物理焦点参考与合成成像中的编码器符号一致，全部伺服关节的质量、惯量与控制参数保持不变。结构依据、估算部位与 0.16 mm 工作距离定义差异见 [光学装配](microscopy_optical_assembly.md#已接入的电动调焦与物镜装配)。

| 任务 | MuJoCo 仿真时长（s） | 原生 PhysX 实际步数 | 每个声明相机的原生 RGB 帧数 |
| --- | ---: | ---: | ---: |
| 推珠 | 9.590 | 16,230 | 17 |
| 电动夹取放置 | 8.070 | 8,070 | 9 |
| 常规注射 | 10.210 | 10,210 | 11 |
| 贴壁细胞注射 | 7.037 | 7,038 | 8 |
| 双侧吸持注射 | 9.901 | 9,901 | 10 |

五项源流程及五项原生流程分别通过 49 项任务检查。原生五项来自一个新进程，显式每仿真秒采样 1 帧、每帧路径追踪采样 16 次；步数等于本任务实际物理事件数，初始化的两个事件单列。微珠任务声明两个相机，细胞任务声明三个相机，完整原生 RGB 数量与对应运行记录一致。这里的帧数不是实时性能或两个后端数值等价指标，`parity_qualified` 与 `scientific_process_validated` 仍为 false。

五条原生关节轨迹经归档哈希校验及纯正运动学重建，用 FCL 在每个记录步的平移边界内检查器械表面间距，全部通过、零跳过。此检查排除任务接触尖端与样本，并不证明同一总成配合、包含关系或任意全行程。另对贴壁细胞参考高度的镜架／调焦／物镜／解析延长件在 −1 mm、零位、+1 mm 三处做精确 CAD 公共体积检查，无超过 `1e-5 mm³` 阈值的交叠；这是三个静态位置的检查。

实际浏览器完成手动调焦与自动对焦、独立吸持轴、双通道堵塞设置、单侧移动夹指、轴限位和五项完整任务，脚本错误为零、五段录像的 Range 请求返回 206。新录像位于 `recordings_focus_assembly/`；独立原生证据位于 `native_focus_assembly/`，网页截图位于 `browser_focus_assembly/`，精简汇总为 `focus_assembly_verification_summary.json`，均在忽略目录。本阶段 9 项相关测试通过，跨阶段独立相关测试共 81 项；旧阶段及失败证据保留其原配置和来源。

网页显微操作由 MuJoCo 工作进程执行；此验收没有实现实时 Isaac 显微控制。照明、聚光器、载物台细节、管镜和连续检测光路仍需补齐，独立相机不是厂商 CAD，厂商微操手／夹爪的本地复用授权也仍未确认。整机不宣称一比一或完整生物物理标定，Goal 继续 active。

## 照明与相机夹具内部配合修正阶段

`assembled` 最新配置使用 14 种未缩放 openFrame 原始部件、1 个 M25 包络衍生盘及独立尺寸参考相机、物镜和电动调焦。照明新增真实 Cairn 规格夹具，并按有限孔轴修正横杆支撑方向；原创八实体聚光照明由 STEP 生成，八个网格封闭。它不是厂商 MonoLED 或品牌整机复刻，也未做生物／辐射标定。安装与首次 UV 极点三角形转换失败的边界见 [光学装配](microscopy_optical_assembly.md#照明支架与原创聚光器)。

扩大内部装配检查后发现早期 SM2 夹具穿插 TL、CORE、FL-MOT 三层，原功能与器械之间的间距检查没有覆盖这些内部配合。最新配置加入三个原创 6.5 mm 空心隔柱，夹具安装面移到 X=75 mm，转接筒与相机同步外移；镜架接口实际有限插入 8 mm、夹具重叠 22 mm、相机前安装面 X=105.65 mm。原始 STEP 未改动。新检查共 32 个实体、838 次配对，无超过 `1e-5 mm³` 阈值的交叠；仅包含列出的光学部件和三个调焦位置，不证明连续全行程、螺旋牙型、公差、XY 台或微操总成内部配合。

| 任务 | MuJoCo 仿真时长（s） | 原生 PhysX 实际步数 | 每个声明相机的原生 RGB 帧数 |
| --- | ---: | ---: | ---: |
| 推珠 | 9.590 | 16,230 | 17 |
| 电动夹取放置 | 8.070 | 8,070 | 9 |
| 常规注射 | 10.210 | 10,210 | 11 |
| 贴壁细胞注射 | 7.037 | 7,038 | 8 |
| 双侧吸持注射 | 9.901 | 9,901 | 10 |

两后端各通过 49 项流程检查。最新原生五项来自同一验证进程，实际事件数等于报告步数；每仿真秒 1 帧、每帧路径追踪采样 16 次，每个声明相机的 RGB 计数匹配。微珠任务两路相机、细胞任务三路相机；总计 51,449 实际步、128 张原生 RGB。帧数不是实时性能，两后端仍不宣称数值、像素或科学准确性等价。

五条实际原生轨迹经过归档哈希、步数／时间和正运动学核验，再用 FCL 检查每个记录步的器械平移边界，全部通过、零跳过。此证明排除预期任务接触尖端与样本，不是全装配连续碰撞证明。实际浏览器五项任务、手动离焦／自动对焦、独立吸持轴、双通道堵塞、单侧夹指与限位均通过，脚本错误 0，五段录像的 Range 请求返回 206；ffprobe 确认五段均为 H.264、1680 × 760、4 帧／秒。

本阶段 13 项相关独立测试通过：照明 5、相机 4、焦点装配 2、CAD 法线／有限面 2。照明在 CAD 与应用依赖环境分别运行，共享近轴测试计一次；跨阶段相关独立测试合计 86，不是一次整库测试。精简证据为私有 `illumination_assembly_v2_verification_summary.json`，新录像、原生结果、网页截图及轨迹检查分别在 `recordings_illumination_assembly_v2/`、`native_illumination_assembly_v2/`、`browser_illumination_assembly_v2/`、`illumination_assembly_v2_native_clearance/`。旧装配的功能通过与内部穿插失败各保留其原配置，不重写为新配置通过。

局域网 8082 已加载本轮源码、装配说明和录像，网页仍由 MuJoCo 控制。完整检测光路、电动 XY 台、原生 Isaac 网页手动会话、照明与显微图像的实际标定、品牌 CAD 权限和整机一比一仍未完成；Goal 继续 active。
