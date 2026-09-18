# 显微镜与显微操作场景调研

调研日期：2026-09-17；2026-09-18 补充三类显微镜、品牌资产与电动工具要求。调研基线：`7926b4f`。本文保留调研时的能力和数值实验记录；后续原创工作站的实现与验证见[显微实验 demo](microscopy_demo.md)。

## 结论

建议新增可替换仪器和工具的显微操作工作站：机械臂装载样本，载物台定位和对焦，显微图像驱动微针或微夹爪操作，最后成像、测量和保存实验记录。保留 MuJoCo / Isaac 两个机械仿真后端，显微成像采用独立模块。

第一版使用校准片和惰性微珠，先完成装载、标定、自动对焦、视觉定位及接触推移。随后验证夹取、释放和微组装。细胞注射、液体内操作、SEM 和 AFM 的测量模型需要后续专项验证。

用户补充要求：覆盖体视、倒置、正置显微镜，查找 Olympus / Evident、Nikon、ZEISS 资产；微操机构必须电动，工具支持电动微夹爪、注射针与实心微针。具体候选、许可状态与三类装配方案见[显微资产方案](microscopy_assets.md)。手动机构只作设计参考，不作为可操作设备。

## 1. Hooke 现有能力

| 部分 | 实际状态 | 对新场景的用途 |
| --- | --- | --- |
| 光学显微镜 | `optical_microscope_display` 为展示任务；模型有底座、简化镜筒和一个滑动关节 | 可保留目录入口；外形、载物台和功能需要重建 |
| SEM / TEM / AFM | 使用同一展示任务工厂，尚无相应测量与微操流程 | 后续装载和扫描任务的入口 |
| 展示任务的成功判定 | `StaticDisplayTask.check()` 返回 `True`，专家直接结束，没有验证仪器操作 | 新任务必须独立实现动作、观测和判定 |
| UR5e 和夹爪 | 已有宏观机械臂、夹爪和碰撞资产 | 装载载玻片、培养皿支架或样本盒；精细工具应有自己的运动机构 |
| 科学观测 | 现有光谱仪采用机械条件约束的合成响应；`ScienceRecords` 区分公开观测与评估真值 | 可复用仪器校准、记录、数据来源和观测边界设计 |
| 简化流动模型 | `PressureTransferSystem` 连接储液体积账本，报告中明确没有机械质量反馈 | 不能直接作为显微流体内粒子、细胞和微针的相互作用模型 |

代码依据：[展示任务](../Hooke/archetypes/static_display.py)、[显微镜资产](../Hooke/model/instrument/optical_microscope.xml)、[仪器观测](../Hooke/experiments/spectrometer.py)、[公开记录](../Hooke/experiments/records.py)、[流动连接](../Hooke/science/systems.py)。

## 2. 可复用开源资源

以下项目提供硬件设计、控制或科学模型，各有不同用途。接入前仍需确认具体文件的来源、单位、装配关系、许可和依赖。

| 资源 | 已核查内容 | 建议角色 | 许可与边界 |
| --- | --- | --- | --- |
| [OpenFlexure Microscope](https://openflexure.org/projects/microscope/) | 倒置光路、可电动载物台、STL 和装配说明；官方提供整机交互式三维视图 | 显微镜外形、载物台和扫描/对焦流程参考 | 核查的 [v7 beta5 设计许可](https://build.openflexure.org/openflexure-microscope/v7.0.0-beta5/license.html) 为 CERN-OHL-S-2.0；[服务器软件](https://gitlab.com/openflexure/openflexure-microscope-server/-/raw/master/LICENSE) 为 GPLv3。不能套用旧版本许可；柔性机构运动需要另建模型 |
| [openUC2](https://docs.openuc2.com/Investigator/FRAME/Introduction_and_Overview/) | FRAME 文档包含 XYZ 定位、模块化光路和自动样本处理扩展；旧 UC2-GIT 仓库明确不再积极维护 | 适合组织开放式显微操作平台及可替换光学模块 | 旧 [UC2-GIT](https://github.com/openUC2/UC2-GIT/blob/master/License.md) 分别声明软件 MIT、硬件 CERN-OHL-1.2、文档 CC-BY；当前 FRAME 的具体资产需要单独核查，旧许可不自动覆盖新仓库 |
| [MicroManipulatorStepper](https://github.com/0x23/MicroManipulatorStepper) | 电动 XYZ 并联平台；v4 FreeCAD 装配和零件、STL、运动学源码、固件及 Python API；设计应用包括显微操作 | 优先评估的整机微操资产；可配置左右两个独立机构和工具安装座 | [作者修改的 MIT 文本](https://github.com/0x23/MicroManipulatorStepper/blob/main/LICENSE) 明确覆盖软件、硬件设计和文档，并要求派生内容保留通知。应原样保存，不能仅用标准 MIT 文本替代；第三方芯片手册另查 |
| [MicroManipulator](https://github.com/0x23/MicroManipulator) | 手动 XYZ 柔性平台；固定提交含 FreeCAD、OBJ 打印网格及夹持附件 CAD，不是电动版本 | 仅作结构与安装件参考；不满足当前电动微操要求 | [MIT](https://github.com/0x23/MicroManipulator/blob/main/LICENSE)；没有发现现成 URDF/MJCF/USD，柔性机构和附件的完整功能尚未验证 |
| [FilMBot](https://github.com/Jiangkun-Yu/FilMBot) | 三自由度电磁驱动薄膜柔性微操机构，提供 5 个 STL、切割图和 Arduino 固件；没有 URDF/MJCF/USD 整机模型 | 微针夹持件和机构外观候选；柔性机械手的后续研究对象 | [根许可](https://github.com/Jiangkun-Yu/FilMBot/blob/main/LICENSE) 为 MIT。作者请求论文引用；仓库还含第三方芯片手册，不整体搬入。转换外形不能视为复现薄膜和电磁动力学 |
| [FOSH Micro_Manipulator](https://github.com/FOSH-following-demand/Micro_Manipulator) | 三轴机构、针具安装件 STL；README 的部分装配步骤仍待补齐 | 直线运动机构参考和备选资产 | [GPLv3](https://github.com/FOSH-following-demand/Micro_Manipulator/blob/master/LICENSE)，复用派生内容保留相应条款；没有核查到可直接运行的双后端任务 |
| [DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2) | 官方 [光学模块](https://deeptrackai.github.io/DeepTrack2/latest/src/optical.optics.html) 提供明场、荧光、像差和显微图像生成；另有噪声模块 | 明场微珠图像和成像模型的首选评估对象 | [MIT](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/LICENSE)；不是机械仿真器。当前明场模型为相干传播模型，需要声明与实际照明的差异；还未安装或接入 Hooke |
| [microsim](https://github.com/tlambert03/microsim) | [流程文档](https://talleylambert.com/microsim/stages/) 区分样本、光学响应、采样和检测器 | 荧光/三维 PSF 模型备选 | [BSD-3-Clause](https://github.com/tlambert03/microsim/blob/main/LICENSE)；示例中的外部生物数据许可另查，不能把荧光模型直接当作通用明场成像 |
| [Pycro-Manager](https://github.com/micro-manager/pycro-manager) / [ImSwitch](https://github.com/openUC2/ImSwitch) | 前者提供 Python 显微硬件控制和采集，后者提供模块化仪器控制 | 后续实机接入适配器；参考运动、曝光和采集接口 | Pycro-Manager 为 [BSD-3-Clause](https://github.com/micro-manager/pycro-manager/blob/main/LICENSE)，其底层设备和依赖单独核查；ImSwitch 为 [GPLv3](https://github.com/openUC2/ImSwitch/blob/master/LICENSE)。优先独立接口，保留各自许可，不能据此保证整个集成自动兼容 MIT |

### 微操模型与任务研究

- [FilMBot 论文](https://arxiv.org/abs/2410.23059) 报告其路径跟踪精度约 6.3 微米。这是上游装置的实验指标，不能写成 Hooke 的已实现精度。
- [MicroPush](https://arxiv.org/abs/2602.23607) 使用二维过阻尼、接触黏滑和可选背景流模型，适合参考微流体内推移任务和评估指标。本次未定位并核查可下载源码仓库及许可，不列为已获准导入的依赖；其磁性滚动机器人也不同于固定微针机械手。
- [On a Vision-Based Manipulator Simulator](https://www.mdpi.com/2076-0825/12/2/78) 讨论摄像机反馈下的抓取、保持、释放和力测量，可参考任务结构。本次没有核查到可直接复用的开源代码许可。
- [SOFA](https://www.sofa-framework.org/about/features/) 可作为柔性机构和组织形变研究备选，其核心为 LGPL，插件单独核查。它不是当前两个后端的直接插件，第一版无需引入新的完整运行平台。

### 研究用途与发布

上述已核查的开放许可允许相应研究用途，同时仍有署名、许可保留和部分派生内容公开等要求。“academic/research”不代替遵守许可。

建议每个导入文件记录上游 URL、提交或发布版本、哈希、作者、许可、修改说明和单位变换。开源软件、硬件 CAD、论文图像、实验数据和第三方资料分别核查。CERN / GPL 派生内容保留自身条款，不能统一改标为 Hooke 的 MIT。独立进程或接口是工程组织建议，不是免除许可义务的保证。

### 商用显微操作手与 CAD 核查

以下状态是本次公开官网检索的结果。“未核查到”不表示厂家没有 CAD；可能需要按型号索取。公开下载和用于 academic/research 都不能自动证明允许转换后上传公开仓库。控制 SDK 的许可也不自动覆盖硬件模型。

| 品牌 / 型号 | 已核实的设备或资产 | CAD 与复用状态 | 对 Hooke 的建议 |
| --- | --- | --- | --- |
| [Sensapex uMp-3 / uMp-4](https://sensapex.com/products/ump-micromanipulators-2/) | 压电闭环微操；XYZ 各 20 mm；分辨率 5 nm、重复定位 100 nm；uMp-3 的第 4 轴是虚拟轴，uMp-4 有物理 D 轴 | 本次尚未核查到公开下载且许可明确的整机 CAD | 适合作为膜片钳 / 微针操纵的规格和布置参考；获取具体 CAD 与条款后再决定导入 |
| [Sutter MP-285](https://www.sutter.com/micromanipulation/mp-285) | 步进电机与蜗杆 / capstan 机构，XYZ 各 25 mm；粗档 0.2 µm/步、细档 0.04 µm/步；虚拟第 4 轴 | 官方有适配器机械图和手册；本次未核查到许可明确的整机三维资产 | 可参考微针运动和串口协议；不能标成压电驱动或 40 nm 实际精度 |
| [Thorlabs](https://www.thorlabs.com/thorproduct.cfm?partnumber=MP10) | 官网搜索索引列出 MP10 刚性支架的 STEP、SolidWorks 等支持文件；新站部分产品页需要动态加载 | 已确认部分配套件提供 CAD；未确认 Sensapex 合作电动微操整机的文件和发布条款，也未下载 MP10 CAD | 先区分支架 / 位移台与整机，逐文件核查，不能用配套件 CAD 证明整机可复用 |
| [Zaber M-LSM](https://www.zaber.com/products/micromanipulators/M-LSM/documents) | 官方明确提供左 / 右侧、平底 / 立柱安装共 4 个整机 STEP 和 SolidWorks 装配；[产品页](https://www.zaber.com/products/micromanipulators/M-LSM) 给出 XYZ 各 25 mm、分辨率优于 0.05 µm、可调探针夹持器与虚拟第 4 轴 | STEP 下载地址已定位；本次未核查到模型的研究仿真与公开再分发条款，不能标为开源资产 | 商用双侧微操外形的重点候选，明确许可后再整理关节、碰撞和质量 |
| [Narishige MHW / MMO](https://products.narishige-group.com/group1/electro/english.html) | MHW-3 是三轴水压操作手；[MMO 等注射系列](https://products.narishige-group.com/group1/injection/english.html) 有油压机构 | 产品与显微镜安装资料可查；本次未核查到许可明确的整机三维资产 | 生物注射工作站布置参考；MHW-3 不应误标为油压 |
| [TransferMan 4r](https://www.eppendorf.com/product-media/doc/en/68978/Eppendorf_Cell-Technology_Installation-guide_TransferMan-4-m-r-InjectMan-4_Micromanipulator-5191-92-93.pdf) | Eppendorf 历史官方安装资料包含微操手与显微镜的装配关系；[官方公告](https://www.eppendorf.com/hk-en/special-pages/micromanipulation-and-microinjection-portfolio-discontinued/) 说明产品线在 2023 年转至 Calibre Scientific | 本次未核查到许可明确的整机 CAD；安装手册不等于开放模型，当前获取渠道需核对承接方 | 参考双侧电动微操布局，具体型号性能另查 |
| [SEMISHARE SS-700e-m / SS-700](https://semishareprober.com/accessories/micropositioner.html) | 电动 SS-700e-m：XYZ 各 15 mm，分辨率 0.1 µm、重复定位 ±1 µm；SS-700 是不同规格的探针机构 | 本次未核查到许可明确的整机 CAD | 晶圆 / 电路探针任务候选，不直接当作细胞注射装置 |
| [RWD 瑞沃德](https://www.rwdls.com/product-solutions/solution/research-platform/modeling-programme/data_250.html) | 官网有显微注射操纵系统方案 | 本次未核查到许可明确的整机 CAD；用户列出的 MMS 型号与性能尚未逐项验证 | 国产微注射工作站参考；不据此填入未经核查的精度参数 |
| [纳腾](https://www.shnti.com/index/product/index/category_id/74.html) / [CoreMorrow 芯明天](https://www.coremorrow.com/) | SEM 原位操纵与压电定位方向的候选厂家 | 本次未核查到可直接公开复用的具体整机 CAD；LF-2000 等型号的性能需按对应资料继续核对 | 后续 SEM / 压电精密定位场景候选，暂不作为首版生物微操资产 |
| [Micromanipulator](https://micromanipulator.com/products/accessories/probe-positioners/525-2525-manipulators/) / [WPI](https://wpiinc.com/products/var-3141-joystick-controlled-manual-micromanipulator) | 前者有探针定位器，后者有显微注射配套操纵器 | 本次未核查到许可明确的整机 CAD；不能以品牌概括所有产品的精度 | 分别作为半导体探针与生物显微操纵的备选，按具体任务选型号 |

### 精度参数的记录方式

分辨率是读数或命令增量，定位精度是到达目标位置的误差，重复定位描述多次到达同一位置的离散程度，漂移描述随时间发生的位置变化。需要分别记录，注明测量位置、载荷、速度、温度和标定条件；轴编码器读数不能自动视为针尖相对样本的位置。

Sensapex 的 5 nm / 100 nm 分别是分辨率 / 重复定位指标，官网的静态零漂移说明也带有恒温和静态外载条件。Sutter 的 40 nm 是细档步长，说明步进电机装置也可能提供纳米量级命令增量。MicroManipulatorStepper 作者报告最小步进可到 50 nm，同时明确提醒绝对精度更差；本次未验证这些实机性能。

建议用任务和工具分类：光学显微镜下的微针 / 微夹爪、晶圆探针、SEM 内原位操纵。每类再声明驱动和闭环方式；不按一个分辨率数字推断整机精度或把所有机构归为压电。

## 3. 已下载资产的小规模检查

实际下载了固定 FilMBot 提交中的两个零件，放在忽略目录 `temp/microscopy_research/assets/`，没有加入产品资产目录。

| 零件 | 下载大小 | 三角面 | 合并重复顶点后的网格 | 当前结论 |
| --- | --- | --- | --- | --- |
| `Coil holder_3 legged.stl` | 287,684 B | 5,752 | 不封闭 | 可读；适合外观候选，碰撞/质量计算需另处理 |
| `needle holder V3.STL` | 138,484 B | 2,768 | 封闭 | 可读；尚需核实单位、装配和工具安装变换 |

STL 没有声明长度单位，不能仅凭数值直接标成米或毫米。两个零件的解析不表示整机已在 MuJoCo / Isaac 运行。OpenFlexure 的官方整机 GLB 地址已定位，获取时超过本次 20 MB 单文件研究上限，没有保存或导入；后续选择必要组件并离线整理。

另实际下载了固定 MicroManipulatorStepper 提交的 v4 末端 STL、整机 FreeCAD 装配文档、运动学源码和原始许可：

- `EndEffector.stl`：539,584 B，10,790 个三角面，合并顶点后封闭，原始外包围尺寸为 29 × 29 × 29；STL 本身没有单位声明。
- `Assembly_MicroManipulator.FCStd`：400,582 B，ZIP 校验与内部 XML 解析通过。装配包含外部链接，尚未在 FreeCAD 中解析全部子零件或验证完整装配，不能视为单文件自包含整机。
- `kinematic_model_delta3d.cpp`：12,310 B，可用于核对并联机构运动学；本次没有编译或对照实机验证。
- `LICENSE`：1,264 B，作者修改的 MIT 文本明确覆盖硬件。完整文本及文件哈希留在本地证据中。

Zaber 官方 STEP 的首次获取超过 20 MB 研究上限，仅保留 4,096 B 文件头检查。2026-09-18 后续完整下载 34,904,794 B，记录于 `assets/zaber-step-manifest.json`；未用 CAD 内核解析或导入关节，不能写成整机导入成功。

## 4. 微尺度对现有后端的要求

### 4.1 本机 MuJoCo 数值实验

使用 MuJoCo 3.3.0，玻璃密度 2,500 kg/m³，显式球体质量和惯量，固定时间步 0.1 ms，两接触几何使用一致的接触参数。可编译案例步进 0.5 s。模型仅包含自由小球和地面，不含黏附、流体和光学。

| 球直径 | 米 / 千克 / 秒 | 毫米 / 千克 / 秒 |
| --- | --- | --- |
| 100 µm | 编译失败：运动体质量/惯量必须大于 `mjMINVAL` | 可编译并完成步进，状态有限，无警告 |
| 200 µm | 同上 | 可编译并完成步进，状态有限，无警告 |
| 500 µm | 可编译并完成步进，状态有限，无警告 | 同左 |
| 1,000 µm | 可编译并完成步进，状态有限，无警告 | 同左 |

所有可运行案例最终接触穿透约 0.637 µm，这是选定软接触参数下的数值结果，不是真实材料验证。500 / 1,000 µm 两种单位的最终位置换回米后吻合至数值精度；100 µm 小球的质量在两种单位下均为约 `1.309e-9 kg`。

这个实验支持进一步研究一致的单位变换：毫米求解时，长度数值乘 1,000、重力加速度乘 1,000、惯量乘 1,000,000，保持千克和秒。它没有改变物体的实际尺寸或质量。力、扭矩、弹簧、阻尼、流体、接触阈值和传感器参数也必须按各自量纲转换。

不能把这些结果推广为任意微小物体都可准确模拟；阈值影响随形状、材料、自由度和单位选择变化。无需通过虚增小球质量绕过本次问题。

### 4.2 当前 Isaac 转换器需要补的部分

- `SceneBridge` 的常规 `contact_offset_m` 默认是 1 mm；含特殊几何时默认 50 µm。需要根据工具尺寸、运动速度、时间步和目标精度选择并验证；较大偏移是提前产生接触约束的范围，不等同于实际网格膨胀。
- 多关节同体转换会为部分串联辅助链接加入 `1e-6 kg` 质量和 `1e-9 kg m²` 惯量。这个逻辑不是每个自由微珠都会触发，但微操机构采用相应关节布局时需要单独处理。100 µm 玻璃球质量约 `1.309e-9 kg`，说明宏观辅助参数不能直接推广到微尺度机构。
- 当前相机转换使用透视相机公式，没有处理源相机的正交投影分支。显微视野需要独立的投影、裁剪和物理像素标定。
- 现有转换器以米制处理参数，新增其他求解单位必须覆盖完整转换链。本次没有运行 Isaac 的微尺度接触或单位实验。

代码依据：[USD 转换器](../Hooke/backends/usd_scene.py)。[Isaac 4.5 文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/simulation_fundamentals.html) 说明小而薄的物体需要考虑接触偏移和时间步；[PhysX 单位文档](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/API.html#using-different-units) 要求一致单位与相应容差；[USD 单位文档](https://openusd.org/release/api/group___usd_geom_linear_units__group.html) 表明长度单位属于 stage 级设置，不能只改一个标签而不换算已有内容。

建议公共配置与实验记录继续使用 SI 单位，后端承担求解单位适配。先验证 0.5–1 mm 惰性对象的共同任务，再验证 100–200 µm。若采用独立微尺度子场景，要显式定义样本交接、坐标变换、同步时间和反力反馈；不声明未经实现的完整双向动力学。

## 5. 成像和接触不能由外观替代

### 显微成像

仪器配置至少应包含物镜倍率和数值孔径、波长或照明谱、工作距离、样本折射率、传感器像素尺寸、曝光、图像分辨率和标定版本。

第一版可以采用标定的正交视野与简化光学响应，并明确其模型范围；再用 DeepTrack2 的适用模型比较和标定。PSF 即点扩散函数，描述一个点被成像系统扩散成多大范围。焦点移动应改变图像响应，不能通过切换预制清晰图片实现自动对焦。明场与荧光使用各自成像假设，普通三维渲染图不能直接标为科学显微图像。

图像还应包含适用的离焦、照明不均、曝光饱和、光子/读出噪声及遮挡。相机、载物台与针尖需要坐标标定，输出物理尺度尺。算法使用图像和允许公开的编码器读数；目标真实位置、私有分割和评估答案仅由渲染/评估使用。离线数据标注另设明确的数据导出模式。

[MuJoCo 相机文档](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#body-camera) 已支持正交投影；当前 Hooke 的 Isaac 转换尚需适配。正交相机只是成像几何近似，并不自动实现显微镜光学。

### 微操力学

根据样本和环境选择模型：干式操作关注表面黏附、接触摩擦及释放；液体内关注黏性阻力、浮力和适用条件下的布朗运动；跨液面或含液桥时考虑毛细力；柔性样本增加形变和损伤；微注射另需膜穿刺和压力/流量耦合。

[液体表面张力的微机器人研究综述](https://www.annualreviews.org/content/journals/10.1146/annurev-control-062422-102559) 说明毛细作用在微操作中的意义。各项模型需要参数来源和标定，不能给所有尺度统一加一个常数吸引力，也不能用自动焊接替代夹取和释放。若初期关闭黏附，任务应声明为简化干式刚体接触。

力学后端与附加模型对阻力、接触和样本状态各自只有一个更新方，避免重复施力。随机噪声与布朗过程的状态纳入复位和实验记录。

## 6. 推荐场景与推进顺序

| 阶段 | 场景与任务 | 主要资产和验收 |
| --- | --- | --- |
| A：成像工作站 | 机械臂装载校准片，固定样本，标定视野，自动对焦，扫描拼图 | 显微镜、XYZ 载物台、样本盒/载玻片、标尺片；检查装载接触、固定、像素尺度、离焦响应和记录完整性 |
| B：显微操作 | 固定平台上两侧微针/微夹爪，完成微珠推移、定位，再扩展夹取释放 | 先用 0.5–1 mm 惰性对象；检查针尖标定、图像反馈、接触力、对象移动、退出后保持和双后端结果 |
| C：微组装和样本测量 | 微丝/微零件定位，夹取放置，矿物颗粒尺寸与形状统计 | 可替换工具、样品盘和定位槽；报告位置/角度误差、夹取释放成功率、破坏率及测量偏差 |
| D：液体与柔性样本 | 微珠在液体内操作，随后研究细胞固定、穿刺和微注射 | 密闭腔、微吸管和压力系统；先标定流体/形变，再定义膜穿刺、注射体积和损伤判据 |

第一版精密运动机构建议采用可解释的 XYZ 台和微针模块。整机资产优先评估 MicroManipulatorStepper：须核对闭链约束、驱动关节、运动学、装配引用和工具接口，不能用一个固定 STL 代替可运动机构。若首版使用简化 XYZ 运动模型，应记录近似并与完整并联机构分开命名。FilMBot 零件可作为有来源的外观/夹持件候选；若采用刚性关节或降阶柔性模型，必须记录这项近似。

建议界面同时显示工作站总览和显微视野，提供载物台/工具状态、焦点扫描曲线、尺度尺及测量记录。局部放大不改变物理几何尺寸。

可与既有太空任务衔接：采集样本返回受控实验舱，在显微镜下做粒径和形状分析。低分辨率岩石扫描网格不提供矿物晶粒和细胞结构；需要单独的微观样本模型及来源。空间站仪器和样本固定到舱体，液体使用受控密闭腔；月球/火星外部环境与实验腔环境分别声明。设置压力边界不等于已模拟气密、沸腾或完整流体。

## 7. 解耦与验收建议

建议分别组织工作站场景、仪器运动、工具、样本物性、成像、控制、任务与评估。下面是接口职责建议，尚未创建这些接口：

- 机械适配器：共同单位的命令、关节/编码器状态、接触和力；内部适配两个后端。
- 成像适配器：接收样本及光学状态，按模拟时钟生成图像和标定元数据；可替换简化、DeepTrack2 或实机采集实现。
- 微尺度模型：明确环境和参数、力与状态的更新方、复位及随机状态。
- 控制器：依据公开图像和仪器读数完成标定、对焦和视觉反馈运动。
- 任务评估：私有真值计算成功与误差，并验证过程证据。

验收分开报告：

1. **机械**：加载/复位、单位一致性、针尖定位误差、峰值力、穿透、限位、样本交接；进行时间步收敛检查。
2. **成像**：物理像素尺度、焦点响应、噪声/饱和、图像与物理时间一致性、测量偏差。
3. **任务**：多随机种子的成功率、失败原因、完成时间、最终位置/角度误差；无图像反馈、错误焦点和关闭相应物理项的对照实验。
4. **双后端**：同一物理样本和仪器参数、同一控制协议分别独立运行，在预先声明的容差内比较；不能用轨迹回放证明两个引擎都完成操作。
5. **性能**：加载耗时、物理步进、成像耗时、端到端控制延迟与显存，注明分辨率、对象数和硬件。

调研阶段没有显微任务成功率、实机精度或 Isaac 微尺度性能结果。后续 demo 文档记录具体任务和数值验收阈值；仍不代表实机精度。

## 8. 核查版本与本地证据

| 上游 | 本次固定提交/版本 |
| --- | --- |
| FilMBot | `e137ead571c20a9c7e142d244a56c7f69dc0da66` |
| MicroManipulatorStepper | `63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94` |
| MicroManipulator | `df9b8a912928a72f13fab861bc32d4e4da580226` |
| FOSH Micro_Manipulator | `0bde80db859d0e9a10eadfc744e8726d8b6594ea` |
| UC2-GIT | `afc9de36f27d5a387e6b9ef1d85ae0cf33b4d3d6` |
| openUC2 ImSwitch | `f33e7f3a3551a4a271f95dbea63dd73ad1cfd884` |
| DeepTrack2 | `14df60c21d23f61d6cc5c20814900c9f9bdb4e1f` |
| Pycro-Manager | `c9c8acd1969fb5e55619d8284b6c1fd96fc236b4` |
| microsim | `3aedf764545de5037e3b20cc8ce8d41a3e6fd7f7` |
| OpenFlexure 设计页面 | `v7.0.0-beta5`；软件许可来自调研日期访问的 master 文件，未固定软件提交 |

固定提交通过公开 Git 引用读取，许可文件实际获取并保存哈希。原始核查材料和下载零件位于忽略目录 `temp/microscopy_research/`：`sources.json`、`assets/manifest.json`、`assets/geometry-preflight.json`、`mujoco-scale-probe-v2.json`。首轮探针的地面保留了默认接触参数，第二轮对两个接触几何统一了参数；上表只使用第二轮结果。

微操资产补充证据：`assets/stepper-manifest.json`、`assets/stepper-geometry-preflight.json`、`assets/zaber-step-prefix-check.json`。Zaber STEP 只检查前缀，其哈希不是完整文件哈希。商业 CAD 的许可状态均未标为已获授权。

2026-09-18 已完整取得 SmarAct SGE-17 本体及 STD-001 夹指 STEP，文件声明毫米单位。大小、哈希、完整标记和检查范围记录在 `assets/smaract-step-preflight.json`；仍未导入整机、验证运动或确认仿真及发布权限。三类品牌显微镜与工具的最新获取状态见[显微资产方案](microscopy_assets.md)。

本次仅新增调研文档并进行局部数值/网格检查，没有修改运行时代码、安装显微控制软件或提交到 GitHub。
