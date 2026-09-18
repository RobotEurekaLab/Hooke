# 显微镜、电动微操手与可换工具资产方案

核查日期：2026-09-18。这是品牌资产调研与集成设计。已建立原创参考配置和可选真机 CAD 配置，运行方法和功能边界见[显微实验 demo](microscopy_demo.md)、[CAD 集成](microscopy_cad_integration.md)。商业模型商店候选尚未购入；公开厂商 CAD 只保存在忽略目录用于本地研究。后端尺度实验见[基础调研](microscopy_research.md)。

许可明确的真实电动机构现已加入：`parallel-v4` 使用 MicroManipulatorStepper
的 173 件实际机械设计实例，与独立参考镜体、载物台和工具集成。
该配置无需私有品牌 CAD，来源、完整作者声明、获取／转换命令及当前
验收范围见[开源并联微操](microscopy_parallel.md)。它不代表品牌整机或
纳米精度已验收。

当前镜体已按 Nikon Ti2-E 官方照片独立重建外观，代码为 `Hooke/microscopy/instrument.py`，原始参考留在 `temp/microscopy_research/appearance/`。不包含官方 CAD、照片贴图或品牌标识，也未声称尺寸精确复刻。该外观不增加第三方整机资产导入数。

新增真实开源硬件配置：Imperial College London / Cairn Research 的 openFrame 核心 CAD 已在本地解析为 35 个有效实体，最新 `assembled` 配置使用 14 种未缩放原件与 1 个明确记录的 M25 包络衍生安装盘。它与 Ti2-E 参考模型是不同仪器；相机／物镜／电动调焦为独立尺寸参考，聚光照明为原创 STEP，电动 XY 台与完整检测光路仍待核验。整机一比一未验收，详见 [光学装配](microscopy_optical_assembly.md)。

## 1. 配置要求

- 覆盖体视、倒置、正置三类显微镜，优先核查 Olympus / Evident、Nikon、ZEISS。
- 微操定位机构必须电动；微夹爪的开合也必须电动。注射针、实心微针安装到电动机构，注射另配可控制的压力系统。
- 显微镜、工具、样本及固定支架组成实际可装配的工作站；型号、单位、运动限位、工作距离与针尖坐标均有依据。
- 外观资产、可运动机构和科学模型分别核查。整机网格不自动包含运动学、光学或真实设备精度。

## 2. 三类仪器的品牌参考

下表是已找到官方产品资料的配置参考，**不表示已经获得这些整机的可复用三维资产**。

| 类型 | Olympus / Evident | Nikon | ZEISS | 建议操作 |
| --- | --- | --- | --- | --- |
| 体视 | [SZX16](https://evidentscientific.com/en/products/upright/szx16) | [SMZ1270 / SMZ1270i](https://www.microscope.healthcare.nikon.com/products/stereomicroscopes-macroscopes/smz1270-smz1270i) | [Stemi 508](https://www.zeiss.com/microscopy/us/products/light-microscopes/stereo-and-zoom-microscopes/stemi-508.html) | 矿物颗粒分选、夹取、微组装；按所选物镜确认工具空间 |
| 倒置 | [IXplore IX85](https://evidentscientific.com/en/products/inverted/ixplore-ix85) | [ECLIPSE Ti2-E](https://www.microscope.healthcare.nikon.com/products/inverted-microscopes/eclipse-ti2-series) | [Axio Observer](https://www.zeiss.com/microscopy/en/products/light-microscopes/widefield-microscopes/axio-observer-for-life-science-research.html) | 透明样本腔内微珠操作，后续扩展固定与注射 |
| 正置 | [BX53](https://evidentscientific.com/en/products/upright/bx53) | [ECLIPSE Ni 系列](https://www.microscope.healthcare.nikon.com/products/upright-microscopes/eclipse-ni-series) | [Axio Imager 2](https://www.zeiss.com/microscopy/en/products/light-microscopes/widefield-microscopes/axio-imager-2-for-life-science-research.html) | 开放样本上的侧向微针操作、表面观察与测量 |

不能按系列名称推断所有配置均电动：Ti2-E 是电动型号，Ti2-A / U 是不同配置；SMZ1270i 的智能倍率读数不等于电动变焦。SZX16 官方规格包含手动变焦；若增加自动化，应选择有依据的电动附件并单独记录。SZX16 虽位于官网 `upright` 路径，其产品类型仍是体视显微镜。

## 3. 厂商 CAD 和尺寸资料

| 来源 | 已核查的获取方式 | 已获得什么 | 当前边界 |
| --- | --- | --- | --- |
| [Evident 显微组件 STEP](https://evidentscientific.com/en/insights/3d-cad-data-microscope-components) | 填写表单后，通过邮件获得下载链接；官方说明为外观模型，适合布局检查 | 已定位入口，未提交表单或获取组件 | 列出的物镜、镜筒组件不等于 SZX / IX / BX 整机；公开再分发许可未确认 |
| [Nikon OEM CAD](https://www.microscope.healthcare.nikon.com/products/oem/cad) | 表单申请，由当地 OEM 代表发送所选组件 CAD | 已定位入口，未提交表单 | 不能标为公开直下、开源整机资产 |
| [Nikon OEM 官方图册](https://downloads.microscope.healthcare.nikon.com/phase7/literature/Brochures/OEM_2CE-MUZH-9_12602T_82P.pdf) | 官方公开 PDF，包含 Ni、Ti2、SMZ 的结构、系统图和尺寸资料 | 已完整下载 15,241,827 B；PDF 可读，42 个 PDF 页面，版面含跨页图册内容 | 尺寸和配置参考，非开放三维资产；原始文件保留在忽略目录 |
| ZEISS 官方产品资料 | 产品与应用页面、具体物镜和附件资料 | 已核查三类产品；Axio Observer 官方列出微操手支持 | 本次未找到许可明确、公开下载的现代整机 CAD |

ZEISS 的 [Axio Observer 官方应用说明](https://www.zeiss.com/microscopy/en/products/light-microscopes/widefield-microscopes/axio-observer-for-life-science-research.html) 明确列出 Narishige、Eppendorf、Luigs & Neumann 微操手支持，可以参考其装配思路；具体安装适配器仍需逐型号核查。

## 4. 第三方整机模型候选

### 已取得的真实开源显微镜部件：openFrame

[openFrame 官方仓库](https://github.com/ImperialCollegeLondon/openFrame)提供真实制造使用的模块化显微镜核心部件设计。本地固定版本为 `19c312931f4fdcbfaf3725fbef3db075a32413c6`，原始设计、制造说明、装配 wiki 和衍生网格均留在 `temp/microscopy_research/`。

- 35 份 STEP 已经 OpenCascade 解析、拓扑有效性检查并转换为米制 OBJ，每份包含一个有效实体；转换容差为 0.04 mm。原厂零件名称、原始 CAD 包围盒与网格包围盒差值逐件记录。
- 官方 README 明确将该目录的核心组件设计置于 **CERN-OHL-P-2.0**。33 份 STEP 另有文件内声明，2 份较早的照明臂/立柱文件依据仓库整体声明；随附许可已读取并保留。衍生文件保留上游来源、声明和 2026-09-18 网格转换修改记录。许可适用的是这些开放核心设计，不自动覆盖另购物镜、相机和其他品牌附件。
- `OF-LL-CORE` 的 CAD 直径为 150 mm，主体高度 72 mm，底部燕尾另占 5 mm，与官方组件说明一致。`OF-LL-SP` 载物台安装板为 300 × 175 mm，含底部接口总厚度约 16 mm。
- 运行配置采用 5 层镜体、载物台柱夹、立柱、双孔夹具与照明臂；实际孔中心提取并核验安装对齐。CAD 镜体已与原创电动载物台、调焦、光学附件和真机 Zaber 机构集成，在 MuJoCo 和原生 Isaac 跑通四项任务；完整光学与实机精度未验收，不能计作品牌整机一比一。

发布衍生设计时保留声明、许可和修改说明；按上游要求在论文中引用 [openFrame 论文](https://doi.org/10.1111/jmi.13219)。许可与声明随资产清单保留，原始文件不自动纳入 Git。

### Nikon Ti2-E 尺寸参考核查

官方 OEM 图册 PDF 第 33 页、印刷页 65 的 **ECLIPSE Ti2-E/B / Ti2-E Main Body** 图纸标注：宽 240 mm、主体侧视长度 495.3 mm、前后另有 32.2 / 21.7 mm 标注，主镜体图示高度 378 mm。相邻左页的 331.5 mm 高度属于 **Ci-E**，不能当作 Ti2-E 参数。图中尺寸不包含所有独立照明附件，不能直接用整个场景的包围盒对照主镜体高度。

目前 `instrument.py` 尚未按此图完成逐部件重建；仅取得图纸不表示已达到尺寸还原。未给出尺寸的曲面、外壳内部和接口细节继续标为估算。

以下几何、格式与部件信息来自作者的商品说明，尚未下载或在 Hooke 中验证。没有购买模型。

| 模型 / 作者 | 类型与内容 | 许可 / 获取状态 | 建议 |
| --- | --- | --- | --- |
| [Olympus SZX7 / Vertexmonk_Studio](https://www.turbosquid.com/3d-models/stereo-microscope-olympus-szx7-3d-model-2175166) | 体视镜；Maya / MAX / FBX / OBJ，57,520 个多边形，作者标注厘米；单一合并网格，无贴图或展开 UV | 付费；页面标有 Editorial Uses Only | 可作体视镜外观参考；仍需拆分运动部件、补材质和尺寸核对 |
| [ZEISS Stemi 305 / kapibardbp](https://3dexport.com/de/3d-model-zeiss-stemi-305-revit-459359) | 体视镜；仅 Revit `.rfa`，有材质，无贴图、绑定和 PBR | 页面标为免费；未下载，未核查完整许可 | 免费不等于开放；转换成本与外观细节待评估，不优先用于高逼真 demo |
| [Olympus BX51M / 3d_molierInternational](https://free3d.com/3d-model/microscope-olympus-bx51m-5403.html) | 正置材料显微镜；82,409 个多边形，有贴图，OBJ / glTF / USDZ 等，无绑定 | 付费；页面标有 Editorial Only | 外观细节参考；不作为首个生物注射工作站或开放仓库资产 |
| [Olympus GX71 / 3d_molier](https://free3d.com/3d-model/inverted-metallurgical-microscope-olympus-gx71-2235.html) | 倒置金相镜；54,336 个多边形，有贴图，OBJ / glTF / USDZ 等，无绑定 | 付费；页面标有 Editorial Only | 用于金属 / 材料观察的候选；不能以倒置外形替代生物样本的光路和夹具 |
| [Nikon SMZ1270i / flugaria](https://www.turbosquid.com/FullPreview/1653733) | 体视镜；C4D / FBX / OBJ，102,804 个多边形，作者称尺寸近似真实 | 付费；页面标有 Editorial Uses Only | 整机外观候选，尺寸还需对照官方资料，复用范围需单独确定 |
| [ZEISS Axio Imager 2 / skelington](https://www.cgtrader.com/3d-models/science/laboratory/microscope-zeiss-axio-imager-2) | 正置镜；214 个独立部件，MAX / OBJ / FBX，202,348 个多边形，作者标注毫米 | 付费；CGTrader 标为 Royalty Free；同一作者的 [TurboSquid 页面](https://www.turbosquid.com/3d-models/microscope-research-3d-model-1580787) 标为 Editorial Only | 有利于拆分外观零件，但许可不能跨平台套用；未验证完整 PBR、关节或训练数据权利 |
| [Olympus IX71 倒置镜线索](https://3dwarehouse.sketchup.com/model/47eaf78b907a9e7efcb2ee3b834dbc86/Inverted-Microscope) | 搜索索引指向生物倒置镜；本次未成功打开模型详情或验证文件 | 模型页、文件和具体权限待核查；3D Warehouse 平台有再分发限制 | 保留线索，不列为已获准复用资产 |
| [通用倒置镜 / NakedSingularityStudio](https://sketchfab.com/3d-models/invert-light-microscope-low-poly-pbr-029ed9bc23c647cda29c5ef94ace3dfc) | 作者声明 PBR、可分离部件、绑定与动画；5,174 个三角面 | 原页有 NoAI 标记，销售已迁至 Fab，本次未核查新商店条款 | 低面数外观备选，不能标成指定品牌或默认用于训练数据 |

权限需要分别核查：本地研究仿真、渲染图片 / 论文展示、训练数据、模型文件公开再分发。[3D Warehouse 官方 FAQ](https://help.sketchup.com/en/3d-warehouse/3d-warehouse-terms-use-faq) 明确指出，将其模型纳入发布的研究项目需取得各开发者批准。[CGTrader 当前条款](https://www.cgtrader.com/pages/terms-and-conditions) 对再分发、嵌入产品与 No AI 许可分别设有限制。不能以 academic/research、付费或 Royalty Free 字样代替具体许可核查。

开放外观备选：Sketchfab 公共 API 确认两个通用显微镜模型可下载，标为 CC-BY-4.0：[Compound Microscope / kg007](https://sketchfab.com/3d-models/compound-microscope-8709be06c5b24549b19bba7a1988a56f) 和 [Microscope / lionclaw0612](https://sketchfab.com/3d-models/microscope-dd9e73d51edb43198bd40d0bbd2dd54e)。仅核查元数据，未取得文件；正常下载仍需账号流程，尺寸与逼真度待验证。现代品牌整机的开放许可缺口目前仍然存在。

补充核查：[SmarAct 网站版权声明](https://www.smaract.com/en/imprint-motion)限制商业复制、分发、修改及转载网站对象，并未给出已确认的 CAD 研究仿真许可。[Zaber 官方文件页](https://www.zaber.com/products/micromanipulators/M-LSM/documents)确实提供整机 STEP，但页面本身没有确认模型的开放许可。当前品牌模型的本地网格转换、研究仿真、论文图片、训练数据与再分发权利仍须分别记录，不能仅写“再分发未知”而暗示其他用途都已获准。现有本地几何原型与回归记录不构成授权证明；品牌资产继续不进入 Git。相机包的实际复制限制与独立尺寸模型替代方式见 [光学装配](microscopy_optical_assembly.md)。

2026-09-18 再次核对上述两个官方页面，仍未找到明确覆盖当前仿真用途的授权。
许可明确的替代候选 MicroManipulatorStepper 已补齐固定提交
`63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94` 下的 19 个 FreeCAD 文档，
连同原始许可共 3,766,022 B；ZIP 校验和 XML 解析通过，文档中的外部
CAD 文件名均已解析到实际下载文件。原始作者修改的 MIT 许可明确覆盖
硬件设计，完整文本与逐文件哈希保存在忽略目录
`temp/microscopy_research/stepper_linked_cad/`。这是文件获取与引用文件名
检查，不是 FreeCAD 特征求值、装配坐标验证或运动机构集成通过；完整
并联运动不能用固定网格或串联 XYZ 冒充。

保存的 340 个非空 BRep 通过 OpenCascade 拓扑检查。另有 265 个零字节
特征字段，经 XML 核对属于 `InternalShape`、`SuppressedShape` 或
`AddSubShape`；它们没有几何载荷，不当作 265 个成功解析的零件。
首次检查的误分类和修正后的 `brep-preflight-retry1.json` 均保留。

## 5. 电动微操资产与末端工具

| 模块 | 资产与实际检查 | 集成用途与限制 |
| --- | --- | --- |
| [MicroManipulatorStepper](https://github.com/0x23/MicroManipulatorStepper) | 固定提交已获得 FreeCAD 装配、末端 STL、运动学源码与硬件适用的原始许可；详情见基础调研 | 开放电动机构首选；须补齐外部引用并复现并联运动，不能把整机替换成固定网格 |
| [Zaber M-LSM](https://www.zaber.com/products/micromanipulators/M-LSM/documents) | 官方右侧完整文件 34,962,531 B，经 CAD 内核解析为 184 个有效实体 | 固定/移动归属按原厂零件名建立；3 级真实轴、25 mm 行程接入两后端。原创悬臂和工具连接另记估算；公开再分发条件待确认 |
| [SmarAct SGP-17F](https://www.smaract.com/en/micro-grippers/product/sgp-17f) | 官方 STEP 2,659,877 B，经 CAD 内核解析为 56 个有效实体；原厂尺寸图已读取 | 单个平行滑台、一侧移动另一侧固定，保留 OEM 夹指并加原创精细接触延长指。MuJoCo 真实接触夹取通过，原生回归单独记录；公开再分发条件待确认 |
| [SmarAct SGE-17](https://www.smaract.com/en/micro-grippers/product/sg-1730) | 本体 STEP 1,835,096 B，41 个有效 CAD 实体；STD-001 夹指 STEP 157,441 B | 一体式柔性夹指，不能整件刚体平移。当前为后续柔性机构研究候选，未将其开合伪装为实现 |
| [Sensapex uMp-3 / uMp-4](https://sensapex.com/products/ump-micromanipulators-2/) | 已核查压电电动设备规格，尚未取得整机 CAD | 生物微针工作站参考；真实 D 轴与虚拟轴分开建模 |
| [Sutter MP-285](https://www.sutter.com/micromanipulation/mp-285) | 电动步进机构，安装图与手册可查，整机 CAD 未确认 | 微针持架、运动和控制协议参考 |
| [SIRIL 双臂微操](https://github.com/zycrobot/SIRIL) | 固定提交 `456f33408fa36d1e28e3075de1688a48cd70dbbb`；说明有两路 DC 电机实现夹取 / 轴向旋转，FBG 力反馈 | 有价值的行为与机构研究参考；未找到项目级 LICENSE 或所示微夹爪 CAD。仓库里的 STL / URDF 属于 Phantom Omni 主控力反馈设备，不能误算成微操手资产 |

SGE-17 当前官方规格为 17 × 25.5 × 9 mm、14 g、开口小于 1 mm，额定夹持力 1 N。小于 10 nm 的夹持分辨率是设备指标，不能宣称 Hooke 已实现该精度；1 N 也不能直接作为细胞或脆弱样本的安全夹持力。真空兼容属于可选型号，不能给全部变体默认启用。产品 URL 仍使用旧 `sg-1730` 名称，应以当前页面和 STEP 文件的 **SGE-17** 为准。

三类工具应拆成独立资产与行为模块：

| 工具 | 外观与几何 | 必需功能 |
| --- | --- | --- |
| 电动微夹爪 | 微型驱动本体、左右夹指、螺钉、接线；夹指形状按样本任务选择 | 开合、限位、接触、力限制、夹取与释放；闭环传感器只在对应配置中启用 |
| 注射针 / 玻璃微吸管 | 空心玻璃管、拉制锥形尖端、针座、密封和压力管路；内径、外径与锥度独立记录 | 电动空间定位、轴向进退、压力 / 体积控制；膜穿刺与注射物理需单独验证 |
| 实心微针 / 探针 | 金属细杆、尖端和夹持座；按具体用途声明绝缘与材料 | 电动推移 / 定位、接触和力记录；电极外观不自动实现电生理功能 |

针具可按经核实的尺寸参数独立生成，而非从受限商用网格提取。玻璃注射针、保持 / 吸持管与实心探针分别配置，不能用同一圆锥替代。尺寸参考可查 [Eppendorf 历史微操图册](https://www.eppendorf.com/media/MAIN/03-Service-Support/Produktkatalog/Eppendorf-Catalog-2022_AK01005621_INT-comp.pdf) 和 [WPI 微电极资料](https://wpiinc.com/blogs/all/metal-microelectrodes-basics)；本次未取得这些针具的开放 CAD，也未验证参数化模型。

Eppendorf [官方公告](https://www.eppendorf.com/hk-en/special-pages/micromanipulation-and-microinjection-portfolio-discontinued/) 说明 TransferMan、InjectMan、FemtoJet 和相关微吸管产品在 2023 年转至 Calibre Scientific。历史资料仍可作配置参考，但获取当前资产与权限应核对承接方。手动 / 液压微操手只保留为布置参考，不作为本方案的可操作设备。

## 6. 三套工作站的装配

**体视工作站：** 样本盘 + 左右电动 XYZ 微操 + 微夹爪 / 实心微针，镜体按具体物镜的工作距离安装。照明可选择环形或侧向反射，透明样本可另配透射照明。电动对焦附件和载物台单独选择。双目目镜不等于数字双目相机：普通单摄影口只输出单幅图像，双路数字视野需要对应光路与标定。

**倒置工作站：** 物镜与观察光路位于样本下方，透明培养皿 / 样本腔固定在开孔载物台上，聚光器位于上方。两侧电动微操采用可调整角度的针座；注射配置可采用一侧吸持、另一侧注射。工具路径需避开聚光器、皿壁、支架和另一支针；玻璃底厚度、物镜校正与工作距离按具体配置核查。压力控制器、管路与样本状态共同参与实验记录。

**正置工作站：** 物镜位于样本上方，工具从侧面接近开放样本，验证与物镜外壳、转盘、支架的碰撞。用于接触操作的区域不能被盖玻片挡住。若扩展浸液生物操作，选择适合的长工作距离物镜与开放腔；不通过让针尖穿过盖玻片来解决空间问题。

工具安装接口至少记录针尖 / 夹持中心坐标、接近方向、允许角度、工具质量、伸出长度、安装变换、运动范围和标定版本。显微镜接口另记录相机接头、物镜螺纹与光路配置；C-mount、RMS、M25、M27 等不能默认通用。

安装件还有具体厂商线索：[PI P-545.SH4](https://www.japan-pi.com/en/products/sensors-components-accessories/p-545sh4-microscope-slide-and-petri-dish-holder-100000019) 支持 25 × 75 mm 载玻片、直径 35 mm 培养皿，并列出 Nikon Eclipse / Ti2 兼容配置。该页三维下载栏目却标为 **P-545.SH3** 且需申请；应先核对型号与适配器，不能把栏目存在写成 SH4 CAD 已取得。

## 7. 逼真度与实施顺序

1. **装配与外观：** 按官方尺寸和选定配置核对比例、部件、光路位置、安装孔和工具工作空间；补齐相机、管线、针座、控制器及减振固定。外观网格、碰撞几何与物性分别维护。
2. **运动与观察：** 载物台、焦点、工具进退及夹爪开合有实际状态；总览与显微图像使用同一模拟时钟。变焦 / 对焦改变视野、尺度与离焦响应，不能切换固定样张。
3. **任务与物理：** 先标定、自动对焦和微珠推移，再夹取释放与分选。液体阻力、黏附、细胞形变、穿刺及注射分别建立模型与验证，不能用外观网格或厂家分辨率证明完成。

建议先打通倒置工作站的电动双侧微针、标定与微珠推移；同时整理体视镜的微夹爪分选和正置镜的侧向探针配置。三类采用共同的工具、成像与控制接口，各自保留独立的装配约束。品牌高细节模型仍需取得适用权限，开放实现可先使用来源明确的组件与独立参数化结构。

验收分别报告装配尺寸误差、干涉与限位、针尖定位 / 力 / 接触、图像标定与对焦、任务成功率，以及 MuJoCo / Isaac 各自的独立运行结果。品牌整机显微镜导入数仍为 **0**；开放核心 CAD 的运行装配单独记录，不与品牌整机混算。

## 8. 本地证据

下载、第三方仓库和原始核查材料留在忽略目录 `temp/microscopy_research/`；没有放进公开运行资产目录。新增证据包括：

- `assets/smaract-step-preflight.json`：完整文件大小、SHA-256、STEP 首尾、毫米单位与检查范围。
- `nikon-oem-brochure.pdf`：官方尺寸与组件图册。
- `nikon-oem-manifest.json`：完整图册大小、SHA-256 和 PDF 元数据。
- `sketchfab-collection-metadata.json`：开放备选的模型 UID、许可与下载状态。
- `SIRIL/`：固定研究提交；不以仓库公开代替复用许可。

后续使用 OpenCascade 检查和剖分：SGE-17 为 41 个实体、SGP-17F 为 56 个实体、Zaber 为 184 个实体，源单位毫米，网格单位米，容差 0.04 mm。网格、逐实体包围盒和哈希在 `cad_meshes/*/manifest.json`。预览使用中性材质，运行材质为独立指定；实际动作与干涉证据另存 `temp/microscopy_demo/cad_integration/`。

原 `assets/zaber-left-flat.step` 曾按大小记录下载完成，实际 CAD 解析失败。右侧通过 HTTP 分段续传、重叠字节校验和总长度校验取得完整文件；失败副本与原记录保留。关节映射、网格校验、装配与任务运行时代码位于 `Hooke/microscopy/`，第三方资产不随代码提交。

离线转换入口为 `Hooke/microscopy/cad_import.py`，依赖可选 `cadquery-ocp`，无需安装到实时仿真环境。当前隔离研究环境在 `temp/microscopy_research/cad_env/`。例如从仓库根目录执行：

```bash
PYTHONPATH=Hooke temp/microscopy_research/cad_env/bin/python -m microscopy.cad_import \
  temp/microscopy_research/assets/zaber-rhf-complete.step \
  --output temp/microscopy_research/cad_meshes/zaber-rhf --preview-gpu 7
```
