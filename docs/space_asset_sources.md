# 空间站、月球与火星的开放场景和资产调研

核查日期：2026-09-17。本页记录候选来源和接入方案，不是外部资产的运行验收报告。当前 `/space-worlds` 仍使用 Hooke 原资产与程序化场景；后续已下载并接入美国实验舱、扶手、Apollo 样品 1 与火星样品管，新增独立的 `/space-assets` 场景变体。范围、使用条款和实际证据见 [资产复用记录](space_asset_reuse.md)。完整 SRB、OmniLRS、MARTIAN 框架仍未运行验证。

## 优先来源

| 用途 | 来源 | 可以复用什么 | 格式、许可与接入边界 |
| --- | --- | --- | --- |
| 空间站舱内实验区 | [NASA Astrobee Media](https://github.com/nasa/astrobee_media) | 美国实验舱、欧洲实验舱、日本实验舱、节点舱、观察舱，以及扶手、对接站、舱内贴图 | DAE、PNG、URDF；媒体遵循 NASA Media Usage Guidelines，URDF 含 Apache-2.0 声明；原运行环境是 Gazebo，需要转换网格和资源路径 |
| 三种空间域与实验操作参考 | [Space Robotics Bench](https://github.com/AndrejOrsula/space_robotics_bench) | 月面、火星地形、轨道操作模板，样品收集任务，机器人和工具的配置方式 | 代码 MIT / Apache-2.0；原生安装方案当前使用 Isaac Sim 5.0；本机 Isaac 4.5 中优先复用独立资产，完整框架未验证 |
| 实验样品和工具 | [SRB 独立资产库](https://github.com/AndrejOrsula/srb_assets) | 22 个 Apollo 样品模型、Mars 2020 采样管、采样铲、太阳能板，以及着陆器和探测车 | USDC、USDZ；作者自制资产 CC0，第三方资产按归属清单逐项记录原许可；样品外形不提供质量、化学成分或仪器标定 |
| 月球地形与光照 | [OmniLRS](https://github.com/OmniLRS/OmniLRS) | Lunalab、Lunaryard、小区域地形、岩石、陨坑、日地位置计算和大区域 DEM 方法 | USD、纹理、DEM；代码 BSD-3-Clause，外部数据和资产来源仍需逐项记录；当前配置固定 Isaac Sim 5.0，资产使用 Git LFS |
| 月面细节生成的备选 | [NASA/JPL LuNaSynth](https://github.com/nasa-jpl/lunasynth) | 从月球 DEM 生成岩石、陨坑和不同日照下的地形图像 | 代码 Apache-2.0；适合作离线地形与图像生成工具，不是交互式实验仿真环境；程序化细节不能标成实测地形 |
| 月壤表面与岩石视觉 | [Poly Haven Lunar 扫描](https://blog.polyhaven.com/moon/)、[Moon Rock 01](https://polyhaven.com/a/moon_rock_01) | 岩石模型、漫反射、法线、粗糙度和位移贴图 | 模型提供 USD、glTF 等，资产 CC0；扫描来自 Spaceport Rostock 的地面月壤模拟实验室，不是真实月球原位扫描；先取低分辨率成品，无需下载约 800 GB 原始扫描集 |
| 火星实测区域 | [NASA/JPL MARTIAN](https://github.com/nasa-jpl/martian)、[HiRISE Jezero DTM](https://www.uahirise.org/dtm/ESP_045994_1985) | HiRISE 高程与正射影像生成的火星地形、太阳与相机配置 | 工具代码 Apache-2.0，Blender 4.0；独立 DTM 导入插件为 GPL；输入 IMG、JP2，可生成 BLEND 后再导出；这是离线地形/图像工具，气体与机器人动力学需要另接 |
| 太空背景与外部装备 | [NASA 3D Resources](https://www.nasa.gov/3d-resources/)、[ISS 外观模型](https://science.nasa.gov/3d-resources/international-space-station-iss-b/) | 空间站外观、探测器等背景与装备模型 | 可下载不等于统一开源软件许可，按资源说明与 NASA 媒体条款使用；ISS 外观模型不能代替可活动的实验舱内部 |

[SRB 归属清单](https://andrejorsula.github.io/space_robotics_bench/misc/attributions.html) 将 Apollo 样品列为 CC0，Mars 2020 采样管归于 JPL-Caltech，DLR Oberpfaffenhofen 测试场为 CC BY-SA 4.0，部分背景图另有许可。不能用资产库的 CC0 声明覆盖这些第三方条款。

[NASA 媒体条款](https://www.nasa.gov/nasa-brand-center/images-and-media/) 覆盖三维模型的纹理和多边形数据，要求注明来源并区分第三方内容及标识；不要把 NASA 媒体资产统一改标为 CC0。[HiRISE 图像使用说明](https://www.uahirise.org/media/usage.php) 声明其公开图像可使用，并请求注明 NASA/JPL/University of Arizona；地形产品同时保留自己的产品标签和引用信息。[Poly Haven 资产许可](https://polyhaven.com/license) 为 CC0。

## 已定位的具体资产

以下文件路径已在固定上游提交的 Git 树中核查，其中首批四个资产已转换和核查，其余仍为候选。

### 空间站

[Astrobee 舱段目录](https://github.com/nasa/astrobee_media/tree/1cb0620121099cc848f5a3db480464188dc91ea3/astrobee_iss) 包含：

- `meshes/us_lab.dae`、`eu_lab.dae`、`jpm.dae`：实验舱候选。
- `meshes/node_1.dae`、`node_2.dae`、`node_3.dae`、`cupola.dae`：连接与观察舱。
- `media/materials/textures/`：对应的机架、舱壁等贴图。
- `urdf/model.urdf`：实际检查了模块安装变换、网格引用及碰撞声明；`package://` 路径需要重新解析。
- 同仓库的 `astrobee_handrail_*`、`astrobee_dock`：扶手和对接资产。

第一批建议只选美国实验舱及扶手，把已有机械臂、实验台、仪器和样品架装入。使用美国实验舱外观不代表复制了真实 ISS 仪器配置或提供了气密舱。

### 月球与火星实验资产

[SRB 固定资产版本](https://github.com/AndrejOrsula/srb_assets/tree/54b1282a5f89ead8ff51b869060e68083fa3ad2f) 包含：

- `object/rock/apollo_sample1.usdz` … `apollo_sample22.usdz`：有来源的月球样品外形。
- `object/rock/spaceport_moon_rock1.usdz` … `spaceport_moon_rock7.usdz`：地面模拟场扫描岩石。
- `object/sample_tube.usdc`：火星采样管候选。
- `object/scoop/`、`object/solar_panel.usdz`：采样工具与环境装备候选。
- `scenery/lunalab.usdc`、`scenery/oberpfaffenhofen_test_site.usdc`：地面模拟测试场，不能标成实测月面/火星表面。

外形模型与实验物性分开保存。样品真实质量、惯量、材料和目标反射谱需要独立定义、引用和验证；视觉贴图不作为成分或光谱测量的真值。

## 实测地形的数据边界

月球地形可直接从 [LRO LOLA/PDS](https://pds-geosciences.wustl.edu/missions/lro/lola.htm) 选取高程产品，不必把整个 OmniLRS 仿真栈作为依赖。[LOLA 数据说明](https://pds-geosciences.wustl.edu/missions/lro/lola_faq.htm) 区分格网、测量轨迹和原始数据；读取 IMG/JP2 时按对应标签解释比例、偏置、无效值和投影。保存区域、产品 ID、采样间距与原始数据哈希；插值和新增岩石分别标记为生成内容。

火星第一处可选 [Jezero：DTEEC_045994_1985_046060_1985_U01](https://www.uahirise.org/dtm/ESP_045994_1985)。产品页标明高程格网为 1 m/pixel，DTM 约 381 MB，配套较低分辨率正射影像约 34 MB。原始产品随后裁出局部实验区；一米格网不提供毫米级实验台地面细节，新增小石块和细节仍属于程序化几何。

## 适配当前 Hooke 的方案

本机已验证的是 Isaac Sim 4.5、535.230.02 驱动及 GPU 6。核查到 [SRB Dockerfile](https://github.com/AndrejOrsula/space_robotics_bench/blob/7528ff81f1ac0b34ba259b1ae150fdb4f6a90b5e/Dockerfile) 和 [OmniLRS pixi.toml](https://github.com/OmniLRS/OmniLRS/blob/5429512dfb80808a047e44c1b97dc0aadc3ce327/pixi.toml) 均使用 Isaac 5.0.0。这不证明独立 USD 资产无法供 4.5 使用，也不证明完整项目能在本机直接运行；需要分别检查资产和运行时。

[Isaac 4.5 格式文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/formats.html) 支持将 OBJ、FBX、glTF 转为 USD。DAE 需要先通过支持它的工具转换，不能假定同一转换器直接支持所有源格式。USD/USDC/USDZ 也要检查引用、贴图、材质、单位、轴向和 schema 兼容性。

接入遵循以下顺序：

1. 固定来源版本，只取首批资产及其贴图、引用依赖；记录原作者、许可、产品 ID 与哈希。
2. 保留源资产，离线生成米制、Z-up 的公共网格和材质。视觉网格与碰撞几何分开，两个引擎使用同一安装变换和碰撞设计。
3. 先接空间站单舱、月面一块局部地形、火星一块局部地形；复用同一套 Hooke 机械臂与实验资产。
4. 再接 Apollo 样品外形、采样管和铲具，补齐可抓握区域、质量惯量、关节、操作与复位接口。
5. 每批资产验收双引擎加载、真实截图、尺度、自由空间、接触、复位和具体操作；性能测量记录加载时间、步进时间与显存。

空间站是关键碰撞特例：Astrobee URDF 将舱段网格同时用于视觉和碰撞。把完整舱段直接换成一个凸包会填满舱内空腔；必须拆成舱壁、地板、机架等保留通道的碰撞结构。两个引擎分别支持的静态三角网格方案也需实际验证，不能仅凭图片判断机器人能在舱内运行。

资产接入后继续由 `worlds/profiles.py` 声明并配置重力与环境分区。空间站舱内有空气、舱外近真空；月面暴露工作区近真空；火星工作区有稀薄 CO₂。背景图、岩石材质和气压字段不会自动实现气流、压差、沸腾、热传导、尘埃静电或辐射损伤。外部项目的热学/地面模型另行评估其接口、参数来源和验收，不能作为导入模型后的默认能力。

## 固定版本与本次证据

| 上游 | 实际读取的提交 | 核查内容 |
| --- | --- | --- |
| `nasa/astrobee_media` | `1cb0620121099cc848f5a3db480464188dc91ea3` | Git 文件树、README、ISS URDF、model.config |
| `AndrejOrsula/srb_assets` | `54b1282a5f89ead8ff51b869060e68083fa3ad2f` | Git 文件树、README、独立资产路径及官方归属清单 |
| `AndrejOrsula/space_robotics_bench` | `7528ff81f1ac0b34ba259b1ae150fdb4f6a90b5e` | Git 文件树、Dockerfile、资产子模块配置与官方文档 |
| `OmniLRS/OmniLRS` | `5429512dfb80808a047e44c1b97dc0aadc3ce327` | Git 文件树、README、license、pixi.toml、LFS 配置与官方安装 wiki |
| `nasa-jpl/martian` | `f03c16d47b42ebabb3e63d392576ea75b8bf5b87` | Git 文件树、README、LICENSE、地形输入与离线导出说明 |

LuNaSynth、Poly Haven、NASA 3D 和 PDS 产品页面属于网页核查，不列为已下载或已导入模型。开发机的检索记录与局部 Git 元数据保存在 `temp/space_asset_research/`，不把这些临时目录加入产品依赖。首批四个模型的实际接入见 [资产复用记录](space_asset_reuse.md)；已有三世界原型的实际检查见 [场景文档](space_worlds.md)。
