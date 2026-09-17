# 空间站、月球和火星的连续实验操作

## 范围

三个环境共用 UR5e / Robotiq 2F-85、两个干燥样品盒、存储位和仪器插槽。第三方 NASA/SRB 外观资产沿用 [资产来源与条款](space_asset_reuse.md)。新增样品盒和夹具是 Hooke 的示例硬件，尺寸和物性不是 NASA 实测参数。

每个环境有三个任务：

| 环境 | 装载与取回 | 质量测量 | 合成反射谱 |
| --- | --- | --- | --- |
| 空间站 | `space_orbital_sample_transfer` | `space_orbital_mass_measurement` | `space_orbital_spectral_measurement` |
| 月球 | `space_lunar_sample_transfer` | `space_lunar_mass_measurement` | `space_lunar_spectral_measurement` |
| 火星 | `space_martian_sample_transfer` | `space_martian_mass_measurement` | `space_martian_spectral_measurement` |

网页 `/space-experiments` 提供真实运行截图、连续 MP4、测量曲线和运行入口。`/backends?task=<任务名称>` 可切换 MuJoCo / Isaac 并实际运行。

## 机械流程与保持结构

机器人实际接触双侧夹爪垫后才允许释放存储约束。接着抬升、转运、插入；只有样品位置误差不超过 6 mm、角度误差不超过 0.12 rad 且接触仪器托盘时才允许锁定。夹具捕获当前相对位姿，不修改样品位置。

存储和仪器使用公开声明的固定约束。仪器有实际驱动的锁扣指示件，质量测量还要求该指示件到位。它不包含锁扣接触力、螺栓柔性或认证机械锁的模型。机器人抓取没有辅助固定约束，运输由夹爪接触实现。

若首次插入没有实际座面接触，控制器最多做三次小幅落座修正，目标最多比插槽基准低 4 mm；仍无接触或夹持丢失会失败。该反馈不会放宽锁定的接触、位置与角度判据。

取回要求正确样品再次被双侧垫接触后才解锁；返回指定存储位并接触座面后重新保持。最终判据还检查实际夹持累计时间、抬升幅度、装样后机器人离开样品的时间和取回历史。无动作运行不能凭初始位置满足任务。

空间站为固定舱壁工作面，没有落地桌腿。局部自由落体参考系取 g=0，样品仍有质量、惯量，存储依靠显式保持。月球 g=1.62 m/s²，火星 g=3.73 m/s²。气压是环境声明；干燥、闭合样品盒不代表气密性或材料真空资格。热见证仍为独立简化模型。

## E1：单未知样品盒总质量

物理链路为：实际滑台位移 → 编码器噪声 → 弹簧定律或振动拟合 → 空载/参考校准 → 总质量估计。测量函数不读取未知质量。

- 月球、火星采用竖直弹性支撑，F=-kq，等待稳定后估计力信号。用空载、0.1 kg 参考盒和未知盒的观测差值标定。
- 空间站采用水平弹簧滑台。公开激励后自由振动，记录实际轨迹，拟合衰减率和频率，修正阻尼后用空载/参考周期比例估计质量。
- 微重力静态称重报告不可辨识。未锁定、机器人接触、触及行程边界、数据太短、信号低于噪声或标定版本不一致均应拒绝估计。
- 输出目标是盒体总质量。统计标准差只包含编码器噪声和拟合，不包含夹具模型、材料参数或现实仪器系统误差。

编码器声明噪声为 1 µm，默认采样 50 Hz；模型参数用于示例与接口校验。当前支撑力由实际位移和声明的弹簧定律换算，PhysX 原生力传感器尚未验收。

## E2：解析测试曲线的合成仪器

流程为暗场、参考盒、未知盒、重复未知测量、参考校正和匹配/拒识。机械锁定、样品身份和光学挡板实际位置控制数据有效性。返回带噪原始曲线，分析通过公开参考和模板完成。

当前为三个公开解析曲线和一个库外曲线，波长 0.5–2.4 µm，192 点；所有世界使用同一参考定义。曲线不来自真实行星材料，也不由纹理、岩石颜色或轨道图像推断。

[USGS Spectral Library v7 官方数据页](https://www.usgs.gov/data/usgs-spectral-library-version-7-data) 标注 CC0；本轮 ScienceBase 下载返回访问挑战页面，没有得到有效数据包。当前运行没有导入 USGS 数据。后续接入须保留具体光谱 ID、样品元数据、波长/缺失掩码、处理步骤和文件哈希。

## 科学接口与评估隔离

`/api/science/capabilities` 列出受支持的读取和分析。已完成的网页任务提供：

- `GET /api/science/jobs/<job>/measurements`
- `GET /api/science/jobs/<job>/measurements/<measurement_id>`
- `POST /api/science/jobs/<job>/analyze`，参数仅为 `method` 与 `measurement_ids`。

记录 ID 受限，读取字段采用白名单；不接受物理 stage、任意路径、隐藏质量/材料或任意 Python。记录读取限定在同一个 job；分析还检查标定版本一致，不允许客户端指定其他 job 的目录。

公开 episode seed 与样品分配分离。宿主私有 campaign key 产生可复现的样品参数，校准 ID 使用不同用途的摘要。真值只写到本地 evaluator 文件和开发用物理档案，不进入科学测量接口。网页目录也只发布任务状态、时长和通过判据，不返回评估误差对、私有目录或 campaign key。空间实验的任务状态和结果下载也采用字段白名单，不返回包含真实质量和 stage 路径的原生构建档案；离线开发验收报告不提供给科学工具。该接口是 HTTP 工具边界，不是运行任意 Agent 代码的操作系统沙箱。

当前机器人流程由确定性状态反馈控制器执行，尚未验收视觉定位、自由决策的 LLM Scientist Agent、多个未知样品排序或正式盲测。

## 真实地形资源

`worlds.pds_terrain` 按精确 HTTP byte range 读取 PDS3 float32 栅格，保留标签和每段原始字节哈希，处理 scale/offset、NoData、投影与南北行方向。只支持明确声明的非旋转等距圆柱投影；不支持的标签会拒绝。

| 地形 | 产品 | 源采样间距 | 缓存 |
| --- | --- | ---: | --- |
| 月球 Hyginus | `NAC_DTM_HYGINUS_E079N0063` | 5 m | 5×5 像素，约 20×20 m |
| 火星倒置河道/扇区 | `DTEEC_034394_1920_034249_1920_L01` | 1.0109258 m | 21×21 像素，约 20.2×20.2 m |

原采样间距和精度不会因网格插值提高。月球源说明给出的 SOCET SET 精度为 5 m，选中片区也有明显地形起伏；不能直接当成精密实验台的安装平面。新增资源遵循 [PDS 科学档案公开使用说明](https://pds-ppi.igpp.ucla.edu/faq.jsp)，保留 NASA/LROC/ASU 与 NASA/JPL/UArizona 来源；每个缓存目录有 `NOTICE.md`。

真实裁剪数据只进入独立地形碰撞验收。默认机器人实验仍使用程序化受控工作区。将真实地形用于完整操作前，需要验证地面安装、地形质量/置信度、可达性和完整任务。完整 OmniLRS、MARTIAN、SRB / Isaac Lab 框架没有因此获得运行资格。

## 重现与验收

从 `Hooke/` 使用项目 Python：

```bash
../.venv/bin/python -m experiments.scene
../.venv/bin/python -m experiments.qualify \
  --backends mujoco isaac --seeds 0 1 --render --controls --gpu 6 \
  --output ../temp/space_experiments/reproduction
../.venv/bin/python -m experiments.instrument_checks \
  --gpu 6 --output ../temp/space_experiments/instrument-checks
../.venv/bin/python -m worlds.pds_qualify \
  --gpu 6 --output ../temp/space_experiments/terrain-checks
```

图像采样可用 `HOOKE_RENDER_FPS=2`；它不改变物理步长或控制频率。不同进程配对时用同一个宿主私有 `--campaign` 文件，GPU 6 的图形任务顺序执行。使用本机 Isaac 4.5 和原驱动，不修改共享系统配置。

完整任务、同长度无动作对照、小型已知负载物理检查、地形碰撞和单元测试分别记录。最终数值以 [完整任务报告](validation/space_experiments_summary.json) 及其关联证据为准，正在运行的项不会记为通过。MP4 和 WebM 导出后逐个完整解码验证；网页优先选择 MP4，不支持 H.264 时使用 WebM。

初版 MuJoCo 的 90 次任务有 12 次未接触座面而失败，错误和哈希保留在 [修复与重跑记录](validation/space_experiments_attempts.json)。加入有上限的接触落座反馈后，完整重跑月球、火星的 60 次任务及同长度对照全部通过；原空间站 30 次不受该修正分支影响。当前 [152 项单元检查及实际中止验证](validation/space_experiments_unit.json) 通过。

每次资格验证使用新的 `--output` 目录；已有 episode 不覆盖。未通过、被中止和修复前的本地原始数据保留，最终报告统计修复后的不同任务/种子配置。

## 2026-09-17 实测验收

本表统计修复后的不同任务/后端/种子配置。开发重跑和媒体重跑不重复增加分母；两个网页实际运行案例另列。

| 验收项 | 实测结果 | 范围 |
| --- | ---: | --- |
| MuJoCo 完整操作 | 90/90 | 九项任务各 seed 0–9 |
| Isaac 完整操作 | 21/21 | 九项任务各 seed 0、1，另复查月球三项 seed 8 |
| 同长度无动作对照 | 111，误成功 0 | 两端分别覆盖对应任务实际时长 |
| 配对初始模型 | 21/21 | XML SHA-256、每对 412 个编译数组精确比较 |
| 保存轨迹审查 | 222/222 | 实际状态有限、时钟/步数一致、对照足够长 |
| 已知负载仪器夹具 | 30/30 | 每端 15 项，含步长、倾斜轴与解锁负对照 |
| 实际 PDS 地形碰撞 | 4/4 | 月球/火星裁剪，两端竖直导向球 |
| 单元检查 | 152/152 | 机械、观测、分析、接口与已有功能 |
| 发布媒体 | 18 PNG + 18 MP4 + 18 WebM | 九项任务 × 两端；连续图像、完整解码和帧数核验 |
| 桌面/手机网页 | 36/36 | 三环境 × 三操作 × 两端 × 两视口 |
| 实际网页双后端运行 | 2/2 | 火星质量测量、公开观测重新分析、默认仪器视角 |

质量测量的最大绝对误差：MuJoCo **0.0333 g**，Isaac **0.0169 g**。这是已声明弹簧/编码器模型与宿主真值的开发验收，不是现实仪器精度。最大仿真时长 54.568 s，声明上限 120 s。合成光谱共记录 25 次库内匹配和 12 次库外拒识，最大重复 RMS 0.00067。

真实地形两端最终接触高度差：

| 裁剪 | 高度差 | 来源一致性 |
| --- | ---: | --- |
| lunar | 0.0448 mm | 相同源标签和裁剪 SHA-256 |
| martian | 0.0250 mm | 相同源标签和裁剪 SHA-256 |

这些高度差验证导向球和同一裁剪的有限碰撞结果，不能解释为卫星 DEM 精度、地质精度或完整地面机器人验收。

混合小夹具与机器人场景的长批处理中，发生一次 Isaac 热重载超时；该对照仿真时长为 0 秒，没有计为完成。保留原记录后，以相同场景、种子和 campaign 在新 worker 中完成全时长重跑；最终表采用重跑结果。详见 [失败与恢复记录](validation/space_experiments_attempts.json)。网页每个任务使用独立进程；长期热重载稳定性不能由本表外推。

关联报告：[完整操作](validation/space_experiments_summary.json)、[轨迹审查](validation/space_experiments_state_audit.json)、[仪器物理](validation/space_experiments_instruments.json)、[真实裁剪碰撞](validation/space_experiments_terrain.json)、[浏览器](validation/space_experiments_browser.json)、[实际网页运行](validation/space_experiments_live_web.json)。

### 已出图案例的实际耗时

以下为相同场景、相同 GPU 6、2 fps 图像采样的 seed 0 案例。墙钟包含场景加载、控制、同步、记录和出图；批量 Isaac worker 的应用冷启动在单回合计时之外，因此这不是完整启动成本或纯引擎性能基准。

| 环境 | 操作 | MuJoCo 回合墙钟 | Isaac 回合墙钟 |
| --- | --- | ---: | ---: |
| 空间站 | 装载/取回 | 21.44 s | 112.28 s |
| 空间站 | 质量测量 | 36.44 s | 292.74 s |
| 空间站 | 合成光谱 | 28.31 s | 206.81 s |
| 月球 | 装载/取回 | 14.66 s | 110.75 s |
| 月球 | 质量测量 | 22.23 s | 256.18 s |
| 月球 | 合成光谱 | 19.28 s | 197.97 s |
| 火星 | 装载/取回 | 14.99 s | 107.70 s |
| 火星 | 质量测量 | 22.51 s | 254.43 s |
| 火星 | 合成光谱 | 20.17 s | 115.89 s |

Isaac 当前仍明显更慢，玻璃、阴影、反射与视图响应也未获得像素等价认证。相同资产、初始模型和任务成功不代表性能或视觉完全一致。

### 查看实际结果

服务器局域网入口：[空间实验操作](http://10.5.174.93:8080/space-experiments)。可切换三个环境和三种操作，查看 MuJoCo/Isaac 的真实图像、完整 MP4 和公开测量曲线，再打开任务实际运行。

浏览器实际截图：[空间站](assets/space-experiments-web-orbital.png)、[月球](assets/space-experiments-web-lunar.png)、[火星](assets/space-experiments-web-martian.png)、[手机页面](assets/space-experiments-web-390.png)、[实际运行](assets/space-experiments-web-live.png)。真实裁剪的 Isaac 出图：[LROC 月球](assets/space-experiment-terrain-lunar-isaac.png)、[HiRISE 火星](assets/space-experiment-terrain-martian-isaac.png)。

两个后端的种子覆盖不同，实际耗时和渲染响应也不同；以上通过率不证明任意 MJCF、逐像素、轨迹或性能等价。当前仍使用确定性反馈控制器和解析光谱，默认操作地面仍为程序化工作区。

### 后续生成物与仓库管理

新增截图、视频、完整实验记录和原始日志放入已忽略的 `temp/`；代码、文档和精简验收摘要可正常提交。现有 `docs/assets/` 是本次保留的文档与网页演示素材。

`python -m experiments.publish` 默认把摘要写入 `temp/space_experiments/publication/summary.json`，图片和视频写入其 `media/` 子目录，公开测量证据写入 `media/evidence/`。可通过 `--output`、`--media-dir` 指定其他本地输出路径；后续更新沿用忽略目录约定。
