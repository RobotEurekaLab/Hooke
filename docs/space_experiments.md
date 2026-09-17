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

公开 episode seed 与样品分配分离。宿主私有 campaign key 产生可复现的样品参数，校准 ID 使用不同用途的摘要。真值只写到本地 evaluator 文件和开发用物理档案，不进入科学测量接口。该接口是 HTTP 工具边界，不是运行任意 Agent 代码的操作系统沙箱。

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

完整任务、同长度无动作对照、小型已知负载物理检查、地形碰撞和单元测试分别记录。最终数值以 [完整任务报告](validation/space_experiments_summary.json) 及其关联证据为准，正在运行的项不会记为通过。MP4 导出后逐个完整解码验证。

初版 MuJoCo 的 90 次任务有 12 次未接触座面而失败，错误和哈希保留在 [修复与重跑记录](validation/space_experiments_attempts.json)。加入有上限的接触落座反馈后，完整重跑月球、火星的 60 次任务及同长度对照全部通过；原空间站 30 次不受该修正分支影响。当前 [148 项单元检查及实际中止验证](validation/space_experiments_unit.json) 通过。

每次资格验证使用新的 `--output` 目录；已有 episode 不覆盖。未通过、被中止和修复前的本地原始数据保留，最终报告统计修复后的不同任务/种子配置。
