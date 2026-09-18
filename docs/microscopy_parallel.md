# 开源电动并联微操机构

Hooke 的 `parallel-v4` 配置复用 MicroManipulatorStepper v4 的实际 FreeCAD
机械设计，集成到倒置显微实验台。镜体、载物台、针座、悬臂和工具附件采用
独立参考建模；此配置无需 Zaber、SmarAct 或 X-ASR 私有 CAD。

## 来源、许可与尺度

固定来源为 [MicroManipulatorStepper](https://github.com/0x23/MicroManipulatorStepper/tree/63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94)，
提交 `63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94`。作者的
[许可原文](https://raw.githubusercontent.com/0x23/MicroManipulatorStepper/63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94/LICENSE)
明确覆盖硬件设计、软件、文档及概念，允许研究使用、修改与分发，要求保留
版权和许可声明。完整原文另保留在
[随代码的声明](../Hooke/microscopy/licenses/MicroManipulatorStepper.txt)。

部分 FCStd 文档的 `License` 属性仍是 `All rights reserved`，与仓库声明不同；
获取记录保留该元数据。本配置依据作者明确覆盖硬件设计的固定版本仓库许可，
没有把文件属性改写为 MIT，也不将这份许可扩大到其他品牌资产。

19 个 FCStd 文档解析出 **173 件物理实例、53 种保存形状**。保存的链接位置、
零件外部变换和毫米单位分别核对；剖分后的运行网格采用米。
这是一套保存几何快照，不要求在服务器重新执行 FreeCAD 特征树。
禁止缩放整机来配合针尖；显微近景通过相机放大。

## 运动机构与独立接口

- 三个电动转轴驱动六根刚性连杆；自由平台由六个闭合约束连接。
- 固定底座、定子、转子、连杆和平台按实际 CAD 归属分别绑定；没有只移动整机网格。
- 公共控制仍使用相对初始位置的 XYZ 米制位移；内部逆运动学转换成电机角度。
原始电机弧度不会被当作网页的微米轴。批量命令在修改轴、压力前统一验证。
- 当前工具轴命令范围各为 ±12 mm，另检查逆运动学可达性；不是任意组合位置或
  实机完整工作空间的认证。载物台允许 X/Y 各 ±4 mm，调焦 ±1 mm。
- 原创外侧立柱、悬臂及倾斜板支撑机构，避免立柱穿过载物台和镜体。
  支撑、180 mm 工具伸出、针座适配及承载均为设计估算，不称原厂安装或一比一配合。
- 标定微珠夹爪增加原创估算的电机外壳、导轨和滑块；两侧夹爪由原有电动
  滑动轴实际开合。杆后端上抬 2 mm，避免靠近微腔壁。外壳不添加碰撞或质量，
  不称真实品牌 CAD、电子学或载荷模型。

注射侧针座与水平夹角 35°，吸持侧 30°；针尖外径 1.2 µm、内径 0.5 µm。
贴壁细胞名义尺寸为 36 × 28 × 8 µm，悬浮细胞直径 36 µm。
三维轮廓、合成显微成像与毛细流阻共用米制参数。

上游 [README](https://raw.githubusercontent.com/0x23/MicroManipulatorStepper/63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94/README.md)
明确区分编码器分辨率与绝对精度。本仿真不继承其 50 nm 分辨率主张；
质量、惯量、控制增益、重力补偿、器械精度均未实机标定。
橡皮筋网格随连杆刚性移动，不模拟弹性预紧、电机电子学或热漂移。
细胞采用未生物标定的降阶力学和流量模型，显微视野为合成成像。

### 原生求解坐标与位置反馈

Isaac 的求解原点移到注射平台的名义初始位置，以减小大坐标下的浮点
量化。源模型、网页和归档轨迹继续使用原来的米制世界坐标；刚体、
世界关节锚点、相机、显示曲面、接触点和力的作用点统一转换。
转换记录保留 `physics_options.world_origin_m`，默认零偏移供其他场景使用。

细胞操作在运动稳定等待后，按真实物理步采集 50 ms 位置窗口。
闭环判断窗口平均残差 ≤50 nm，同时检查各轴 RMS 抖动 ≤0.3 µm、
峰值偏离窗口均值 ≤0.8 µm；最多十次受限电机校正，失败即停止。
每次采样同时记录平均残差、瞬时残差和抖动，避免把均值抵消解释成
瞬时定位精度。细胞受力、穿刺和流量仍使用每一步实际位置，未经过平均。
这些是当前仿真的数值停止条件，没有验证实机精度或生物安全性。

## 本地准备与运行

从仓库根目录获取固定来源到新的忽略目录：

```bash
PYTHONPATH=Hooke .venv/bin/python -m microscopy.parallel_assets \
  --output temp/microscopy_research/parallel-source
```

该命令保留源文件哈希、ZIP/XML 检查、外部文件引用及完整许可。
若 Conda 的默认证书路径失效，可显式传入系统信任库
`--ca-file /etc/ssl/certs/ca-certificates.crt`；TLS 证书验证保持开启。
已有目录不会被覆盖，失败记录保留，重试应使用新目录。

在包含 `cadquery-ocp` 的离线环境转换；运行仿真无需 OpenCascade：

```bash
PYTHONPATH=Hooke /离线CAD环境/bin/python -m microscopy.freecad_snapshot \
  --source temp/microscopy_research/parallel-source \
  --output temp/microscopy_research/parallel-meshes
```

从 `Hooke/` 启动网页，使用实际空闲 GPU 编号及已有 Isaac 安装配置：

```bash
HOOKE_MICROSCOPY_ASSETS=reference \
HOOKE_MICROSCOPY_STAND=te2000-s-reference \
HOOKE_MICROSCOPY_STAGE=reference HOOKE_MICROSCOPY_OPTICS=estimated \
HOOKE_MICROSCOPY_MANIPULATOR=parallel-v4 \
HOOKE_MICROSCOPY_PARALLEL_ROOT=/仓库绝对路径/temp/microscopy_research/parallel-meshes \
HOOKE_MICROSCOPY_MEDIA_ROOT=/仓库绝对路径/temp/parallel-demo \
HOOKE_MICROSCOPY_GPU=0 HOOKE_ISAAC_GPU=1 HOOKE_MICROSCOPY_LIVE_FPS=1 \
../.venv/bin/python -m webui.server --host 0.0.0.0 --port 8087
```

访问 `/microscopy?experiment=suction_injection`，通过选择器切换两种环境；
追加 `&backend=isaac` 可以直接打开 Isaac。原始 CAD、网格、轨迹、截图和
录像只放在忽略目录，当前工作没有提交或推送。

## 检查方法与当前证据

装配表面检查按实际模型的底座、杆系、平台和支撑树分组，逐件断言 CAD
零件没有遗漏。初版支撑与镜体、导轨相交；改用外侧立柱后，双侧与三工具
初始配置全部通过，分别检查 823 与 937 件表面，零跳过。

运动检查记录每个物理步，以表面距离减去平移和半径乘转角的位移上界。
原生检查读取哈希绑定的实际 PhysX 关节轨迹，只用 MuJoCo 计算几何位置，
不重新运行 Source 物理。该方法覆盖记录中的姿态与声明容差，不能认证
未观测动作、零件包含、同总成内部所有配合、加工公差或实机承载。

当前证据目录：

- `temp/microscopy_research/parallel_assembly_reset_v1/`：初版支撑的真实干涉记录。
- `temp/microscopy_research/parallel_assembly_reset_v2/`：支撑修改后的两个初始配置。
- `temp/microscopy_demo/cad_integration/parallel_v4_web_v1/browser_full_v3/`：当前两后端十条实际网页流程、98 项判据与截图；二十个新录像完整解码通过。
- `temp/microscopy_demo/cad_integration/parallel_v4_motion_v3/isaac/`：当前五条原生完整运动表面检查，零跳过。
- `temp/microscopy_research/parallel_sampled_motion_v1/mujoco/`：当前五条 Source 完整运动表面检查，零跳过。
- `temp/microscopy_research/parallel_v4_controls_sampled_v1/`：两后端二十项失败对照、84 项行为判定。
- `temp/microscopy_demo/cad_integration/parallel_v4_production_v1/`：8084 主网页部署、二十个录像、十个实际浏览器回放及冷启动检查。
- `temp/microscopy_demo/cad_integration/parallel_v4_final_v1/verification-summary.json`：逐项目标验收、来源与轨迹／媒体哈希核验。

[当前主网页](http://10.5.174.93:8084/microscopy?experiment=suction_injection&backend=isaac)
可直接打开 Isaac。此次 demo 目标的验收指标与模型边界见[目标验收](microscopy_goal.md)。
