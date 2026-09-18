# 双侧吸持与悬浮细胞注射

Current cell display uses estimated grayscale phase contrast and a separate
fluorescence channel. See [cell imaging](microscopy_imaging.md) for the image
formation model, nominal dimensions and calibration limits. Earlier image
records below retain their original renderer and settings.

局域网双环境入口：`http://10.5.174.93:8084/microscopy?experiment=suction_injection`。
选择“运行实验”可执行完整流程，也可分别控制吸持管、注射针与两条压力通道。选择“三维近景”可查看管口、针尖与细胞；显微视野为带标尺的合成图像。

## 场景与流程

- 本机演示采用 TE2000-S 独立参考镜体，两个电动微操机构各使用 Zaber M-LSM 的完整 184 部件 CAD；另保留 openFrame 核心 CAD 配置。镜体光学附件、延长支架、针座与玻璃管是明确标注的原创组件。
- 左侧空心吸持管内半径 4 µm、外半径 6 µm，与水平夹角 30°；右侧空心注射针内半径 0.25 µm、外半径 0.60 µm，与水平夹角 35°。两者暴露段长 12 mm；完整 CAD 微操机构、针座和玻璃管随安装姿态一起调整。吸持管的 12 µm 外径用于负压吸持，与 1.2 µm 外径的穿刺针分开。
- 注射针距口部 100 µm、300 µm 处的外径分别为 2 µm、4 µm，随后逐渐过渡至后端 1 mm 外径玻璃管；不再使用整段直锥。三维网格、合成显微轮廓、内孔分段流阻共用名义 SI 轮廓。这是独立估算的拉制针形，不是厂商实测针形。
- 目标是半径 18 µm、名义体积 24.429 pL 的悬浮球形 phantom，中心距玻璃面 26 µm。它不是卵母细胞、胚胎或真实细胞类型的标定模型。
- 自动对焦与蓝色核标记图像定位 → 吸持管接近 → 建立负压 → 注射针接近与膜压凹 → 穿刺 → 胞质注射 0.5 pL → 关闭注射压力与补偿压力 → 退针 → 解除吸持 → 吸持管后退与示踪检查。

独立 Hold 与 Inject 通道参考了 [Sutter XenoWorks 官方说明](https://www.sutter.com/microinjection/xenoworksdm)。该资料支持设备工作方式，不为本演示的具体尺寸、压力和细胞参数提供生物标定。

## 力学与液体边界

器械电机、关节、位姿与反力分别由所选 MuJoCo 或原生 Isaac/PhysX 后端执行。细胞平移使用独立的简化过阻尼模型，不是这两个后端的软体细胞解算：

`γ·v = Fneedle + Fbuoyancy + Khold·(anchor − position)`

`γ = 6πµr`，介质黏度 0.001 Pa·s；吸持刚度 0.06 N/m。吸持状态下对线性受力响应作每步解析积分。游离状态按当前针力与浮力更新位置，适用范围是演示中的准静态、小变形近似，不能据此预测高速流动或细胞损伤。

细胞与介质密度都设为 1,000 kg/m³。器械仍处于 9.81 m/s² 重力环境；目标细胞的重力由浮力抵消。密度匹配是 phantom 的明确条件，不是关闭整个场景的重力。

吸持能力为实际管口负压乘以内开口面积；默认 −1,500 Pa 对应约 75.40 nN。负压有 40 ms 响应时间；管口需要满足接近条件并持续 30 ms 才能建立吸持。针力负载超过能力、负压不足或吸持管堵塞会导致吸持丢失。针与吸持管的模型反力投影至实际电机轴；游离细胞不将针反力强行传至载物台。

膜形变与穿刺沿用未标定的标量松弛/阈值模型；注射流量使用分段线性内孔的串联 Poiseuille 流阻、压力响应和守恒账本。提前释放不会将液体记录抹掉：若有液体在失去吸持后进入模型细胞，任务保留记录并判定失败。三维细胞位置与示踪颜色读取该模型状态；压凹和穿刺标记仅在合成显微图像展示。

未实现真实吸入舌形变、细胞旋转、膜修复、存活、浴液 CFD 或实验生物参数标定。界面显示的小数位不证明实机定位精度。

## 当前 30°／35° 分段针形的验证

MuJoCo 与原生 Isaac 各完成 11,353 步、19/19 检查，胞内剂量各为
0.5 pL。原生步数与实际事件数、完整关节归档一致；记录路径的
器械外壳表面检查通过。空心管完整网格到玻璃上表面的最小距离
下界计入原生位姿误差：注射针约 31.714 µm，吸持管约 27.396 µm。
证明限定于这些记录路径，不代表任意动作、制造配合或生物标定。
相关单位测试已覆盖 seed 0、1、4 正常流程及吸持失效对照。
新记录在忽略的 `temp/microscopy_demo/cad_integration/pipette_mount_v1/`。

## 此前 45° 针形配置的验证记录

| 项目 | 结果 |
| --- | --- |
| MuJoCo CAD 完整任务 | 9,901 步，19/19 检查通过，胞内剂量 0.5 pL |
| 原生 Isaac CAD 完整任务 | 实际 PhysX 9,901 步，与报告一致，19/19 检查通过 |
| 标准配置 3 种图像布局 | seed 0、1、4 全流程通过；无焊接约束、无 mocap 细胞驱动 |
| 正常吸持 | 约 4.628 s，最大吸持负载约 29.93 nN，最大细胞位移约 0.50 µm |
| 无吸持 | 细胞被针力推离，未建立穿刺和胞内剂量 |
| 吸持管堵塞 | 上游存在负压，管口有效吸持能力为零，不建立吸持 |
| 负压不足 | 吸持丢失并保留提前释放记录；受支持穿刺不能通过 |
| 注射时提前释放 | 保留失去吸持后的剂量与失败记录 |
| 浏览器 | 独立电动吸持轴、吸持堵塞、自动双侧任务与其余 4 项任务均通过，无脚本错误 |
| 机构间距 | MuJoCo 9,901 步和记录的原生 PhysX 9,901 步均通过器械表面距离界限检查；原生检查计入位置及姿态误差余量 |

任务测试见 [test_suction_injection.py](../tests/test_suction_injection.py)，离线间距与原生归档验证见 [test_assembly_clearance.py](../tests/test_assembly_clearance.py)。
间距检查覆盖该次轨迹中的所选器械外壳与支架表面，不覆盖同一机构内部装配、实体包含、全部自由行程或被排除的针尖/细胞接触区域。

本地完整记录、录像、截图与原生 RGB 在忽略的 `temp/microscopy_demo/cad_integration/`：

- `recordings_final/suction_injection/`：MuJoCo 执行结果、阶段图像和演示录像。
- `native_suction/suction_injection/`：原生执行报告、实际关节轨迹与 Isaac RGB。
- `suction_clearance/`、`native_clearance/suction_injection/`：实际器械网格及逐步间距证据。
- `browser_suction/validation.json`：实际浏览器操作与截图。

上述是历史 `cad` 阶段的证据。当前主网页改用 `reference` 镜体／载物台与许可明确的 `parallel-v4` 真实电动机构，双侧吸持已在两后端及实际浏览器通过；准备见[并联 CAD](microscopy_parallel.md)，完整目标结果见[验收范围](microscopy_goal.md)。

## 最终 CAD 失败对照与颜色修复

曲面法线与逐对象离焦配置分别完成五类刻意定位对照：无吸持、吸持管堵塞、负压不足、注射时提前释放、注射针堵塞。在 MuJoCo 和原生 Isaac 都符合预期：无支持的受压细胞移动而不建立有效穿刺/剂量；堵塞可以保留上游压力而有效通道能力为零；失去吸持保留失败状态和已经注入的液体。该组测试使用真值位置组织压力对照，不是图像控制器成功率，也不是生物验证。

原生提前释放时保留约 0.0761 pL 剂量与失败记录。此情形首次暴露部分剂量颜色使用 NumPy 标量而被二进制通信编码成字节的错误；显示提供者现输出 Python float。修复后的五类原生对照全部完成实际 PhysX 步进与 RGB，事件数均等于观察轨迹步数；8 项吸持测试包含部分剂量二进制往返回归并通过。

新原生对照在 `failure_controls_depth_fixed/isaac/`，对应 MuJoCo 对照在 `failure_controls_depth_v2/mujoco/`；修复前的原生 RGB 失败另保存在 `failure_controls_depth_v2/isaac/`，不计作对照通过。新五项正常流程、浏览器与间距证据分别见 `native_optical_depth/`、`browser_optical_depth/`、`optical_depth_native_clearance/`。全部位于上述忽略的本地记录根目录。
