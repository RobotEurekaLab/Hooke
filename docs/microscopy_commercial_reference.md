# 商用倒置镜与紧凑针座参考场景

本阶段根据用户指出的镜体、工具夹持和整体比例问题，新增独立的 TE2000-S 参考场景。它以厂家尺寸和系统实拍为依据，保留真实尺寸的 Zaber 电动微操 CAD；镜体和 HI-7/HIR 针座为独立建模，尚未取得商用镜体完整 CAD，也未通过整机一比一验收。

本文记录镜体／针座的尺寸依据和历史品牌 CAD 集成。当前主网页的电动微操已替换为许可明确的[真实并联 CAD](microscopy_parallel.md)，完整 demo 验收见[目标报告](microscopy_goal.md)；镜体与针座的独立重建边界继续适用。

## 实物依据与尺度

镜体采用 [Nikon TE2000 官方册](https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf)第 3 页照片和第 24 页 TE2000-S 前／侧视图。白色底座、前方双目观察筒、左输出口、后方照明柱及上方聚光器，与 [Yu Sun 团队贴壁注射系统 Fig. 2](https://amnl.mie.utoronto.ca/data/J109.pdf)的布局方向一致。不同型号和附件不混用：本场景参考 TE2000-S，不能写成 TE2000-E、Ti2 或 Olympus IX83。

| 项目 | 厂家参考尺寸 | 当前处理与边界 |
| --- | --- | --- |
| 镜架底座 | 宽 195 mm，侧视标注深 476.4 mm | 原创曲面按这些尺寸建立，未标出的轮廓为估算；左侧相机附件不算入裸镜架宽度 |
| 配置高度 | 611 mm | 从台面到照明头顶部建立，编译后网格高度测量验证 |
| Eyepoint | 台面以上 449.4 mm，瞳距 64 mm | 作为观察筒布局参考；尚未计算出瞳或眼距，不能视为目镜光学校准 |
| 样本平面 | 当前台面以上 290 mm | 保持已有 SI 坐标；高度、支撑与样本嵌板为估算。可选 X-ASR＋AP114 真机 CAD 载物台单独核验，不宣称 ProScan／Nikon 载物台或完整镜体安装已验证 |
| HI-7 针座 | 总长 140 mm，适配 1 mm 外径玻璃管 | 参考[厂家页面](https://products.narishige-group.com/group1/HI-7/injection/english.html)和线虫论文补充材料实物图；玻璃暴露段另计 12 mm |
| HIR 旋转夹 | 7 × 7 × 18 mm，适配 4 mm 轴 | 参考[厂家页面](https://products.narishige-group.com/group1/HIR/injection/english.html)；压帽、夹具内孔、螺纹和密封细节为估算 |
| 电动微操 | 184 个 Zaber 原始 CAD 实体／套，25 mm 总行程 | 保留真实尺寸、固定／移动归属和三级滑台；未将其伪装为论文的 MP-285 或 MX7600 |

双针场景的原工具连接为 330–380 mm；新场景从 TCP 到 CAD 夹持参考位置为 180 mm，其中玻璃暴露段 12 mm、针座 140 mm、原创短连接 28 mm。机构朝向独立于针尖接近方向，运动命令通过实际滑台轴映射。原 openFrame 场景及原始记录保留自身结构。

贴壁目标仍为 36 × 28 × 8 µm，悬浮目标直径仍为 36 µm；相机、显微视野和整机外观的变化不扩大三维样本。具体视场与剂量边界见[论文与尺度](microscopy_paper_references.md)。

## 结构与运行

镜架选择、外观几何、针座、气路外观、滑台和任务分别组织。镜体选择由 `microscopy/stand.py` 处理；商用参考镜体和针座分别在 `commercial_stand.py`、`needle_holder.py`，双通道压力台与曲线气管在 `pneumatics.py`。压力响应、堵塞、吸持和流量仍由独立任务模型计算。

显式选择本场景：

可选电动载物台配置、安装尺寸及检查边界见[电动载物台](microscopy_stage.md)。

```bash
cd Hooke
export HOOKE_MICROSCOPY_ASSETS=cad
export HOOKE_MICROSCOPY_STAND=te2000-s-reference
export HOOKE_MICROSCOPY_OPTICS=estimated
export HOOKE_MICROSCOPY_ASSET_ROOT=/absolute/path/to/private/cad_meshes_surface_normals
```

`HOOKE_MICROSCOPY_STAND=auto` 保持按资产配置选择既有镜体；显式 `openframe` 需要 CAD。商业参考镜体拒绝 `mechanical`／`assembled` openFrame 光学装配，避免把不同镜架的接口混装。无私有 CAD 时可采用 `reference` 微操机构和本参考镜体；紧凑真实 CAD 双针安装只在上述 CAD 配置启用。

气管外观按实际编码器位姿更新：控制台端固定，针座后端跟随所属器械，两通道独立；曲线及后端接口为估算。它不计算弹性受力、接触或分布式管内流动，也不增加模型质量或控制量。针座外观几何为零质量；固定子体使用 1 mg、带重力补偿的数值质量占位，未使用未经测量的厂商物性。控制与碰撞仍采用声明的滑台和工具接触模型。

### 细胞针具的安装姿态与细颈

细胞任务的左吸持管／holder 与水平夹角现为 30°，右注射针／holder
为 35°。两套未缩放 CAD 微操机构随安装位置调整，平台相对原 45°
位置分别降低约 37.279 mm、24.035 mm；针座轴向与玻璃管轴向一致。
针尖外径 1.2 µm、内径 0.5 µm；后退 100／300 µm 处的外径为 2／4 µm，
随后过渡到后端 1 mm 玻璃管身。名义拉制轮廓同时供三维、合成显微
轮廓和流阻使用；无几何放大。吸持管口的 12 µm 外径用于吸持，与
穿刺针区别记录。安装连接、轮廓与压力参数仍为未标定的独立估算。
修正版及其独立证据位于忽略的 `pipette_mount_v1/`，详见[持续目标](microscopy_goal.md)。

## 已发现并保留的失败记录

初版左支撑柱与侧相机相交，已将支撑柱外移 45 mm、用短横向支撑连接原 CAD 基座。修正后的双针和三工具复位姿态三角面检查均无表面交叉、零跳过；这不覆盖总成内部配合、螺纹或任意运动。

新增针座子体最初引入未补偿质量，导致针尖受重力下沉；直接把子体全部质量置零又被 MuJoCo 编译器拒绝。最终把外观质量与动力学分开，添加上述明确声明的数值占位。轴测量测试在玻璃、细胞和对侧针之外进行；两针同时命令到同一 TCP 会发生接触，不适合作为空载轴测量。

`te2000_reference_v1`～`v3` 的图片、检查与失败日志均在忽略目录保留。后续完整回归使用独立 `te2000_reference_v4` 记录，不能将初版的图片或失败结果改名为最终验证。

最新验证和局域网入口见[持续目标](microscopy_goal.md)。整机外观已有可查看渲染，但细部、实机接口、完整光路和一比一验收仍待继续。

修正版重新解释 Nikon 官方外形图：135.4 mm 标注指向载物台前沿，
并非光轴距机身前沿。当前光轴距离 238.2 mm 为图示估算；前观察筒
与后照明柱据此重新定位。可选厂商载物台替换旧参考支撑，但原创
安装接口及部分全行程物镜间隙尚未验收，见[安装与限位](microscopy_stage.md)。

商用参考配置的贴壁与双侧吸持细胞任务已使用按 [Nikon MRH08430 官方尺寸图](https://www.microscope.healthcare.nikon.com/images/diagrams/Optics/Super-Plan-Fluor-Series/CFI-S-Plan-Fluor-ELWD-40XC_2.svg)独立重建的 40×／NA 0.6 物镜外轮廓。最大外径 34 mm，接口为简化 M25×0.75；玻璃厚度 1.2 mm，图纸名义工作距离 3.1 mm。60.41 mm 是图纸的具体基准，不等同于 CFI60 的名义齐焦距离 60 mm。六孔平面安装盘、旧参考物镜衬套及内孔是原创估算；保留三支未核验的旧物镜，没有重建原装转盘倾角和转动机构，不能称实机 CAD 或一比一。详见[物镜装配与成像边界](microscopy_optical_assembly.md)。原始尺寸图和测量记录仅保存在忽略的 `temp/`。

`te2000_reference_v4` 在 MuJoCo、原生 Isaac、实际浏览器分别完成五项任务，各后端通过 49 项流程检查。原生共 51,453 个实际步／事件、128 张 RGB；五条实际原生器械轨迹间距检查通过、零跳过。11 项本阶段相关独立测试通过，网页脚本错误为 0。实际渲染、录像与完整记录均位于忽略的 `temp/microscopy_demo/cad_integration/`，哈希与逐任务指标保存在 `te2000_reference_v4_verification_summary.json`。

五段新录像均完整解码通过，为 H.264、1680 × 760、4 帧／秒；逐段编码信息与哈希另存 `te2000_reference_v4_media_verification.json`。完整镜体、正面和针座近景各为 1280 × 960。局域网首屏实际打开并截图，其运行状态明确标注参考镜体和真实微米尺度。

局域网：[双环境显微平台](http://10.5.174.93:8084/microscopy?experiment=suction_injection)。网页可以选择 MuJoCo 或原生 Isaac，使用实际状态进行手动控制；部署和成像边界见[网页说明](microscopy_live_backends.md)。此前 8082／8083 单环境服务已关闭，原始试验记录与哈希保留。
