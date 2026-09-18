# 电动 XY 载物台

默认参考载物台仍可独立运行。可选 `x-asr100` 组件使用厂商发布的
X-ASR100B120B-SE03D12 STEP 和 AP114 适配板 STEP，全部网格保存在忽略的
`temp/` 中。公开下载不等于开源许可；目前未确认 CAD 的明确复用授权，
该组件用于本地资产检查，不随仓库分发。

2026-09-18 再次核对[官方 CAD 下载页](https://www.zaber.com/products/scanning-microscope-stages/X-ASR-E/documents)，
可见 STEP 和 SolidWorks 下载，但未找到这些机械模型的明确复用
许可。手册中软件组件的 MIT/BSD 条款及
[Software Portal 条款](https://software.zaber.com/tos/)不能当作这些产品
CAD 的授权依据；目前也不宣称学术用途已获许可。

**整机及全行程安装尚未验收。** 修正安装方向、旧支撑及镜体尺寸
解释后，当前版本在名义姿态未发现厂商实体与参考镜体的表面交叠。
但全行程抽样发现部分端点会与下方物镜交叠，因此保留厂商行程，
并将当前控制接口的 X/Y 操作范围限制为各 ±4 mm。没有实现物镜
粗调退让，不能直接使用全部硬件行程。

初版的 13 组交叠结果仍保留在
`temp/microscopy_demo/cad_integration/stage_cad_v1/stand_nominal_clearance/stage-stand-surfaces.json`。
它不等于 13 个独立实体，也不是 BRep 穿插体积测量。新版本使用独立
`stage_cad_v3/` 记录，不覆盖旧失败证据。

## 几何与运动

厂商原始 SolidWorks 文件的 `ASR_base`、`ASR_mid`、`ASR_top` 虚拟组件
特征标签，结合 STEP 的导轨位置，用于划分固定底座和两层滑台。38 个
载物台实体和 2 个适配板实体全部使用米制原尺寸网格，不缩放。
控制器及两侧外壳属于中层滑台，随第一轴运动。

| 层级 | STEP 实体数 | 世界坐标运动 | 当前范围 |
| --- | ---: | --- | --- |
| 固定底座 | 10 | 固定 | — |
| 下层滑台 | 23 | `stage_y` | 硬件 ±50 mm，当前操作 ±4 mm |
| 上层滑台及样本 | 5 | `stage_y` + `stage_x` | 第二轴硬件 ±60 mm，当前操作 ±4 mm |
| AP114 适配板 | 2 | 固定 | — |

厂商给出的下层和上层行程分别为 100 mm、120 mm。
仿真零位使用 STEP 名义姿态作为中心，未复现厂商回零和限位标定。
驱动继续使用已有的刚性滑轨和位置伺服模型；部件质量、惯量、速度、
微步分辨率及定位误差未按真机标定。
[厂商规格](https://www.zaber.com/products/scanning-microscope-stages/X-ASR-E/specs?part=X-ASR100B120B-SE03D12)。

AP114 绕 CAD 竖直轴旋转 90° 后，无缩放、无平移即可对齐六个 M3 孔轴。
原始 STEP 的名义姿态检查包含 76 组载物台与适配板实体组合，未发现
超过 `1e-5 mm³` 的体积穿插。结果仅覆盖这两份 CAD 的名义装配，
不证明 Nikon 镜体安装、螺纹配合、制造公差或完整行程间隙。
[AP114 资料](https://www.zaber.com/products/accessories/AP114)。

另检查下层 ±50 mm、上层 ±60 mm 的中心、端点和四角，共九个抽样
姿态、504 组移动件与适配板实体组合，同一阈值下没有体积穿插。
这是离散姿态检查，不证明连续运动；镜体、样本嵌板和器械未包含在
这项检查中。记录为 `ap114-travel-samples.json`。

160 × 110 mm 开口样本嵌板、镜体支撑和紧固件为独立参考建模。
嵌板支撑现有玻璃底，样本基准面保持不变。贴壁细胞仍为
36 × 28 × 8 µm，悬浮细胞直径仍为 36 µm；显微成像仍使用
160 µm 视场和 20 µm 比例尺。画面放大由相机完成，几何显示倍率为 1。
整机一比一还原及生物力学标定尚未验收。

## 安装方向与运动空间

安装方向参考 [AP114 安装图](https://www.zaber.com/fs/accessories/AP114/docs/AP114.pdf)：
控制器在右侧，适配板位于前后。样本通过最终滑台的 `sample_plane`
标记挂接，控制器仍使用世界坐标 X/Y，不依赖内部滑台先后顺序。

Nikon TE2000-S 侧视图的 135.4 mm 标注指向载物台前沿，不能当作
光轴到机身前沿的距离。当前以图示估算该光轴距离为 238.2 mm，并
据此修正前观察筒及后照明柱的位置；这个距离仍是估算项。
[官方外形图，第 24 页](https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf)。
厂商载物台和适配板整体前移 20 mm，原始 CAD 尺寸不变。原创嵌板
使玻璃底及细胞继续位于原光轴基准上。

### 当前运动空间检查

40 个厂商实体对 257 个参考镜体几何体的名义姿态检查通过，最近
表面距离约 0.583 mm。九个完整 XY 行程抽样姿态对镜体也未发现
表面交叠；另对物镜的九姿态检查中，三个 `stage_y = -50 mm`
姿态存在交叠。两项检查的对象不同，不能合并宣称完整行程通过。

对当前 ±4 mm X/Y、±1 mm 调焦操作范围，按各刚性层的名义表面
距离减去最大相对平移量计算保守下界。载物台及嵌板对物镜的最小
下界约 3.07 mm，移动件对原创支撑的最小下界约 11.06 mm；固定
厂商件对镜体的 0.583 mm 间隙不随 XY 移动。该方法覆盖所声明的
平移范围，而非仅检查端点。它不检查封闭实体包含、固定螺栓接触、
嵌板座配合、其他器械轨迹或制造公差。

原始报告位于 `stage_cad_v3/stand_nominal_clearance/` 的
`stage-stand-surfaces.json`、`stage-stand-travel-samples.json`、
`stage-objective-travel-samples.json` 和 `operating-envelope-surface-bounds.json`。
控制请求超出当前操作范围时，在压力改变及物理步进前返回错误；
`actuator_limits_m` 保留硬件行程，`command_limits_m` 返回当前输入范围。

## 本地启用

```bash
export HOOKE_MICROSCOPY_ASSETS=cad
export HOOKE_MICROSCOPY_STAND=te2000-s-reference
export HOOKE_MICROSCOPY_OPTICS=estimated
export HOOKE_MICROSCOPY_STAGE=x-asr100
export HOOKE_MICROSCOPY_ASSET_ROOT=/absolute/private/cad_meshes
```

私有资产目录需包含 `x-asr100/`、`ap114/` 及电动微操手所需资产。
各组件须有转换后的 `manifest.json` 和米制 OBJ。加载器检查 STEP
来源哈希、网格哈希和实体数。显式请求该组件时，缺少资产或使用
未经验证的镜体组合会报错。

运行状态包含 `stage_profile` 和米制执行器范围。界面根据实际模型更新
输入范围，再以 µm 显示；`stage_y` 上的玻璃底和细胞随真实关节反馈
一起移动。相同场景由 MuJoCo 与原生 Isaac 使用。

## 原始核验记录

- 原始 STEP、PDF、SolidWorks 及来源哈希：
  `temp/microscopy_research/stage_cad/mount_research_v2/`。
- 六孔及 BRep 检查：该目录下 `ap114-mount-audit.json`。
- 编译尺寸和独立运动测试：`tests/test_microscopy_stage.py`，私有
  CAD 测试通过 `HOOKE_TEST_STAGE_ASSET_ROOT` 指定资产目录。
- 新截图、运行记录和录像单独写入
  `temp/microscopy_demo/cad_integration/stage_cad_v1/`，保留此前记录。

## 40× 细胞工位的装配范围

新的商用参考细胞工位采用 1.2 mm 玻璃，上表面保持样本零位；原创嵌件安装面位于 −1.2 mm，嵌件厚 1.8 mm，横支撑比原参考配置降低 10 mm。40 个原厂载物台／适配板 CAD 实体、轴层级及工厂行程保持原尺寸。宏观微珠任务保留原配置。

包含新物镜、三支旧物镜、衬套和六孔平面盘的 27 件物镜总成参与检查。载物台与镜体／物镜／支撑的八项操作范围保守表面下界均为正；对物镜最小约 2.927 mm。完整行程九姿态中仍有三个 Y=−50 mm 姿态相交，X/Y ±4 mm、调焦 ±1 mm 限位继续生效。物镜总成与支撑的调焦范围下界约 3.618 mm；固定安装接触、实体包含、螺纹配合和制造公差未验收。数据在忽略的 `objective_40x_v2/reset_clearance/`，操作范围使用包含衬套的扩展检查；失败的早期装配另存于 `objective_40x_v1/`。[物镜说明](microscopy_optical_assembly.md)区分原始尺寸和估算件。
