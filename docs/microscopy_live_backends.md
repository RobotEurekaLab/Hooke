# 显微网页双环境控制

All five public experiments now use micrometre-scale simulated cells. Pushing,
grasp/transfer and the original volume-controlled injection entry use the
[cell manipulation models](cell_manipulation.md); injection doses are in pL.
Earlier bead/chamber recordings are historical artifacts and are preserved.

显微实验页面可以选择 MuJoCo 或原生 Isaac／PhysX，使用相同的任务、器械资产和控制指令。三维全景、仪器近景和细胞近景来自所选环境自己的渲染器；显微视野由共享的合成成像模块读取该环境的实际状态生成。

## 使用

启动 Web 服务时分别指定两种环境可用的 GPU，以及忽略目录下的媒体位置。例如从仓库根目录执行：

```bash
cd Hooke
MUJOCO_GL=egl \
HOOKE_MICROSCOPY_ASSETS=reference \
HOOKE_MICROSCOPY_STAND=te2000-s-reference \
HOOKE_MICROSCOPY_STAGE=reference HOOKE_MICROSCOPY_OPTICS=estimated \
HOOKE_MICROSCOPY_MANIPULATOR=parallel-v4 \
HOOKE_MICROSCOPY_PARALLEL_ROOT=/本地许可明确的并联CAD转换目录 \
HOOKE_MICROSCOPY_MEDIA_ROOT=/仓库绝对路径/temp/显微网页记录 \
HOOKE_MICROSCOPY_GPU=3 HOOKE_ISAAC_GPU=6 \
HOOKE_MICROSCOPY_LIVE_FPS=1 HOOKE_ISAAC_SAMPLES_PER_FRAME=16 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../.venv/bin/python -m webui.server --host 0.0.0.0 --port 8084
```

真实并联 CAD 的下载、保存快照转换和许可见[准备说明](microscopy_parallel.md)。没有转换网格时，去掉上述资产、镜体、载物台、光学和并联机构选择变量，以默认参数化模型运行。Isaac 需要已有的本地安装；可用 `HOOKE_ISAAC_PATH` 指定目录。安装路径及权限以运行网页后端的 Linux 账号为准，其他账号部署见[运行说明](isaac_operations.md#其他账号与服务器部署)。

本机当前服务使用 `reference` 载物台、`parallel-v4` 真实机构和 `parallel_v4_production_v1/` 媒体。
五类流程在 MuJoCo、Isaac 各通过 49 项判据；二十个 MP4／WebM 均来自
当前流程，完整解码和分段读取核验通过，十个 WebM 实际播放通过。局域网入口为
[显微实验平台](http://10.5.174.93:8084/microscopy?experiment=suction_injection)。
总成全行程及实机一比一没有验证；此次 demo 的逐项完成指标与边界见[目标验收](microscopy_goal.md)。

主服务 Source GPU 3、Native GPU 6；8085 检查服务 Source GPU 2，使用独立
媒体目录。GPU 编号按本机可用资源配置，不停止其他训练任务。当前细胞配置
另包含可见电动调焦安装链，接口、载荷与估算边界见[装配说明](microscopy_optical_assembly.md#可见电动调焦安装链)。

商用参考细胞场景增加“采集光路近景”，相机位于实际空腔内；不隐藏
镜体壁面或替换成剖面图片。它展示原创名义管镜、折转镜及相机接筒，
不证明完整光学成像或原厂内部结构。宏观任务没有该光路配置，页面禁用
这个视角，接口拒绝使用遗留图片。尺寸依据见[原创采集路径](microscopy_optical_assembly.md#原创采集路径)。

当前装配另完成各后端十条失败对照，覆盖堵针、过深、未穿刺、退针后
施压和吸持失效；当前原生对照共 16,800 个实际步／事件。对照用目标真值
布置故障，属于降阶模型行为验证，不能当作生物标定或自主定位实验。

细胞场景的左吸持管／holder 水平夹角为 30°，右注射针／holder 为
35°；网页显示夹角和注射针尖外径 1.2 µm、内径 0.5 µm。三维、合成
显微轮廓及流阻共用同一分段拉制轮廓；针形与安装仍为独立估算。

本地载物台检查配置另设置 `HOOKE_MICROSCOPY_STAGE=x-asr100`，使用
X-ASR 与 AP114 的 40 个原尺寸 CAD 实体；资产、运动层级、许可状态及
安装核验范围见[电动载物台](microscopy_stage.md)。修正版已消除名义
姿态中的镜体交叠，但部分完整行程位置会碰到物镜；当前操作范围
限制为 X/Y 各 ±4 mm，尚未实现物镜粗调退让。默认仍为 `reference`。
载物台和调焦输入范围由控制接口返回，并以 µm 显示；原厂硬件
行程与当前允许操作范围分别保存在 `actuator_limits_m` 和 `command_limits_m`。

打开 `/microscopy?experiment=suction_injection`，选择环境与任务，再使用调焦、微操轴、夹爪和压力按钮，或点击“运行实验”。URL 可追加 `backend=isaac` 直接选择 Isaac。首次原生启动和场景转换需要等待，期间页面显示载入状态；失败会显示实际错误，不会自动切换环境。

每个控制请求推进指定的模拟时间。请求结束后暂停，没有后台实时时钟；完成流程后的录像可独立回放。采样率是每模拟秒保存的画面数量，不能作为交互响应速度或实时性能指标。

## 重置与画面同步

重置和完整实验都会用指定种子重新建立场景，包括静态细胞位置、关节初值和共同的简化实验系统。原生场景重新导出后载入同一个 Isaac 进程，避免只重置关节留下旧细胞位置。

切换环境会关闭原会话及其渲染器，再启动所选环境。单个 Web 服务只接受一个并行控制请求；实验运行时可以读取进度和画面，其他控制请求返回忙碌。多个服务须使用不同媒体目录；共用原生 GPU 时须串行使用，统一 GPU 锁会拒绝并发启动。

MuJoCo 和 Isaac 的实时输出分别位于 `live/` 和 `live-isaac/`。每次启动使用独立会话标识，每次重建场景使用新序号，原始场景与观察记录保存于 `sessions/<会话>/<场景序号>/`。状态包含 `backend`、`physics_engine`、`world_image_source`、`microscope_image_source` 和原生步进计数。

## 录像

原生控制请求还将本次场景的每一步实际 PhysX 关节观测保存为
`sessions/<会话>/<序号>/trajectory.npz`。结果中的 `native_trace` 记录
SHA-256、步数、实际物理事件数和运动学误差；归档前要求步数与事件数
相同。轨迹与稀疏渲染画面分开，录像不能替代完整运动轨迹证明。
离线间隙检查读取这些观测关节及已校验哈希的源模型，只计算几何
位姿，不再运行 MuJoCo 物理积分。

录像只能使用对应环境的已保存观察画面，不启动第二套物理仿真。将一条完成的会话编码为录像：

```bash
cd Hooke
../.venv/bin/python -m microscopy.recording \
  --episode /媒体目录/live-isaac/sessions/会话/场景序号 \
  --output /媒体目录/recordings/新版本/isaac/suction_injection/demo.mp4 --fps 4
../.venv/bin/python -m microscopy.recording \
  --episode /媒体目录/live-isaac/sessions/会话/场景序号 \
  --output /媒体目录/recordings/新版本/isaac/suction_injection/demo.webm --fps 4
```

MuJoCo 录像位置为 `/媒体目录/<任务>/demo.mp4`，Isaac 为 `/媒体目录/isaac/<任务>/demo.mp4`；WebM 放在同一任务目录。网页提供 WebM／VP9 和 MP4／H.264 两种来源，由浏览器选择支持的格式；缺失时不使用另一种环境的录像替代。接口通过 `backend=mujoco|isaac` 与 `format=mp4|webm` 指定来源。

### Publish updated recordings

Live image updates do not regenerate saved videos. Encode each completed episode
into a new ignored release directory, preserving earlier videos and their hashes.
The layout is `recordings/<revision>/<backend>/<operation>/demo.mp4` and
`demo.webm`; each encoded file has a `.json` provenance report. Both backends and
all five operations must be present before selecting the release:

```bash
cd Hooke
../.venv/bin/python -m microscopy.recording_catalog \
  --media /absolute/repository/temp/microscopy-media \
  --release /absolute/repository/temp/microscopy-media/recordings/new-revision
```

Publication verifies all 20 video hashes and their backend/operation metadata,
then atomically updates `recordings/current.json`. The server reads this pointer
on each recording request; a server restart is unnecessary after publication.
Without a pointer, the legacy paths above remain available. An incomplete selected
release returns an error instead of silently showing older recordings.

The replay selector starts with the current experiment and follows experiment
changes. The browser obtains the revision from `/api/microscopy/recordings`, uses
it in video URLs, and revalidates cached video responses. Refresh an already open
page after publication. New videos use the saved instrument closeup when present,
with the microscope view alongside it and a separate synthetic fluorescence inset
for cell experiments. Cell recordings retain phase contrast as the primary view.

编码器保持原始画面比例，按观察记录的模拟时间重复已有帧，不插值生成动作。视频标注环境、任务、阶段与实际观察时间；旁边的显微图仍是合成成像。编码 FPS 可以高于实际采样率，但不会增加观察信息。视频摘要保留输入画面和状态的 SHA-256。

## 尺度与模型边界

贴壁细胞名义尺寸为 36 × 28 × 8 µm，悬浮细胞直径为 36 µm，注射针尖外径／内径为 1.2／0.5 µm。细胞合成视场为 160 µm／768 像素，20 µm 标尺对应 96 像素。三维单位为米，显示增益为 1；局部观察通过相机放大。1.2 mm 微珠单独归为机械标定样本，不能作为细胞演示。

商用 CAD 配置下的气管读取实际针座位姿：控制台端固定，器械端跟随编码器反馈；两通道分别生成连续曲线。它们是无碰撞、无质量的外观，不模拟软管受力或分布式管内流动。

悬浮细胞使用平滑椭球网格和示意透明着色。RTX 采用 [OmniPBR 的分数透明度](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniPBR_Opacity.html)，避免把细胞显示成折射玻璃球；保留通用 PreviewSurface 输出。透明度和粗糙度是显示参数，不能当作细胞折射率或光学标定。完全不可见的源几何体在 USD 中也隐藏，碰撞属性仍保留。

PhysX 运行器械关节与刚体接触；细胞膜、吸持、示踪和液体沿用明确标注的简化模型。当前不具备原生软体膜、完整波动光学、生物标定、实机纳米精度或整机一比一核验。商用镜体与针座的已知尺寸、估算项及许可见[商用参考场景](microscopy_commercial_reference.md)。持续目标与分阶段证据见[目标记录](microscopy_goal.md)。

## 大离焦图像计算

细胞图像仍采用未生物标定的独立平面高斯离焦模型。大核 float64、零填充
卷积使用相同采样与截断范围的 FFT 算法，保留正常浮点舍入；小核、其他
数据类型、非零边界和非有限输入继续使用 SciPy 直接实现。缓存仍按完整
输入与参数取键，载荷上限保持 64 MiB，不量化焦点、坐标或剂量。

`focus_mount_v1/blur-image-equivalence/summary.json` 的三个独立模型状态
（0／4／100 µm 离焦）完整 RGB 逐像素一致；100 µm 状态的一次冷图像
计算约 97.94 秒降至 7.95 秒。单个 1537 × 1537 × 3 平面卷积约 10.22 秒
降至 1.21 秒、最大绝对误差约 1.22e−17。这里记录的是特定输入和运行
环境的测量，不能当作任意输入的位级等价或整套网页实时性能保证。
