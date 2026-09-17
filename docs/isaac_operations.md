# 双仿真环境运行与维护

## 已验证的机器配置

本机使用 MuJoCo 3.3.0、Isaac Sim 4.5 和现有 NVIDIA 535.230.02 驱动。本分支不升级驱动，Isaac 使用独立 Python 3.10 进程，Hooke 使用原有 `.venv` Python 3.12。两套 Python 环境通过本机私有 Unix socket 同步实际状态。

从仓库下的 `Hooke/` 目录运行命令，确保原 SDF 插件的相对路径正确：

```bash
cd Hooke
export HOOKE_ISAAC_PATH=/home/amax/data/datacopy/isaacsim
export HOOKE_ISAAC_GPU=6
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
../.venv/bin/python -m backends.doctor --gpu 6 --output ../temp/doctor.json
../.venv/bin/python -m webui.server
```

网页：`http://服务器局域网地址:8080/backends`。可以选择相同目录任务，分别打开两个后端。默认 GPU 为 6，也可以在未跟踪的 `temp/isaac_local.json` 中配置 `isaac_path` 和 `gpu`；环境变量优先。使用其他机器时自行填写安装路径和空闲 GPU，不复制本机账号、SSH 密钥或驱动。

也可从任意目录启动，脚本自动进入源码目录，避免插件相对路径错误：

```bash
/path/to/Hooke/scripts/start_hooke_backends.sh --doctor --gpu 6
/path/to/Hooke/scripts/start_hooke_backends.sh --host 0.0.0.0 --port 8080
```

默认使用仓库 `.venv/bin/python`，可用 `HOOKE_SOURCE_PYTHON` 指定已有的 Source Python 可执行文件。脚本不创建环境或安装依赖。

`doctor` 只读取依赖、安装路径、GPU UUID、显存和驱动，不安装软件、不启动仿真。`files_ready` 只表示必需文件存在，不能代替实际原生回合验证。

诊断记录 MuJoCo、NumPy、SciPy、JAX、TOPPRA、Pillow、Flask 的实际版本及本地 `meshplane` 模块是否可发现；缺少依赖时仍输出其他检查，`environment_files_ready=false`。Source 的安装说明见 [模拟器 README](../Hooke/README.md)，本机 Python 3.12 已实测。原 SDF 和 `meshplane` 为本地二进制组件，复制文件或找到模块不能代替其他 Python/平台上的 ABI 验证。Isaac 使用其独立安装环境。

演示导出和策略客户端另需 `packages/autobio-inference/pyproject.toml` 已声明的 `msgpack`、`websockets`。在上述 Source 环境中补齐缺失依赖：

```bash
../.venv/bin/python -m pip install --no-deps 'msgpack>=1.0.5' 'websockets>=11.0'
```

## 实际物理验证

```bash
../.venv/bin/python -m backends.run --task centrifuge_5430_cycle \
  --backend isaac --mode expert --seed 0 --gpu 6 --no-render \
  --output ../temp/cycle-native
../.venv/bin/python -m backends.run --task centrifuge_5430_cycle \
  --backend mujoco --mode expert --seed 0 --gpu 6 --no-render \
  --output ../temp/cycle-source
```

新增完整离心流程使用原机器人、试管、转子、盖板和相机；新增转子电机与样品固定约束。默认 60 RPM、保持 1 秒，用于机械控制验证，不能据此宣称真实离心分离已经标定。机械成功要求真实转动、载荷保持、配平、制动、停止和安全解锁。

`result.json` 保存最终状态、原任务谓词、独立逐步操作判定、时限、科学模型适用范围和实际耗时。`progress.json` 保存运行中的状态。`trajectory.npz`、`source/` 和 `task_log/` 用于回溯原状态、控制器、液面和资产。占位 `check=True` 的展示结果不计为科学实验成功。

## 简化科学模型

网页“运行内容”可选择热学和毛细管流体实验；其他任务默认保留原机械逻辑。这两个实验的温度与压力边界由实验配置给定，不能解释为机器人已完成设定边界的动作。

```bash
../.venv/bin/python -m backends.run --task thermal_mixer \
  --backend isaac --mode no_action --seconds 90 --science-model thermal \
  --no-render --output ../temp/thermal-native
../.venv/bin/python -m backends.run --task pipette_transfer \
  --backend isaac --mode no_action --seconds 5 --science-model capillary \
  --no-render --output ../temp/flow-native
```

热学是满足 Biot 数约束的两节点模型，记录输入、损失和储存能量。流体是给定压差下、充满液体的稳态层流毛细管模型，记录雷诺数、入口长度和容量约束，直接驱动原容器液面。参数未经过真实仪器标定，流体质量没有反馈到机械惯量；不支持气液两相、自由液滴、蒸发、空间 CFD 或化学反应。

## 显示与性能设置

共享设置：`HOOKE_RENDER_WIDTH`、`HOOKE_RENDER_HEIGHT`、`HOOKE_RENDER_FPS`。默认 640×480、20 FPS。网页科学模型实验使用 1 FPS。配置进入结果记录，时间轴按实际采样率显示。

Native 光照：`HOOKE_ISAAC_SUN_INTENSITY`、`HOOKE_ISAAC_AMBIENT_INTENSITY`。`HOOKE_ISAAC_COLOR_PIPELINE=source_display` 为可选源显示颜色配置，使用原天空纹理、sRGB 颜色转换和线性色调映射；命令行默认 `physical` 保留原生材质响应，网页 Native 默认使用 `source_display`。真实图像误差以对应验证报告为准。

本机同步默认 `HOOKE_ISAAC_TRANSPORT=binary`，可设置 `json` 诊断。二进制格式只用于同一用户启动的私有进程，socket 权限为 0600、消息大小受限，不能用于接收远程或用户上传的序列化对象。代码不删除接触点、不用第二套 MuJoCo 积分代替 PhysX，不将两套引擎宣称为完全等价。

## 恢复与故障排查

```bash
../.venv/bin/python -m backends.matrix --task pipette_transfer \
  --backend isaac --mode expert --mode no_action --seeds 0 1 2 \
  --control-seconds 45 --wall-seconds 1200 --output ../temp/transfer-matrix
# 中断后以相同代码、环境和参数追加 --resume
```

回归只复用代码、场景和参数完全匹配的记录，保留失败及中断文件。遇到 GPU 占用提示，等待运行中的任务结束；不跳过显存检查或 GPU 锁。启动或请求失败时查看该回合的 `isaac-worker.log` 和 `process.log`。运行器会回收自己创建的进程和 socket，网页“停止”只停止对应任务。服务重启后，旧未完成任务显示为中断。

原场景预览、机器人替换预览、MuJoCo 回放和 Isaac worker 共用同一 GPU 文件锁。网页在创建任务前检查锁，外部命令行占用时返回 409 和明确原因；网页任务之间也按共享 GPU 串行运行。子进程启动后仍独立获取锁，防止检查与启动之间的并发竞争。MuJoCo 无渲染 CPU 回归不占该锁。

Native 启动前运行 CPU 能力检查。未知插件、球关节、柔性体、动态地形、任意非线性执行器和其他未实现特性会明确拒绝。静态地形、刚体 wrench、焊接约束等扩展以独立原生小场景的实际报告为准，能力清单 `can_attempt_native` 不是物理等价证书。

## 求解器与实际稳定性

`HOOKE_ISAAC_SOLVER_TYPE` 支持 `TGS`（默认）或 `PGS`；`HOOKE_ISAAC_EXTERNAL_FORCES_EVERY_ITERATION=1` 为 TGS 的可选外力迭代设置，默认关闭。设置进入实际物理配置报告。它们改变求解行为，不能把一种设置的验收结果当成其他设置已通过。

本轮默认 TGS 的静止落球位置不漂移，但返回约 5.94 mm/s 的竖直残余速度，未通过 2 mm/s 门限。TGS 开启逐次迭代外力后，在相同门限下通过 120 秒、60,000 步和故障构造恢复，残余速度不超过 0.013 mm/s。没有改写位置、速度或提高门限；完整目录任务仍保留其冻结设置。[失败、诊断与复测记录](validation/isaac_lifecycle_summary.json)。NVIDIA 的 [求解器配置说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/dev_guide/simulation_control/simulation_control.html) 建议在 TGS 中考虑逐次迭代外力以改善收敛；这里的通过结论来自本机小场景测量。

```bash
HOOKE_ISAAC_EXTERNAL_FORCES_EVERY_ITERATION=1 ../.venv/bin/python \
  -m backends.lifecycle_contracts --gpu 6 --output ../temp/native-lifecycle
```

新 `info` 与关键帧报告使用独立 PhysX 步进回调，区分 Kit 内部初始化、已恢复的回合起点和渲染。当前小场景与离心回放初始化实际为 2 步，随后恢复源位置和速度；恢复后的图片渲染为 0 个物理事件。旧报告的零运行器计数不能证明没有内部初始化步骤，历史文件保留原值并补充解释。
