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

`doctor` 只读取依赖、安装路径、GPU UUID、显存和驱动，不安装软件、不启动仿真。`files_ready` 只表示必需文件存在，不能代替实际原生回合验证。

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

Native 启动前运行 CPU 能力检查。未知插件、球关节、柔性体、动态地形、任意非线性执行器和其他未实现特性会明确拒绝。静态地形、刚体 wrench、焊接约束等扩展以独立原生小场景的实际报告为准，能力清单 `can_attempt_native` 不是物理等价证书。
