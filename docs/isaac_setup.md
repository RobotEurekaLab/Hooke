# Isaac 安装、自动发现与多用户部署

本文适用于当前 Hooke 的 Linux 独立安装适配器，已验证版本为 **Isaac Sim 4.5**。
Isaac 使用安装包自带的 Python；Hooke 使用自己的 Python 环境。
两者不需要安装到同一个虚拟环境。其他版本、pip 安装或 Isaac Lab 的目录
不能仅凭名字认定为当前适配器可用的独立安装。

## 先确定谁在运行服务

Isaac 路径、GPU 和文件权限属于**运行 Hooke Web 后端的 Linux 账号**。
访问者的浏览器账号、GitHub 账号和登录网页的设备不改变这份配置。

| 使用方式 | 配置位置 |
| --- | --- |
| 其他人访问已启动的 Hooke 网页 | 共用服务端配置，无需每位访客安装 Isaac |
| 每个人在自己的 Linux 账号启动 Hooke | 每个服务账号分别配置可访问的安装 |
| 在另一台服务器启动 Hooke | 在那台服务器安装／配置 Isaac，Git 克隆不带安装包 |
| systemd 或容器启动 Hooke | 使用该进程的账号、环境变量和进程内路径 |

在 `/backends` 或 `/microscopy` 展开“服务器环境与配置帮助”，查看服务账号、
实际安装目录、配置来源和失败位置。也可以请求：

```bash
curl -s http://服务器地址:8080/api/backends/environment
```

“安装在 `/home/dongsu/humanoid/isaacsim`”本身不是不可用的证据。
需要同时确认路径已被选择、服务账号可访问、必要文件完整和运行时可启动。

## 1. 安装 Isaac

已有完整安装时直接跳到下一节。没有安装时，从
[NVIDIA 4.5 下载入口](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html)
选择对应的 Linux x86_64 standalone 包。下面沿用
[官方工作站安装说明](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html)
中的目录和安装后处理方式：

```bash
mkdir -p ~/isaacsim
unzip ~/Downloads/isaac-sim-standalone-4.5.0-linux-x86_64.zip -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
```

下载文件名不同则修改命令。服务器通过 Hooke 网页操作，无需打开 Isaac 桌面窗口。
先查阅[官方硬件／驱动要求](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)；
本项目不会自动安装、升级或降级驱动。本机已有 535.230.02 驱动已实测，
不据此保证另一台机器的硬件兼容性。

安装根目录应包含 `python.sh`、`setup_python_env.sh`、
`kit/python/bin/python3` 和 `kit/libcarb.so`。Hooke 会检查这些文件的读取／执行权限。

## 2. 自动发现和保存配置

**没有显式路径配置时，Hooke 会自动使用找到的唯一可访问安装。**
默认查找服务账号的 `~/isaacsim`、`~/isaac-sim`、`~/humanoid/isaacsim`、
常见版本目录、旧的 `~/.local/share/ov/pkg/isaac-sim-*`，以及
`/opt/isaacsim`、`/opt/isaac-sim*`、`/opt/nvidia/isaac*`、
`/usr/local/isaacsim`、`/isaac-sim` 等共享位置。
找到多个完整安装时会列出候选，要求明确选择；发现过程不执行候选程序、不启动 GPU。

从仓库根目录执行以下命令。脚本默认使用 `.venv/bin/python`；如果按 README
创建了 Conda 环境，先激活环境并设置：

```bash
export HOOKE_SOURCE_PYTHON="$(command -v python)"
```

只查看候选，不写配置：

```bash
./scripts/start_hooke_backends.sh --configure-isaac --discover
```

将唯一发现的安装保存到当前服务账号，GPU 编号按本机空闲设备选择：

```bash
./scripts/start_hooke_backends.sh --configure-isaac --auto --gpu 0
```

非标准目录可以增加搜索根；可重复传入 `--search-root`。它最多搜索三层子目录、
256 个目录，达到上限会提示缩小范围，不从不完整搜索结果中自动选择：

```bash
./scripts/start_hooke_backends.sh --configure-isaac --discover \
  --search-root /home/dongsu/humanoid
./scripts/start_hooke_backends.sh --configure-isaac --auto \
  --search-root /home/dongsu/humanoid --gpu 0
```

已知道准确目录时可直接配置，避免搜索其他安装：

```bash
./scripts/start_hooke_backends.sh --configure-isaac \
  --isaac-path /home/dongsu/humanoid/isaacsim --gpu 0
```

保存前验证文件和权限，成功后原子写入 `~/.config/hooke/isaac.json`，权限为 0600。
设置了 `XDG_CONFIG_HOME` 时写入该目录下的 `hooke/isaac.json`。
配置留在仓库外，切换 checkout 后仍可使用。自动搜索／选择失败时不修改原有配置。

路径优先级为：`HOOKE_ISAAC_PATH` → 账号配置 → 旧的 `temp/isaac_local.json`
→ 自动发现。找不到可访问安装时返回各候选的缺失／权限诊断。GPU 使用环境变量／账号配置／旧配置的
同样优先级，未设置时为 0。显式路径无效时会报错，自动发现不会悄悄替换它；
检查旧环境变量或显式运行配置命令。`--auto` 是明确重新发现并保存的操作，
不会清除优先级更高的环境变量。

## 3. 其他账号目录的权限

配置命令须由**实际服务账号**执行。用该账号先检查：

```bash
id
namei -l /home/dongsu/humanoid/isaacsim/python.sh
getfacl /home/dongsu /home/dongsu/humanoid /home/dongsu/humanoid/isaacsim
```

服务账号需要穿过各级父目录，并读取安装文件、执行启动器和自带 Python。
自动发现不会修改其他账号权限。权限不足时有两种处理方法：

1. 在服务账号自己的 `~/isaacsim` 安装，随后用 `--auto` 配置。
2. 安装所有者或管理员为服务账号授予必要访问权限，或部署受控的共享安装。

对于上面的具体路径，安装所有者／管理员可使用 ACL 仅授权所需账号。
下面的 `hooke_service_user` 必须改为网页显示的真实服务账号：

```bash
hooke_service_user=实际服务账号
sudo setfacl -m "u:$hooke_service_user:--x" /home/dongsu /home/dongsu/humanoid
sudo setfacl -R -m "u:$hooke_service_user:rX" /home/dongsu/humanoid/isaacsim
```

这些命令授予父目录遍历和安装读取／必要执行权限；管理员需要核对目录所有权、
现有 ACL 及符号链接指向的其他路径。随后由服务账号重新执行配置／检查命令。
若 Kit 扩展或缓存另有写入要求，以实际 worker 日志为准；文件访问检查不证明
只读共享安装能完成初始化。每账号独立安装更容易隔离缓存和扩展变更。

## 4. 启动和验证

从仓库根目录执行：

```bash
./scripts/start_hooke_backends.sh --doctor
./scripts/start_hooke_backends.sh --host 0.0.0.0 --port 8080
```

`doctor` 检查安装访问、依赖和 GPU，不启动 Isaac。
`installation_ready=true` 表示必要文件可访问；`native_runtime_checked=false`
明确表示尚未验证 Kit／PhysX。打开 `/backends`，在 Isaac 面板打开一个支持的场景，
确认出现实际画面和物理步数。首次启动需要加载扩展和预热缓存。

需要独立命令行运行检查时，从源码目录执行（每次使用新的输出目录）：

```bash
cd Hooke
python -m backends.run --task pipette_transfer --backend isaac \
  --mode no_action --seconds 0.2 --no-render --output ../temp/isaac-smoke
```

这个检查验证运行时与物理步进；网页场景另验证 RTX 渲染。出现 GPU 忙或 GPU 锁冲突时
换用空闲 GPU 或等待任务结束。运行失败查看对应输出目录的 `isaac-worker.log`／
`process.log`；安装配置失败则先按页面的 `error_code` 和 `blocked_path` 修正。

## 5. systemd 和容器

终端中的 `export` 不会修改已经运行的 Web 服务。systemd 必须将设置放到实际服务中，
例如以下 `[Service]` 配置；账号、仓库、Python 和安装路径按部署修改：

```ini
[Service]
User=hooke
WorkingDirectory=/srv/Hooke/Hooke
Environment=HOOKE_SOURCE_PYTHON=/srv/Hooke/.venv/bin/python
Environment=HOOKE_ISAAC_PATH=/srv/isaacsim
Environment=HOOKE_ISAAC_GPU=0
ExecStart=/srv/Hooke/scripts/start_hooke_backends.sh --host 0.0.0.0 --port 8080
```

修改 unit 后由服务管理员执行 `systemctl daemon-reload` 并重启对应服务。
检查 `ProtectHome`、挂载隔离和实际 `User`：即使普通终端可以访问，服务隔离也可能隐藏安装。
账号配置修改后，新 worker 读取新配置；已有 worker 继续使用其启动时的安装和 GPU。

容器里的安装路径必须真实存在于容器内部；主机路径需挂载进去，且容器 UID 有权限、
GPU 已透传。设置 `HOOKE_ISAAC_PATH` 为容器内路径。宿主机的安装、账号配置或
驱动可见，并不自动证明容器运行时可用；在容器内执行同样的检查和实际运行验证。

## 排错速查

| 现象／错误码 | 处理 |
| --- | --- |
| `installation_missing`／`installation_not_found` | 检查自动候选；用 `--search-root` 或 `--isaac-path`；没有安装则先安装 |
| `installation_ambiguous` | 从返回候选中选择一个，显式设置 `--isaac-path` |
| `discovery_incomplete` | 缩小搜索根，最好指向安装根或其直接父目录 |
| `permission_denied`／`not_executable` | 用服务账号核对失败路径；所有者／管理员授权，或使用自己的安装 |
| `installation_incomplete` | 核对缺少的文件；确认指向完整 standalone 安装根，而非项目／源码目录 |
| `configuration_invalid` | 修正 JSON、绝对路径或 GPU 编号；环境变量优先于保存配置 |
| 终端成功，网页失败 | 查页面显示的服务账号、配置来源、服务环境和容器／systemd 路径 |
| 文件检查通过，Kit 启动失败 | 查 worker 日志、版本、RTX GPU、驱动与缓存／扩展权限；进行实际运行验证 |

运行参数、任务范围和恢复方式见[维护说明](isaac_operations.md)。
