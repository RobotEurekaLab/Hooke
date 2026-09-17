# 空间资产复用：范围、条款与实际验证

核查日期：2026-09-17。本文记录下表四个 NASA/SRB 外观资产的复用；后续操作硬件及 LROC/HiRISE 地形裁剪见 [空间实验操作](space_experiments.md)。完整 SRB、OmniLRS 和 MARTIAN 工作流尚未运行验证。

## 平台怎样组合

用户提出的分层是本项目的接入方向：Isaac Sim 提供 Isaac 路径中的物理与渲染，Isaac Lab 可组织机器人、传感器、动作与任务；SRB 提供空间域和机器人任务模板；OmniLRS 提供月球地形与场景生成；MARTIAN/Blender 负责离线制作火星地形。Hooke 保留原 MuJoCo 路径和双后端选择。

[SRB 官方使用示例](https://andrejorsula.github.io/space_robotics_bench/getting_started/basic_usage.html) 包含 USD 样品管、机器人及独立末端执行器、IK 动作组、样品收集和空间域配置。[Isaac Lab 的任务工作流](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/core-concepts/task_workflows.html) 支持管理器与直接实现两种组织方式。[OmniLRS](https://github.com/OmniLRS/OmniLRS) 基于 Isaac Sim/Omniverse。[MARTIAN](https://github.com/nasa-jpl/martian) 从 HiRISE 高程与影像制作 Blender 场景。

这是模块分工，不是互斥平台选择。目前 Hooke 的 Isaac 路径使用现有桥接器，没有迁移到 Isaac Lab。上游固定配置中的 Isaac 5.0 与本机 4.5 需要适配；本轮独立资产可用，不证明完整上游框架可直接启动。

## 已接入资产与研究用途

| 资产 | 三角面 | 漫反射纹理 | 条款与学术使用 |
| --- | ---: | ---: | --- |
| NASA Astrobee 美国实验舱 | 25,126 | 7 | NASA Media Usage Guidelines；教育、研究图形仿真用途可按其条款使用，保留来源、第三方权利与标识限制 |
| NASA Astrobee ISS 扶手 | 4,680 | 2 | 同上；原模型最长轴约 0.783 m，名称中的 `30` 不应解释成 30 cm |
| SRB Apollo 样品 1 | 1,000 | 1 | 官方归属清单列为 CC0；保留 NASA Johnson Space Center 与 SRB 来源 |
| SRB Mars 2020 样品管 | 17,419 | 0 | 原 NASA/JPL-Caltech 三维资源遵循 NASA 媒体条款；SRB 新增盖体、材质与打包修改为 CC0，分别保留来源 |

[NASA 媒体规则](https://www.nasa.gov/nasa-brand-center/images-and-media/) 明确覆盖三维多边形和纹理数据，并允许相关教育用途；不能把这些资产统一重标 CC0，也不能暗示 NASA 认可本项目。[SRB 归属清单](https://andrejorsula.github.io/space_robotics_bench/misc/attributions.html)、[原 Mars 2020 资源页](https://science.nasa.gov/3d-resources/mars-2020-sample-tube/) 和 [CC0 条款](https://creativecommons.org/publicdomain/zero/1.0/) 支持表中的来源判断。

Academic/research 用途仍需要逐项遵守条款。SRB 中 DLR 测试场的 CC BY-SA 4.0、背景图等其他许可不由本表覆盖；本轮没有导入这些资产。代码许可不覆盖所有第三方模型。

每个导出目录均含 `NOTICE.md`，固定提交及原文件 SHA-256 见 [来源清单](../Hooke/worlds/external_sources.json)。导出 OBJ、PNG、XML 的哈希、尺寸、材质分组和示例物性见 [导出清单](../Hooke/assets/space/manifest.json)。本轮保留全部 48,225 个源三角面，没有几何简化；逐面核查位置、UV、法线与解码纹理像素，见 [转换审计](validation/space_assets_geometry.json)。原文件缓存留在忽略的 `temp/`。

论文或网页图片可注明：`Hooke simulation render using NASA Astrobee/IGOAL/Robonaut2 assets, NASA Johnson Space Center Apollo geometry, and NASA/JPL-Caltech Mars 2020 geometry with Space Robotics Bench modifications.` 这是仿真生成图，不能标为 NASA 原始拍摄图或真实实验结果。

## 空间站安装与物理边界

空间站使用连接舱壁的固定工作面、支架与仪器安装底座，没有落地桌腿。刚体固定连接在零重力下仍可承载机械臂作用力；本模型将舱体作为固定参考，未实现站体反冲或螺栓柔性。真实 ISS 使用实验机架，参考 [ESA Columbus 实验舱](https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/International_Space_Station/Inside_the_Columbus_laboratory) 与 [NASA 微重力科学手套箱](https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/science-facilities-on-the-iss/microgravity-science-glovebox-msg/)。本模型是 Hooke 概念实验架，不是认证 ISS 硬件。

样品与仪器不能靠重量和桌面摩擦保持位置：存放样品采用显式固定连接和可见保持结构。自由样品专门用于漂浮/下落验证。开合、锁定、释放、抓取与断裂尚未验收。零重力场景移除普通分析天平，仅保留未校准的单轴惯性滑台；它尚不能输出可信质量。月球/火星的分析天平外观也未按当地重力标定。

ISS 外观网格保持完整，视觉网格关闭碰撞；边界板代理避免整个舱体的凸包填满内部。扶手和样品管用胶囊代理，岩样用凸包。机架凹槽、管内空腔和密封面不是精确碰撞。示例样品质量为 0.1 kg，惯量按包围盒设置；保持底板 0.02 kg、每片保持片 0.001 kg 也是示例物性，不能当作 NASA 实测值。

ISS 模型按内部地板安装，整体向下平移 0.94 m。舱内点光源由源模型导入为归一化球形光源；强度为示例换算，聚光、衰减和环境光没有严格对应。不同引擎的光照、金属感与完整 PBR 一致性仍未验收。月球和火星地形仍为程序化原型，没有导入 OmniLRS DEM 或运行 MARTIAN。

重力与压力分区沿用 [环境说明](space_worlds.md)。压力字段不会自动实现空气、泄漏、沸腾或粉尘。隔离箱和有盖样品管也不证明气密性或真空兼容性。

## 运行与重新制作

网页 `/space-assets` 提供三场景、双引擎、全景/实验台/资产特写，并可进入实际运行页面。任务名称为 `space_orbital_assets_workstation`、`space_lunar_assets_workstation`、`space_martian_assets_workstation`。它们均为展示和物理检查入口，不输出科学实验成功。

离线转换依赖单独的 Python 环境；运行时不需要 USD/COLLADA 解析库：

```bash
python -m venv temp/space-asset-env
temp/space-asset-env/bin/pip install -r scripts/space_asset_requirements.txt
temp/space-asset-env/bin/python scripts/prepare_space_assets.py
temp/space-asset-env/bin/python scripts/audit_space_assets.py \
  --report temp/space-assets-geometry.json
cd Hooke
../.venv/bin/python -m worlds.asset_scenes
HOOKE_RENDER_FPS=2 HOOKE_ISAAC_COLOR_PIPELINE=source_display \
  ../.venv/bin/python -m worlds.qualify --asset-scenes --gpu 6 \
  --output ../temp/backend_parity/space-assets-validation
```

本机下载时使用 `--ca-bundle /etc/ssl/certs/ca-certificates.crt`，pip 可使用对应的 `--cert` 参数。已有全部缓存可加 `--offline`；原文件校验失败会拒绝转换，不关闭 TLS 校验。

[场景启动预检](validation/space_assets_preflight.json)、[实际双引擎报告](validation/space_assets_summary.json) 和 [浏览器记录](validation/space_assets_browser.json) 分别记录验证范围。实际结果为 3 场景 × 2 引擎共 6 项通过，每项 250 步、0.5 秒，共生成 36 张原始截图，网页采用其中 18 张。最大自由运动位置误差约 1.867 mm，符合 2.5 mm 的短时检查公差。最终 133 项单元测试通过，桌面/手机的 18 项选择检查及 18 张图片哈希核查通过。短时漂浮/下落与出图不证明夹具操作、长期接触稳定性或完整科学实验通过。

后续按独立模块推进：SRB 操作模板到 Hooke 的动作/观测/复位接口；OmniLRS 地形生成到缓存 USD/网格；MARTIAN/HiRISE 到有投影和分辨率的局部地形；各自完成来源审计、尺度、碰撞、运行和任务验证后再启用。
