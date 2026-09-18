# 显微注射的论文布局与样本尺度

2026-09-18 根据用户指出的显微镜、工具夹持和尺度问题，重新核对 UofT 的 Xinyu Liu、Yu Sun 相关工作。以下布局结论来自实际查看论文图片和正文；电动关节或 CAD 测试通过不计作外观还原通过。

## 直接查看的系统图

| 原始论文 | 图片与配置 | 对当前场景的要求 |
| --- | --- | --- |
| Xinyu Liu、Keekyoung Kim、Yong Zhang、Yu Sun，[Nanonewton Force Sensing and Control in Microrobotic Cell Manipulation](https://amnl.mie.utoronto.ca/data/J43.pdf)，IJRR，2009，DOI 10.1177/0278364909340212 | 第 3 页 Fig. 2 为实拍：Nikon TE2000、Sutter MP-285、Prior ProScan II、Basler 相机及温控罩；样本为平均直径 98 µm 的鼠胚胎 | 商用倒置镜的底座、观察筒、后方照明架要完整；微操手紧邻 XY 台，样本装置位于光轴中心 |
| Xinyu Liu、Zhe Lu、Yu Sun，[Orientation Control of Biological Cells Under Inverted Microscopy](https://amnl.mie.utoronto.ca/data/J54.pdf)，IEEE/ASME TMECH，2011，DOI 10.1109/TMECH.2010.2056380 | 第 2 页 Fig. 3a 为旋转载物台示意，3b 为实拍：Nikon TE2000-S、MP-285、ProScan、温控罩；注射针 45°，鼠胚胎极体需要避开穿刺区域 | 后续胚胎任务独立设置旋转和极体；当前小细胞 phantom 不冒充该实验 |
| Jun Liu 等、Yu Sun，[Robotic Adherent Cell Injection for Characterizing Cell–Cell Communication](https://amnl.mie.utoronto.ca/data/J109.pdf)，IEEE TBME，2015，DOI 10.1109/TBME.2014.2342036 | 第 2 页 Fig. 2a 实拍、2b 布局示意：Nikon TE2000-S、Siskiyou MX7600、Prior ProScan；文中注射针外径 500 nm、内径 300 nm，贴壁细胞厚度为数 µm | 贴壁细胞需要薄而非球形；压帽、夹具、气管、载物台和相机应按真实装配关系组织。此论文第一作者是 Jun Liu，不能误写成 Xinyu Liu |
| Peng Pan 等、Xinyu Liu，[Robotic microinjection enables large-scale transgenic studies of Caenorhabditis elegans](https://www.nature.com/articles/s41467-024-53108-5)，Nature Communications，2024，DOI 10.1038/s41467-024-53108-5 | Fig. 1a 是整机示意而非实拍；另查看补充材料第 10 页 Fig. 3 的实物组件：Olympus IX83、左 MP-285、右 MX7600R、Narishige HI-7 和 IM-300、ProScan-H117；注射针 15°，微流体装置 30° | 右侧为线虫微流体固定／转移／旋转装置；线虫任务需要生殖腺、流道和独立压力配置，不能简单复用球形细胞的双针任务 |

论文原图、PDF、原始下载 URL、哈希与本地查看图片均在忽略的 `temp/microscopy_research/paper_layout_references/`。本文用链接引用，不将论文图片作为模型纹理或提交到仓库。2024 论文标示 CC BY-NC-ND 4.0；其他论文分别保留其原始声明。下载与私有研究记录不等于取得品牌 CAD 或图片的公开再分发许可。

## 针座与镜体的原始厂家依据

- [Narishige HI-7](https://products.narishige-group.com/group1/HI-7/injection/english.html)：140 mm 长，适配外径 1 mm 玻璃管；前端为不锈钢 collar／压帽，并配 HIR。后续独立建模应有这些结构及气管连接，不能用一个圆筒代替。
- [Narishige HIR](https://products.narishige-group.com/group1/HIR/injection/english.html)：适配 Ø4 mm 针座轴，夹具包络 7 × 7 × 18 mm。轴径与总长属于官方依据；压帽细部、螺纹和装配位置仍需尺寸核验。
- [Sutter MP-285](https://www.sutter.com/micromanipulation/mp-285)：三物理轴，25 mm 行程；可配置的第四轴是组合轴运动。不能将它写作额外实体关节，也不能将厂家分辨率作为仿真精度。
- [Nikon TE2000 官方尺寸册](https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf)：提供 TE2000-E/U/S 各自的前／侧视尺寸。型号和附件配置分别核查。
- [Olympus／Evident IX83 原始资料](https://evidentscientific.com/en/products/obsolete/ixplore-ix83-tirf)：对应配置提供镜架 323 × 475 × 706 mm 的包络和聚光器信息；不同 deck／附件配置不可混用。
- [Siskiyou CAD 下载说明](https://siskiyou.com/cad-downloads/)和 [MX7600 产品页](https://siskiyou.com/mx7600-series-manipulator/)提供模型下载线索；当前未取得并接入该型号的 STEP。商用镜体完整 CAD 也尚未取得，不能计作完成。

## 尺度必须与样本类别一致

| 类别 | 采用的尺度 | 实现状态 |
| --- | --- | --- |
| 贴壁细胞 phantom | 36 × 28 × 8 µm；中心离玻璃 4 µm，核为 12 × 8 × 4 µm | 本轮把旧 14 µm 厚度改为 8 µm，并将名义进针深度改为 2 µm，避开较薄的核；尺寸属于原创名义参数，并非 HL-1 实测标定 |
| 悬浮细胞 phantom | 球形直径 36 µm，核直径 12 µm | 保持实际显示／受力模型尺寸；隐藏参照几何修正为相同球形尺寸。背景细胞为贴壁视觉上下文 |
| 注射针尖 | 外径 1.2 µm、内径 0.5 µm；距尖端 100／300 µm 处外径 2／4 µm；暴露段 12 mm，后端外径 1 mm | 空心网格、合成显微轮廓和分段流阻共用名义轮廓；与 TBME 论文的 0.5／0.3 µm 并不完全相同。针座与玻璃管暴露段长度分别记录，当前注射针／吸持管水平夹角为 35°／30° |
| 鼠胚胎 | 上述 2009 论文平均直径 98 µm | 当前尚无该尺度的独立胚胎任务；需单独配置透明带、极体、视野和进针深度 |
| 成年 C. elegans | 约 1 mm 长、50 µm 量级体径，个体与阶段有差异 | 尚未实现线虫任务。尺寸参考[原始尺寸测量研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6994075/)；生殖腺局部观察和整虫观察需要不同视野 |
| 标定微珠 | 直径 1.2 mm | 机械夹取／推移测试样本；与细胞任务分开标注 |

细胞合成视野为 160 µm／768 像素，即约 0.208 µm／像素；20 µm 标尺对应 96 像素，36 µm 的 X 方向细胞轮廓约占 173 像素。三维几何按 SI 单位建立，局部视角通过相机接近，而非扩大样本几何。网页从运行状态读取三轴尺寸、针尖内外径、视野和像素数。合成物方映射明确标记为未完成实体物镜／传感器光学校准。

尺度测试直接测量合成图膜边界，与编译后三维样本尺寸对照；另在世界坐标测量空心针的轴向长度。测试覆盖贴壁／悬浮两条显示路径，不能用坐标换算一致证明细胞力学、光学或存活已标定。

## 当前外观缺点与改造顺序

当前 CAD 场景中的 openFrame 镜体是有效的开源光学机构，但与上述商用倒置镜的镜体形态不同。原创 330–380 mm 工具延长连接和高立柱使镜体显得小，简单圆筒针夹缺乏实物结构；原始 CAD 精确并没有解决整机比例与布局问题。外观还原目前不通过。

先选定上述一个商用镜体／附件配置，依据官方尺寸和论文实拍建立完整外观；取得许可合适的 CAD 时保留真实尺寸，否则独立重建并列出估算项。再组织紧凑安装板、微操手、140 mm 量级针座、玻璃管及柔性管路，逐步核验接口与运动间距。固定光学件、移动部件、针座和样本分别组织，避免为外观改变样本尺度或驱动映射。镜体、针座、整体布局和显微局部都须实际渲染并与原图比较后再验收。

旧 `illumination_assembly_v2` 录像仍对应 36 × 28 × 14 µm 的旧贴壁样本；保留其原尺寸、模型及检查记录。本轮新 `paper_scale_v1` 记录独立存放，不将旧流程结果重写成新尺寸或论文整机还原结果。

## 本轮尺度验证

21 项相关单元测试通过；MuJoCo 五项任务完成，原生 Isaac 两项细胞任务完成。实际浏览器完成贴壁注射、双侧吸持注射与微珠推移，尺度显示与五段新录像读取正常，脚本错误为 0。新原生两条器械轨迹间距检查无未解决候选、零跳过；实际归档 USD 声明 `metersPerUnit = 1`，目标与核的尺寸与运行报告一致。原始结果与哈希摘要在忽略的 `temp/microscopy_demo/cad_integration/paper_scale_v1_verification_summary.json`。

薄细胞的名义椭球体积约 4.22 pL，现有 0.5 pL 剂量约占 11.8%；这是演示配置，尚无实测体积、存活或剂量标定。原有模型参数没有因尺度一致而获得生物有效性，合成显微图仍不是物镜／相机形成的真实显微光学图像。
