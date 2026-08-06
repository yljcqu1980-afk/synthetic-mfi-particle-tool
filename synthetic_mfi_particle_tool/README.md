# 变压器油真实 MFI 杂质识别 App

本项目已由“合成样本演示版”迁移为“陈匡亚真实 MFI 数据版”。生产推理使用真实图像训练权重；历史合成数据仅保留作开发记录，不参与真实模型训练、验证或浓度计算。

## 真实类别

0. 纤维颗粒（`fiber_particle`）
1. 深色碳颗粒（`dark_carbon_particle`）
2. 浅色碳颗粒（`light_carbon_particle`）
3. 铜颗粒（`copper_particle`）
4. 气泡（`bubble`）
5. 聚集/混合颗粒（`aggregate_particle`）

## 数据与模型

- 原始真实数据：陈匡亚提供的 `datasets/mydata` 图像与 YOLO 标签。
- 清洗数据：`outputs/chen_real_mfi_dataset/`。
- 数据拆分：按增强前的原始来源分组，训练、验证、测试来源互斥。
- 训练集：允许使用陈匡亚真实原图及其增强图，以提高小样本类别曝光。
- 验证/测试集：只使用未增强的陈匡亚真实原图，不放入 `hflip`、`vflip`、`rotate` 等增强图。
- 虚拟图：不进入 `chen_real_mfi_dataset`，只用于 App 演示、界面测试或单独辅助实验。
- 模型：带 P2 检测头的轻量 YOLO，用于论文中常见的微小颗粒。
- App 权重：`models/particle_yolo.pt`。

重建清洗数据集：

```powershell
.\.venv-yolo\Scripts\python.exe scripts\prepare_chen_real_dataset.py
.\.venv-yolo\Scripts\python.exe scripts\build_balanced_train_list.py
```

训练时可优先使用 `outputs/chen_real_mfi_dataset/dataset_balanced.yaml`，其中 `train` 指向平衡采样后的 `train_balanced.txt`，`val` 和 `test` 仍指向来源隔离的真实原图集合。

## 定量口径

依据论文第 2.3.2 节的采集参数：原始图像为 4096×3000，100 张完整原始图像覆盖 0.25 mL，因此每张完整原始帧对应 2.5 μL。

- 完整 4096×3000 上传图：App 自动按 4×3 切为 12 个 1024×1000 区域识别，并按 2.5 μL/帧累计体积。
- 裁剪图、增强图、训练/验证图：只显示识别数量与形态参数，明确标记“不可定量”。
- 同一油样批次只有在全部图像均为完整原始帧时，才换算每 100 mL 颗粒浓度。
- “>5 μm、每 100 mL 不超过 3000 个”作为用户提供的参考限值展示，不替代正式检测标准或实验室校准。

## 启动

```powershell
.\.venv-yolo\Scripts\python.exe app_demo\server.py
```

默认地址：<http://127.0.0.1:8765>

## 数据限制

当前真实数据适合六类目标识别，但并非完整连续采集的定量数据集；独立测试集由来源隔离拆分得到，仍需增加更多独立油样、不同采集日期和不同仪器条件下的原始 4096×3000 图像，才能完成严格的跨批次泛化与浓度校准。
