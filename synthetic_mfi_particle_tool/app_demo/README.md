# 真实 MFI 杂质识别 App

启动：

```powershell
..\.venv-yolo\Scripts\python.exe server.py
```

App 使用 `../models/particle_yolo.pt` 中的 MFI 六分类权重。类别顺序以权重文件为准：纤维颗粒、深色碳颗粒、浅色碳颗粒、铜颗粒、气泡、聚集/混合颗粒。

定量规则固定如下：

- 完整 4096×3000 原始帧：自动切成 4×3 个 1024×1000 区域推理，每帧计 2.5 μL。
- 裁剪图、增强图、训练图：只做识别，API 返回 `quantifiable: false`，界面显示“不可定量”。
- 油样批次内只有全部图像均为完整原始帧时，才计算每 100 mL 浓度。

API：

- `GET /api/status`
- `GET /api/samples`
- `GET /api/sample/{split}/{image_name}`
- `GET /api/report/{split}/{image_name}`
- `POST /api/upload`
