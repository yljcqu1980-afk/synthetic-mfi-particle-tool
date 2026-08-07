from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES = {
    0: "bubble",
    1: "cellulose_short",
    2: "copper_particle",
    3: "light_carbon_particle",
    4: "dark_carbon_particle",
    5: "cellulose_long",
    # 后处理拒识类别，不是 YOLO 权重中的第七个训练类别。
    6: "other_review",
}


CLASS_CN = {
    0: "气泡",
    1: "短纤维素",
    2: "铜颗粒",
    3: "浅色碳颗粒",
    4: "深色碳颗粒",
    5: "长纤维素",
    6: "其他/待复核",
}

@dataclass
class DetectorStatus:
    inference_mode: str
    model_path: str
    model_loaded: bool
    message: str


class YoloDetector:
    """Thin wrapper around a real Ultralytics YOLO model.

    This class is deliberately explicit about demo fallback. It never labels
    static labels or synthetic data as real YOLO inference.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model: Any | None = None
        self.load_error = ""
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"YOLO weight file not found: {self.model_path}"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(str(self.model_path))
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps.
            self.model = None
            self.load_error = f"Failed to load YOLO model: {exc}"

    @property
    def status(self) -> DetectorStatus:
        if self.model is None:
            return DetectorStatus(
                inference_mode="awaiting_real_yolo_weights",
                model_path=str(self.model_path),
                model_loaded=False,
                message=self.load_error or "真实 MFI YOLO 权重尚未加载。",
            )
        return DetectorStatus(
            inference_mode="real_yolo",
            model_path=str(self.model_path),
            model_loaded=True,
            message="Real YOLO model loaded.",
        )

    def predict(
        self,
        image_path: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1024,
        max_det: int = 300,
        device: str = "",
        agnostic_nms: bool = False,
    ) -> tuple[list[dict], float]:
        """Run real YOLO inference. Raises if the model is not loaded."""
        if self.model is None:
            raise RuntimeError(self.status.message)
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "agnostic_nms": agnostic_nms,
            "verbose": False,
        }
        if device.strip():
            kwargs["device"] = device.strip()
        results = self.model(str(image_path), **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        detections: list[dict] = []
        if not results:
            return detections, elapsed_ms
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections, elapsed_ms
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append(
                {
                    "id": idx,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "class_cn": CLASS_CN.get(class_id, f"类别 {class_id}"),
                    "confidence": confidence,
                    "bbox": [int(round(v)) for v in xyxy],
                    "source": "real_yolo",
                }
            )
        return detections, elapsed_ms

