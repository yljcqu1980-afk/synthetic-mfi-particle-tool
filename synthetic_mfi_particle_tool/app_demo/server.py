from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
import time
import urllib.parse
import uuid
import base64
import os
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.explainability import particle_explanation, particle_risk_hint, sample_summary
from backend.services.morphology_analyzer import analyze_roi, load_gray
from backend.services.yolo_detector import CLASS_CN, CLASS_NAMES, YoloDetector


APP_DIR = Path(__file__).resolve().parent
DATASET = ROOT / "outputs" / "chen_real_mfi_dataset"
METADATA = ROOT / "outputs" / "metadata" / "metadata.csv"
MODEL_PATH = ROOT / "models" / "particle_yolo.pt"
UPLOAD_DIR = ROOT / "outputs" / "uploaded_images"
UPLOAD_META = ROOT / "outputs" / "uploaded_images" / "_uploaded_metadata.json"
PROJECT_STATE = ROOT / "outputs" / "uploaded_images" / "_project_state.json"
INFERENCE_CACHE_DIR = ROOT / "outputs" / "uploaded_images" / "_inference_cache"
INFERENCE_SUMMARY_INDEX = ROOT / "outputs" / "uploaded_images" / "_inference_summary_index.json"
MANUAL_LABEL_DIR = ROOT / "outputs" / "manual_label_library"
MANUAL_LABEL_DB = MANUAL_LABEL_DIR / "annotations.json"
RAW_IMAGE_W = 4096
RAW_IMAGE_H = 3000
TILE_W = 1024
TILE_H = 1000
# 陈匡亚论文 2.3.2：100 张完整原始图像覆盖 0.25 mL，即每张 2.5 μL。
# 裁剪图、增强图或不完整批次只允许做识别，不允许换算浓度。
VOLUME_UL_PER_RAW_FRAME = 2.5
DEMO_VOLUME_UL_PER_IMAGE = 100.0
PIXEL_SIZE_UM = 0.7
FIBER_CANDIDATE_CONF = 0.08

DETECTOR = YoloDetector(MODEL_PATH)

DEFAULT_INFER_SETTINGS = {
    "conf": 0.25,
    "other_conf": 0.45,
    "iou": 0.45,
    "imgsz": 1024,
    "max_det": 300,
    "device": "",
    "agnostic_nms": True,
    "pixel_size_um": PIXEL_SIZE_UM,
}


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def infer_settings(query: dict[str, list[str]] | None = None) -> dict:
    query = query or {}

    def first(name: str, default: object) -> str:
        vals = query.get(name)
        return str(vals[0]) if vals else str(default)

    settings = {
        "conf": min(max(float(first("conf", DEFAULT_INFER_SETTINGS["conf"])), 0.0), 0.95),
        "other_conf": min(max(float(first("other_conf", DEFAULT_INFER_SETTINGS["other_conf"])), 0.0), 0.95),
        "iou": min(max(float(first("iou", DEFAULT_INFER_SETTINGS["iou"])), 0.05), 0.95),
        "imgsz": int(first("imgsz", DEFAULT_INFER_SETTINGS["imgsz"])),
        "max_det": int(first("max_det", DEFAULT_INFER_SETTINGS["max_det"])),
        "device": first("device", DEFAULT_INFER_SETTINGS["device"]).strip(),
        "agnostic_nms": parse_bool(first("agnostic_nms", str(DEFAULT_INFER_SETTINGS["agnostic_nms"]))),
        "pixel_size_um": max(0.001, float(first("pixel_size_um", DEFAULT_INFER_SETTINGS["pixel_size_um"]))),
    }
    settings["imgsz"] = min(max(settings["imgsz"], 320), 2048)
    settings["max_det"] = min(max(settings["max_det"], 1), 2000)
    settings["other_conf"] = max(settings["conf"], settings["other_conf"])
    return settings


def parse_multipart_upload(headers, rfile) -> tuple[str, bytes, dict[str, str]]:
    content_type = headers.get("Content-Type", "")
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("multipart boundary missing")
    boundary = match.group("boundary").strip().strip('"').encode("utf-8")
    length = int(headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("empty upload")
    body = rfile.read(length)
    delimiter = b"--" + boundary
    fields: dict[str, str] = {}
    image: tuple[str, bytes] | None = None
    for part in body.split(delimiter):
        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        name_match = re.search(rb'name="([^"]*)"', header_blob)
        field_name = name_match.group(1).decode("utf-8", errors="ignore") if name_match else ""
        content = content.rstrip(b"\r\n")
        if content.endswith(b"--"):
            content = content[:-2].rstrip(b"\r\n")
        if field_name == "image":
            filename_match = re.search(rb'filename="([^"]*)"', header_blob)
            filename = filename_match.group(1).decode("utf-8", errors="ignore") if filename_match else "uploaded.png"
            if not content:
                raise ValueError("uploaded image is empty")
            image = (filename, content)
        elif field_name:
            fields[field_name] = content.decode("utf-8", errors="ignore").strip()
    if image is None:
        raise ValueError("image file is required")
    return image[0], image[1], fields


def load_upload_meta() -> dict:
    if not UPLOAD_META.exists():
        return {}
    try:
        return json.loads(UPLOAD_META.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def save_upload_meta(meta: dict) -> None:
    UPLOAD_META.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project_state() -> dict:
    if not PROJECT_STATE.exists():
        return {"projects": [], "oilGroupProjects": {}, "expandedProjects": [], "deletedOilGroups": []}
    try:
        state = json.loads(PROJECT_STATE.read_text(encoding="utf-8-sig"))
        return {
            "projects": state.get("projects") if isinstance(state.get("projects"), list) else [],
            "oilGroupProjects": state.get("oilGroupProjects") if isinstance(state.get("oilGroupProjects"), dict) else {},
            "expandedProjects": state.get("expandedProjects") if isinstance(state.get("expandedProjects"), list) else [],
            "deletedOilGroups": state.get("deletedOilGroups") if isinstance(state.get("deletedOilGroups"), list) else [],
        }
    except Exception:
        return {"projects": [], "oilGroupProjects": {}, "expandedProjects": [], "deletedOilGroups": []}


def save_project_state(state: dict) -> None:
    PROJECT_STATE.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "projects": state.get("projects") if isinstance(state.get("projects"), list) else [],
        "oilGroupProjects": state.get("oilGroupProjects") if isinstance(state.get("oilGroupProjects"), dict) else {},
        "expandedProjects": state.get("expandedProjects") if isinstance(state.get("expandedProjects"), list) else [],
        "deletedOilGroups": state.get("deletedOilGroups") if isinstance(state.get("deletedOilGroups"), list) else [],
    }
    PROJECT_STATE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key_for(image_path: Path, settings: dict, status: dict) -> str:
    stat = image_path.stat()
    payload = {
        "image_name": image_path.name,
        "image_mtime_ns": stat.st_mtime_ns,
        "image_size": stat.st_size,
        "model_path": status.get("model_path"),
        "model_loaded": status.get("model_loaded"),
        "model_mtime_ns": status.get("model_mtime_ns"),
        "model_schema": status.get("model_schema"),
        "settings": settings,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def load_inference_cache(image_path: Path, settings: dict, status: dict) -> dict | None:
    cache_path = INFERENCE_CACHE_DIR / f"{image_path.stem}_{cache_key_for(image_path, settings, status)}.json"
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload.update(quantification_info("uploaded", image_path))
        refresh_detection_labels(payload)
        payload["cache_hit"] = True
        payload["inference_time_ms"] = 0.0
        return payload
    except Exception:
        return None


def save_inference_cache(image_path: Path, settings: dict, status: dict, payload: dict) -> None:
    INFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = INFERENCE_CACHE_DIR / f"{image_path.stem}_{cache_key_for(image_path, settings, status)}.json"
    cache_payload = dict(payload)
    cache_payload["cache_hit"] = False
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_inference_summary_index(image_path.name, payload)


def update_inference_summary_index(image_name: str, payload: dict | None) -> None:
    summaries = inference_summary_index()
    image_stem = Path(image_name).stem
    if payload is None:
        summaries.pop(image_stem, None)
    else:
        detections = payload.get("detections") or []
        counts = Counter(det.get("class_id") for det in detections)
        dominant = counts.most_common(1)[0][0] if counts else None
        summaries[image_stem] = {
            "particles": len(detections),
            "dominant_class": CLASS_CN.get(dominant, "-") if dominant is not None else "-",
            "recognized": True,
        }
    INFERENCE_SUMMARY_INDEX.write_text(json.dumps(summaries, ensure_ascii=False), encoding="utf-8")


def clear_inference_cache(image_name: str) -> None:
    if not INFERENCE_CACHE_DIR.exists():
        return
    stem = Path(image_name).stem
    for cache_path in INFERENCE_CACHE_DIR.glob(f"{stem}_*.json"):
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass
    update_inference_summary_index(image_name, None)


def quantification_info(split: str, image_path: Path) -> dict:
    width, height = load_gray(image_path).shape[::-1]
    is_raw_frame = split == "uploaded" and width == RAW_IMAGE_W and height == RAW_IMAGE_H
    reason = (
        "完整 4096×3000 原始帧；按论文标定每帧 2.5 μL。"
        if is_raw_frame
        else "该图不是完整 4096×3000 原始帧，仅用于颗粒识别；裁剪图、增强图和训练图不可换算浓度。"
    )
    return {
        "width": width,
        "height": height,
        "image_width": width,
        "image_height": height,
        "acquisition_mode": "raw_full_frame" if is_raw_frame else "recognition_image",
        "quantifiable": is_raw_frame,
        "quantification_reason": reason,
        "volume_ul": VOLUME_UL_PER_RAW_FRAME if is_raw_frame else None,
    }


def _scan_inference_summary_index() -> dict[str, dict]:
    """Read only the newest cache file per image for the sample browser."""
    if not INFERENCE_CACHE_DIR.exists():
        return {}
    newest: dict[str, Path] = {}
    for cache_path in INFERENCE_CACHE_DIR.glob("*.json"):
        match = re.match(r"^(?P<stem>.+)_[0-9a-f]{32}$", cache_path.stem)
        if not match:
            continue
        image_stem = match.group("stem")
        previous = newest.get(image_stem)
        if previous is None or cache_path.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            newest[image_stem] = cache_path
    summaries = {}
    for image_stem, cache_path in newest.items():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        detections = payload.get("detections") or []
        counts = Counter(det.get("class_id") for det in detections)
        dominant = counts.most_common(1)[0][0] if counts else None
        summaries[image_stem] = {
            "particles": len(detections),
            "dominant_class": CLASS_CN.get(dominant, "-") if dominant is not None else "-",
            "recognized": True,
        }
    return summaries


def inference_summary_index() -> dict[str, dict]:
    """Load the compact summary index; build it once for older cache folders."""
    if INFERENCE_SUMMARY_INDEX.exists():
        try:
            value = json.loads(INFERENCE_SUMMARY_INDEX.read_text(encoding="utf-8-sig"))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    summaries = _scan_inference_summary_index()
    INFERENCE_SUMMARY_INDEX.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_SUMMARY_INDEX.write_text(json.dumps(summaries, ensure_ascii=False), encoding="utf-8")
    return summaries


def latest_inference_summary(image_name: str, summary_index: dict[str, dict] | None = None) -> dict:
    if summary_index is not None:
        return summary_index.get(Path(image_name).stem, {"particles": 0, "dominant_class": "待检测", "recognized": False})
    if not INFERENCE_CACHE_DIR.exists():
        return {"particles": 0, "dominant_class": "待检测", "recognized": False}
    stem = Path(image_name).stem
    cache_files = sorted(INFERENCE_CACHE_DIR.glob(f"{stem}_*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for cache_path in cache_files:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        detections = payload.get("detections") or []
        counts = Counter(det.get("class_id") for det in detections)
        dominant = counts.most_common(1)[0][0] if counts else None
        return {
            "particles": len(detections),
            "dominant_class": CLASS_CN.get(dominant, "-") if dominant is not None else "-",
            "recognized": True,
        }
    return {"particles": 0, "dominant_class": "待检测", "recognized": False}


def latest_inference_payload(image_name: str) -> dict | None:
    if not INFERENCE_CACHE_DIR.exists():
        return None
    image_path = UPLOAD_DIR / image_name
    stem = Path(image_name).stem
    cache_files = sorted(INFERENCE_CACHE_DIR.glob(f"{stem}_*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for cache_path in cache_files:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload.update(quantification_info("uploaded", image_path))
        payload.update(upload_meta_for(image_name))
        refresh_detection_labels(payload)
        payload["cache_hit"] = True
        payload["cache_only"] = True
        payload["inference_time_ms"] = 0.0
        return payload
    return None


def upload_meta_for(image_name: str) -> dict:
    meta = load_upload_meta().get(image_name, {})
    return {
        "display_name": meta.get("display_name") or image_name,
        "group": meta.get("group") or "未分组",
        "note": meta.get("note") or "",
    }


def read_json_body(headers, rfile) -> dict:
    length = int(headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(rfile.read(length).decode("utf-8"))


def load_manual_label_db() -> dict:
    if not MANUAL_LABEL_DB.exists():
        return {"schema_version": 1, "images": {}}
    try:
        value = json.loads(MANUAL_LABEL_DB.read_text(encoding="utf-8-sig"))
        return value if isinstance(value.get("images"), dict) else {"schema_version": 1, "images": {}}
    except Exception:
        return {"schema_version": 1, "images": {}}


def save_manual_label_db(value: dict) -> None:
    MANUAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_LABEL_DB.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def manual_review_for(image_name: str) -> dict | None:
    record = load_manual_label_db()["images"].get(image_name)
    return record if isinstance(record, dict) else None


def normalize_manual_annotations(image_path: Path, annotations: list) -> list[dict]:
    width, height = Image.open(image_path).size
    clean = []
    for position, row in enumerate(annotations):
        if not isinstance(row, dict):
            continue
        class_id = int(row.get("class_id", 6))
        if class_id not in CLASS_NAMES:
            continue
        bbox = row.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        clean.append(
            {
                "annotation_id": str(row.get("annotation_id") or uuid.uuid4()),
                "class_id": class_id,
                "bbox": [x1, y1, x2, y2],
                "origin": str(row.get("origin") or "model"),
                "operation": str(row.get("operation") or "confirmed"),
                "model_class_id": row.get("model_class_id"),
                "model_confidence": row.get("model_confidence"),
                "model_bbox": row.get("model_bbox"),
                "position": position,
            }
        )
    return clean


def save_manual_review(image_name: str, data: dict) -> dict:
    image_path = image_path_for("uploaded", image_name)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    review_scope = str(data.get("review_scope") or "partial")
    if review_scope not in {"partial", "complete"}:
        review_scope = "partial"
    db = load_manual_label_db()
    previous = db["images"].get(image_name) or {}
    annotations = normalize_manual_annotations(image_path, data.get("annotations") or [])
    if review_scope == "partial" and not annotations:
        raise ValueError("局部修正不能保存为空；如确认整图无目标，请选择整图审核完成")
    record = {
        "image_name": image_name,
        "split": "uploaded",
        "review_scope": review_scope,
        "training_eligible": review_scope == "complete",
        "reviewer": str(data.get("reviewer") or "人工复核").strip()[:80],
        "note": str(data.get("note") or "").strip()[:500],
        "created_at": previous.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_schema": detector_status().get("model_schema"),
        "annotations": annotations,
        "change_log": (previous.get("change_log") or []) + [
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reviewer": str(data.get("reviewer") or "人工复核").strip()[:80],
                "changes": data.get("changes") if isinstance(data.get("changes"), list) else [],
            }
        ],
    }
    db["images"][image_name] = record
    save_manual_label_db(db)
    return record


def delete_manual_review(image_name: str) -> bool:
    db = load_manual_label_db()
    removed = db["images"].pop(image_name, None) is not None
    if removed:
        save_manual_label_db(db)
    return removed


def apply_manual_review(payload: dict, image_path: Path) -> dict:
    record = manual_review_for(payload.get("image_name", ""))
    payload["manual_review"] = record
    if not record:
        return payload
    gray = load_gray(image_path)
    model_by_id = {
        str(det.get("annotation_id") or det.get("particle_id")): det for det in payload.get("detections") or []
    }
    detections = []
    for idx, annotation in enumerate(record.get("annotations") or []):
        original = model_by_id.get(str(annotation.get("annotation_id")), {})
        class_id = int(annotation["class_id"])
        bbox = annotation["bbox"]
        morph = analyze_roi(gray, bbox, class_id)
        det = {
            **original,
            "id": idx,
            "particle_id": idx,
            "annotation_id": annotation["annotation_id"],
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "class_cn": CLASS_CN[class_id],
            "bbox": bbox,
            "bbox_xmin": bbox[0],
            "bbox_ymin": bbox[1],
            "bbox_xmax": bbox[2],
            "bbox_ymax": bbox[3],
            "confidence": annotation.get("model_confidence"),
            "morphology": morph,
            "eq_diameter_um": morph["eq_diameter_px"] * float(payload.get("pixel_size_um") or PIXEL_SIZE_UM),
            "area_um2": morph["area_px"] * float(payload.get("pixel_size_um") or PIXEL_SIZE_UM) ** 2,
            "source": "manual_label_library",
            "manual_label": True,
            "manual_operation": annotation.get("operation"),
            "model_class_id": annotation.get("model_class_id"),
            "model_confidence": annotation.get("model_confidence"),
            "model_bbox": annotation.get("model_bbox"),
        }
        detections.append(det)
    payload["detections"] = detections
    refresh_detection_labels(payload)
    payload["total_particles"] = len(detections)
    return payload


def manual_label_export() -> dict:
    db = load_manual_label_db()
    images = []
    for image_name, record in db["images"].items():
        if not record.get("training_eligible"):
            continue
        image_path = image_path_for("uploaded", image_name)
        if not image_path.exists():
            continue
        width, height = Image.open(image_path).size
        lines = []
        for annotation in record.get("annotations") or []:
            class_id = int(annotation["class_id"])
            if class_id == 6:
                continue
            x1, y1, x2, y2 = annotation["bbox"]
            lines.append(
                f"{class_id} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} "
                f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
            )
        images.append({"image_name": image_name, "label_file": f"{Path(image_name).stem}.txt", "yolo_labels": lines})
    return {
        "schema_version": 1,
        "note": "仅包含已标记为整图审核完成的图片；类别6其他/待复核不作为YOLO训练目标导出。",
        "class_names": {str(cid): name for cid, name in CLASS_CN.items() if cid != 6},
        "eligible_image_count": len(images),
        "images": images,
    }


def load_metadata() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    if not METADATA.exists():
        return grouped
    with METADATA.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if any(value is None for value in row.values()):
                continue
            try:
                image_name = row["image_name"]
                for key in ["particle_id", "class_id", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]:
                    row[key] = int(float(row[key]))
                for key in ["area_px", "eq_diameter_px", "aspect_ratio", "circularity", "gray_mean", "gray_std"]:
                    row[key] = float(row[key])
            except (TypeError, ValueError, KeyError):
                continue
            cid = row["class_id"]
            row["class_cn"] = CLASS_CN.get(cid, f"类别 {cid}")
            grouped.setdefault(image_name, []).append(row)
    return grouped


META_BY_IMAGE = load_metadata()


def detector_status() -> dict:
    status = DETECTOR.status
    model_stat = MODEL_PATH.stat() if MODEL_PATH.exists() else None
    return {
        "inference_mode": status.inference_mode,
        "model_path": status.model_path,
        "model_loaded": status.model_loaded,
        "message": "陈匡亚真实 MFI 数据六分类模型；低可信目标归入“其他/待复核”。" if status.model_loaded else status.message,
        "model_schema": "chen_real_mfi_v4_round_fiber_rejection",
        "model_mtime_ns": model_stat.st_mtime_ns if model_stat else None,
        "quantification_policy": "仅完整4096×3000原始帧可定量：2.5 μL/帧，100帧=0.25 mL；裁剪图仅用于识别。",
        "default_settings": DEFAULT_INFER_SETTINGS,
    }


def dataset_label_counts(split: str, image_stem: str) -> Counter:
    label_path = DATASET / "labels" / split / f"{image_stem}.txt"
    counts = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            counts[int(float(fields[0]))] += 1
    return counts


def image_url(split: str, image_name: str) -> str:
    return f"/api/image/{urllib.parse.quote(split, safe='')}/{urllib.parse.quote(image_name, safe='')}"


def list_samples() -> list[dict]:
    samples = []
    upload_meta = load_upload_meta()
    summary_index = inference_summary_index()
    for path in sorted(UPLOAD_DIR.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        meta = upload_meta.get(path.name, {})
        summary = latest_inference_summary(path.name, summary_index)
        samples.append(
            {
                "image_name": path.name,
                "display_name": meta.get("display_name") or path.name,
                "group": meta.get("group") or "未分组",
                "note": meta.get("note") or "",
                "split": "uploaded",
                "image_url": image_url("uploaded", path.name),
                "particles": summary["particles"],
                "dominant_class": summary["dominant_class"],
                "recognized": summary["recognized"],
            }
        )
    for split in ["train", "val", "test"]:
        for path in sorted(DATASET.joinpath("images", split).glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            counts = dataset_label_counts(split, path.stem)
            samples.append(
                {
                    "image_name": path.name,
                    "split": split,
                    "particles": sum(counts.values()),
                    "dominant_class": CLASS_CN[counts.most_common(1)[0][0]] if counts else "-",
                    "image_url": image_url(split, path.name),
                }
            )
    return samples


def demo_detections_from_metadata(image_path: Path, image_name: str) -> tuple[list[dict], float]:
    start = time.perf_counter()
    gray = load_gray(image_path)
    detections = []
    for idx, row in enumerate(META_BY_IMAGE.get(image_name, [])):
        bbox = [row["bbox_xmin"], row["bbox_ymin"], row["bbox_xmax"], row["bbox_ymax"]]
        morph = analyze_roi(gray, bbox, row["class_id"])
        confidence = None
        det = {
            "id": idx,
            "particle_id": row["particle_id"],
            "class_id": row["class_id"],
            "class_name": CLASS_NAMES[row["class_id"]],
            "class_cn": CLASS_CN[row["class_id"]],
            "confidence": confidence,
            "bbox": bbox,
            "bbox_xmin": bbox[0],
            "bbox_ymin": bbox[1],
            "bbox_xmax": bbox[2],
            "bbox_ymax": bbox[3],
            "morphology": morph,
            "eq_diameter_um": morph["eq_diameter_px"] * PIXEL_SIZE_UM,
            "area_um2": morph["area_px"] * PIXEL_SIZE_UM * PIXEL_SIZE_UM,
            "source": "demo_synthetic_precomputed_label",
        }
        det["explanation"] = particle_explanation(det["class_id"], morph)
        det["risk_hint"] = particle_risk_hint(det["class_id"], morph)
        detections.append(det)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return detections, elapsed_ms


def fiber_shape_decision(class_id: int, morph: dict) -> tuple[str, str]:
    """Return accept/reject/rescue for a YOLO fiber candidate."""
    if class_id not in {1, 5}:
        return "accept", ""
    axis_ratio = float(morph.get("principal_axis_ratio") or 1.0)
    circularity = float(morph.get("circularity") or 0.0)
    fill_ratio = float(morph.get("bbox_fill_ratio") or 0.0)
    aspect_ratio = float(morph.get("aspect_ratio") or 1.0)

    # Dense, nearly round objects are inconsistent with a visible fiber.
    round_dense = axis_ratio < 1.5 and circularity >= 1.0 and fill_ratio >= 0.50
    compact_blob = axis_ratio < 1.4 and aspect_ratio < 1.2 and circularity > 0.95 and fill_ratio > 0.38
    if round_dense or compact_blob:
        return "reject", (
            f"纤维形态审核未通过：主轴比 {axis_ratio:.2f}、圆度 {circularity:.2f}、"
            f"填充率 {fill_ratio:.2f}，目标更接近团块"
        )
    # A thin continuous object can rescue a low-confidence YOLO fiber candidate.
    if axis_ratio >= 5.0 and circularity <= 0.30 and fill_ratio <= 0.22:
        return "rescue", (
            f"纤维形态审核通过：主轴比 {axis_ratio:.2f}、圆度 {circularity:.2f}、"
            f"填充率 {fill_ratio:.2f}"
        )
    return "accept", ""


def real_yolo_detections(image_path: Path, settings: dict | None = None) -> tuple[list[dict], float]:
    settings = settings or DEFAULT_INFER_SETTINGS
    gray = load_gray(image_path)
    height, width = gray.shape
    sources: list[tuple[object, int, int]] = [(image_path, 0, 0)]
    pil_image = None
    if width == RAW_IMAGE_W and height == RAW_IMAGE_H:
        pil_image = Image.open(image_path).convert("RGB")
        sources = [
            (pil_image.crop((col * TILE_W, row * TILE_H, (col + 1) * TILE_W, (row + 1) * TILE_H)), col * TILE_W, row * TILE_H)
            for row in range(3)
            for col in range(4)
        ]
    raw_detections = []
    elapsed_ms = 0.0
    for source, offset_x, offset_y in sources:
        tile_detections, tile_ms = DETECTOR.predict(
            source,
            conf=min(settings["conf"], FIBER_CANDIDATE_CONF),
            iou=settings["iou"],
            imgsz=settings["imgsz"],
            max_det=settings["max_det"],
            device=settings["device"],
            agnostic_nms=settings["agnostic_nms"],
        )
        elapsed_ms += tile_ms
        for det in tile_detections:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
            det["source"] = "real_yolo_tiled_raw_frame" if len(sources) > 1 else "real_yolo_recognition_image"
            raw_detections.append(det)
    if pil_image is not None:
        pil_image.close()
    detections = []
    for idx, det in enumerate(raw_detections):
        bbox = det["bbox"]
        predicted_class_id = int(det["class_id"])
        morph = analyze_roi(gray, bbox, predicted_class_id)
        confidence = float(det.get("confidence") or 0.0)
        shape_decision, shape_reason = fiber_shape_decision(predicted_class_id, morph)
        fiber_rescued = shape_decision == "rescue" and confidence >= FIBER_CANDIDATE_CONF
        if confidence < settings["conf"] and not fiber_rescued:
            continue
        shape_rejected = shape_decision == "reject"
        confidence_rejected = confidence < settings["other_conf"] and not fiber_rescued
        rejected = shape_rejected or confidence_rejected
        output_class_id = 6 if rejected else (5 if fiber_rescued else predicted_class_id)
        rejection_reason = ""
        if shape_rejected:
            rejection_reason = shape_reason
        elif confidence_rejected:
            rejection_reason = f"最高类别置信度 {confidence:.2f} 低于其他/待复核阈值 {settings['other_conf']:.2f}"
        enriched = {
            **det,
            "id": idx,
            "particle_id": idx,
            "class_id": output_class_id,
            "class_name": CLASS_NAMES[output_class_id],
            "class_cn": CLASS_CN[output_class_id],
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": CLASS_NAMES[predicted_class_id],
            "predicted_class_cn": CLASS_CN[predicted_class_id],
            "classification_rejected": rejected,
            "classification_rescued": fiber_rescued,
            "shape_review": shape_decision,
            "shape_review_reason": shape_reason,
            "rejection_reason": rejection_reason,
            "bbox_xmin": bbox[0],
            "bbox_ymin": bbox[1],
            "bbox_xmax": bbox[2],
            "bbox_ymax": bbox[3],
            "morphology": morph,
            "eq_diameter_um": morph["eq_diameter_px"] * settings["pixel_size_um"],
            "area_um2": morph["area_px"] * settings["pixel_size_um"] * settings["pixel_size_um"],
        }
        enriched["explanation"] = particle_explanation(enriched["class_id"], morph)
        enriched["risk_hint"] = particle_risk_hint(enriched["class_id"], morph)
        detections.append(enriched)
    return detections, elapsed_ms


def class_stats(detections: list[dict], volume_ul: float | None = None) -> list[dict]:
    stats = []
    for cid in sorted(CLASS_NAMES):
        rows = [det for det in detections if det["class_id"] == cid]
        count = len(rows)
        concentration = count / (volume_ul / 1000.0) if volume_ul else None
        eq_mean = sum(det["eq_diameter_um"] for det in rows) / count if count else 0.0
        ar_mean = sum(det["morphology"]["aspect_ratio"] for det in rows) / count if count else 0.0
        circ_mean = sum(det["morphology"]["circularity"] for det in rows) / count if count else 0.0
        stats.append(
            {
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "class_cn": CLASS_CN[cid],
                "count": count,
                "concentration_per_ml": concentration,
                "eq_diameter_um_mean": eq_mean,
                "aspect_ratio_mean": ar_mean,
                "circularity_mean": circ_mean,
            }
        )
    return stats


def refresh_detection_labels(payload: dict) -> None:
    detections = payload.get("detections") or []
    for idx, det in enumerate(detections):
        cid = int(det.get("class_id", -1))
        det["id"] = idx
        det["particle_id"] = idx
        det["class_name"] = CLASS_NAMES.get(cid, f"class_{cid}")
        det["class_cn"] = CLASS_CN.get(cid, f"类别 {cid}")
        morph = det.get("morphology") or {}
        det["explanation"] = particle_explanation(cid, morph)
        det["risk_hint"] = particle_risk_hint(cid, morph)
    counts = Counter(int(det.get("class_id", -1)) for det in detections)
    payload["class_counts"] = {str(cid): counts.get(cid, 0) for cid in sorted(CLASS_NAMES)}
    payload["stats"] = class_stats(detections, payload.get("volume_ul"))
    payload["sample_level_summary"] = sample_summary(detections)


def image_path_for(split: str, image_name: str) -> Path:
    if split == "uploaded":
        return UPLOAD_DIR / image_name
    return DATASET / "images" / split / image_name


def sample_payload(split: str, image_name: str, settings: dict | None = None) -> dict:
    settings = settings or DEFAULT_INFER_SETTINGS
    image_path = image_path_for(split, image_name)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    status = detector_status()
    if split == "uploaded":
        cached = load_inference_cache(image_path, settings, status)
        if cached is not None:
            cached.update(upload_meta_for(image_name))
            return apply_manual_review(cached, image_path)
    if status["model_loaded"]:
        detections, elapsed_ms = real_yolo_detections(image_path, settings)
        mode = "real_yolo"
    else:
        detections, elapsed_ms = [], 0.0
        mode = "awaiting_real_yolo_weights"
    qinfo = quantification_info(split, image_path)
    volume_ul = qinfo["volume_ul"]
    counts = Counter(det["class_id"] for det in detections)
    summary = sample_summary(detections)
    payload = {
        "inference_mode": mode,
        "model_loaded": status["model_loaded"],
        "model_path": status["model_path"],
        "model_message": status["message"],
        "inference_time_ms": elapsed_ms,
        "image_name": image_name,
        "split": split,
        "image_url": image_url(split, image_name),
        **qinfo,
        "pixel_size_um": settings["pixel_size_um"],
        "inference_settings": settings,
        "total_particles": len(detections),
        "total_concentration_per_ml": len(detections) / (volume_ul / 1000.0) if volume_ul else None,
        "class_counts": {str(cid): counts.get(cid, 0) for cid in sorted(CLASS_NAMES)},
        "stats": class_stats(detections, volume_ul),
        "detections": detections,
        "sample_level_summary": summary,
    }
    if split == "uploaded":
        payload.update(upload_meta_for(image_name))
        save_inference_cache(image_path, settings, status, payload)
        return apply_manual_review(payload, image_path)
    return payload


def report_payload(split: str, image_name: str, settings: dict | None = None) -> dict:
    payload = sample_payload(split, image_name, settings)
    return {
        "report_type": "transformer_oil_particle_recognition",
        "generated_by": "MFI particle recognition app",
        "model": {
            "inference_mode": payload["inference_mode"],
            "model_loaded": payload["model_loaded"],
            "model_path": payload["model_path"],
            "inference_time_ms": payload["inference_time_ms"],
            "message": payload["model_message"],
        },
        "image": {
            "image_name": payload["image_name"],
            "split": payload["split"],
            "width": payload["width"],
            "height": payload["height"],
            "pixel_size_um": payload["pixel_size_um"],
            "volume_ul": payload["volume_ul"],
            "quantifiable": payload["quantifiable"],
            "quantification_reason": payload["quantification_reason"],
            "inference_settings": payload["inference_settings"],
        },
        "stats": payload["stats"],
        "detections": payload["detections"],
        "sample_level_summary": payload["sample_level_summary"],
    }


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(path)
        clean = parsed.path
        if clean == "/" or clean.startswith("/static/"):
            rel = "index.html" if clean == "/" else clean.removeprefix("/static/")
            return str(APP_DIR / "static" / rel)
        return str(APP_DIR / "static" / "index.html")

    def send_json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        username = os.environ.get("EXPERT_USERNAME", "").strip()
        password = os.environ.get("EXPERT_PASSWORD", "")
        if not username or not password:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            supplied_user, supplied_password = base64.b64decode(header[6:]).decode("utf-8").split(":", 1)
        except Exception:
            return False
        return supplied_user == username and supplied_password == password

    def _require_auth(self) -> bool:
        if self.path.split("?", 1)[0] == "/api/status" or self._authorized():
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="MFI Expert App"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path == "/api/status":
            return self.send_json(detector_status())
        if path == "/api/samples":
            return self.send_json(list_samples())
        if path == "/api/projects":
            return self.send_json(load_project_state())
        if path == "/api/manual-labels/export":
            return self.send_json(manual_label_export())
        if path.startswith("/api/manual-labels/"):
            image_name = urllib.parse.unquote(path.removeprefix("/api/manual-labels/"))
            return self.send_json(manual_review_for(image_name) or {"image_name": image_name, "review_scope": "none", "annotations": []})
        if path.startswith("/api/cache/"):
            parts = path.split("/")
            if len(parts) >= 5:
                _, _, _, split, image_name = parts[:5]
                if split != "uploaded":
                    return self.send_json({"error": "cache only supports uploaded images"}, 400)
                image_name = urllib.parse.unquote(image_name)
                image_path = image_path_for(split, image_name)
                if not image_path.exists():
                    return self.send_json({"error": "image not found"}, 404)
                payload = latest_inference_payload(image_name)
                if payload is None:
                    return self.send_json({"error": "not recognized"}, 404)
                return self.send_json(payload)
            return self.send_json({"error": "bad cache path"}, 400)
        if path.startswith("/api/sample/") or path.startswith("/api/report/"):
            parts = path.split("/")
            if len(parts) >= 5:
                _, _, _, split, image_name = parts[:5]
                try:
                    if path.startswith("/api/report/"):
                        return self.send_json(report_payload(split, urllib.parse.unquote(image_name), infer_settings(query)))
                    return self.send_json(sample_payload(split, urllib.parse.unquote(image_name), infer_settings(query)))
                except FileNotFoundError:
                    return self.send_json({"error": "image not found"}, 404)
                except Exception as exc:
                    return self.send_json({"error": str(exc)}, 500)
            return self.send_json({"error": "bad sample path"}, 400)
        if path.startswith("/api/image/"):
            parts = path.split("/")
            if len(parts) >= 5:
                _, _, _, split, image_name = parts[:5]
                image_path = image_path_for(split, urllib.parse.unquote(image_name))
                if not image_path.exists():
                    return self.send_json({"error": "image not found"}, 404)
                raw = image_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(image_path))[0] or "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
        return super().do_GET()

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/manual-labels/"):
            image_name = urllib.parse.unquote(parsed.path.removeprefix("/api/manual-labels/"))
            try:
                return self.send_json(save_manual_review(image_name, read_json_body(self.headers, self.rfile)))
            except FileNotFoundError:
                return self.send_json({"error": "image not found"}, 404)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 500)
        if parsed.path != "/api/upload":
            return self.send_json({"error": "not found"}, 404)
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                return self.send_json({"error": "multipart/form-data required"}, 400)
            filename, content, fields = parse_multipart_upload(self.headers, self.rfile)
            suffix = Path(filename).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                return self.send_json({"error": "unsupported image type"}, 400)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            stem = Path(filename).stem
            safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:80] or "uploaded"
            image_name = f"{int(time.time() * 1000)}_{safe_stem}{suffix}"
            out_path = UPLOAD_DIR / image_name
            with out_path.open("wb") as f:
                f.write(content)
            display_name = str(fields.get("display_name") or Path(filename).name).strip()[:120]
            group = str(fields.get("group") or "未分组").strip()[:80]
            note = str(fields.get("note") or "").strip()[:500]
            meta = load_upload_meta()
            meta[image_name] = {"display_name": display_name or Path(filename).name, "group": group or "未分组", "note": note}
            save_upload_meta(meta)
            return self.send_json({"image_name": image_name, "split": "uploaded", "image_url": image_url("uploaded", image_name), **meta[image_name]})
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def do_PATCH(self) -> None:
        if not self._require_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/projects":
            try:
                data = read_json_body(self.headers, self.rfile)
                save_project_state(data)
                return self.send_json(load_project_state())
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 500)
        if not parsed.path.startswith("/api/uploaded/"):
            return self.send_json({"error": "not found"}, 404)
        image_name = urllib.parse.unquote(parsed.path.removeprefix("/api/uploaded/"))
        image_path = image_path_for("uploaded", image_name)
        if not image_path.exists():
            return self.send_json({"error": "image not found"}, 404)
        try:
            data = read_json_body(self.headers, self.rfile)
            display_name = str(data.get("display_name") or image_name).strip()[:120]
            group = str(data.get("group") or "未分组").strip()[:80]
            note = str(data.get("note") or "").strip()[:500]
            meta = load_upload_meta()
            meta[image_name] = {
                "display_name": display_name or image_name,
                "group": group or "未分组",
                "note": note,
            }
            save_upload_meta(meta)
            return self.send_json({"image_name": image_name, **meta[image_name]})
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)

    def do_DELETE(self) -> None:
        if not self._require_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/manual-labels/"):
            image_name = urllib.parse.unquote(parsed.path.removeprefix("/api/manual-labels/"))
            return self.send_json({"deleted": delete_manual_review(image_name), "image_name": image_name})
        if not parsed.path.startswith("/api/uploaded/"):
            return self.send_json({"error": "not found"}, 404)
        image_name = urllib.parse.unquote(parsed.path.removeprefix("/api/uploaded/"))
        image_path = image_path_for("uploaded", image_name)
        try:
            if image_path.exists():
                image_path.unlink()
            if INFERENCE_CACHE_DIR.exists():
                stem = Path(image_name).stem
                for cache_path in INFERENCE_CACHE_DIR.glob(f"{stem}_*.json"):
                    try:
                        cache_path.unlink()
                    except FileNotFoundError:
                        pass
            meta = load_upload_meta()
            meta.pop(image_name, None)
            save_upload_meta(meta)
            return self.send_json({"deleted": True, "image_name": image_name})
        except Exception as exc:
            return self.send_json({"error": str(exc)}, 500)


def main() -> None:
import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    status = detector_status()
    print(f"MFI particle recognition app: http://{args.host}:{args.port}")
    print(f"Inference mode: {status['inference_mode']} | model_loaded={status['model_loaded']} | {status['message']}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

