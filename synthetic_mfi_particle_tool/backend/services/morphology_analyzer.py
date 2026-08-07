from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _bbox_clip(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    return max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 2:
        return points
    pts = sorted(map(tuple, points.tolist()))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return float(len(poly))
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _segment_roi(roi: np.ndarray, class_id: int) -> np.ndarray:
    if roi.size == 0:
        return np.zeros_like(roi, dtype=bool)
    arr = roi.astype(np.float32)
    arr_blur = np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    bg = float(np.percentile(arr_blur, 82))
    std = float(arr_blur.std())
    if class_id == 0:
        mask = np.abs(arr_blur - bg) > max(2.5, std * 0.25)
    elif class_id == 3:
        mask = arr_blur < bg - max(2.0, std * 0.16)
    else:
        mask = arr_blur < bg - max(4.0, std * 0.32)
    if int(mask.sum()) < 4:
        mask = arr_blur < np.percentile(arr_blur, 45)
    if int(mask.sum()) < 4:
        mask = np.ones_like(arr_blur, dtype=bool)
    return mask


def analyze_roi(gray: np.ndarray, bbox: list[int], class_id: int) -> dict:
    h, w = gray.shape
    xmin, ymin, xmax, ymax = _bbox_clip(bbox, w, h)
    roi = gray[ymin:ymax, xmin:xmax].astype(np.float32)
    mask = _segment_roi(roi, class_id)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "area_px": 0.0,
            "eq_diameter_px": 0.0,
            "perimeter_px": 0.0,
            "aspect_ratio": 1.0,
            "circularity": 0.0,
            "elongation": 0.0,
            "solidity": 0.0,
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "edge_sharpness": 0.0,
            "bbox_fill_ratio": 0.0,
            "principal_axis_ratio": 1.0,
        }
    area = float(len(xs))
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    aspect_ratio = float(max(bw, bh) / max(1, min(bw, bh)))
    padded = np.pad(mask, 1)
    center = padded[1:-1, 1:-1]
    eroded = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = center & ~eroded
    perimeter = float(np.count_nonzero(edge))
    circularity = float(4.0 * math.pi * area / max(1.0, perimeter * perimeter))
    points = np.column_stack([xs, ys]).astype(np.float32)
    principal_axis_ratio = 1.0
    if len(points) >= 3:
        eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
        principal_axis_ratio = float(math.sqrt(max(eigenvalues[-1], 0.0) / max(eigenvalues[0], 1e-6)))
    hull = _convex_hull(points)
    hull_area = max(_polygon_area(hull), area)
    vals = roi[mask]
    gy, gx = np.gradient(roi)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "area_px": area,
        "eq_diameter_px": float(math.sqrt(4.0 * area / math.pi)),
        "perimeter_px": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": min(circularity, 1.2),
        "elongation": float(1.0 - min(bw, bh) / max(bw, bh)),
        "solidity": float(area / max(hull_area, 1.0)),
        "gray_mean": float(vals.mean()),
        "gray_std": float(vals.std()),
        "edge_sharpness": float(grad[edge].mean()) if np.any(edge) else float(grad.mean()),
        "bbox_fill_ratio": float(area / max(1, bw * bh)),
        "principal_axis_ratio": principal_axis_ratio,
    }


def load_gray(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
