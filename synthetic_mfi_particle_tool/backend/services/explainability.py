from __future__ import annotations

import numpy as np

from .yolo_detector import CLASS_CN, CLASS_NAMES


def particle_explanation(class_id: int, morph: dict) -> str:
    ar = morph.get("aspect_ratio", 0)
    circ = morph.get("circularity", 0)
    gray = morph.get("gray_mean", 0)
    sharp = morph.get("edge_sharpness", 0)
    if class_id == 0:
        return f"模型将目标识别为气泡；气泡通常具有较高圆度或环状边缘，本目标圆度为 {circ:.2f}。"
    if class_id == 1:
        return f"模型将目标识别为短纤维素；其长宽比为 {ar:.2f}，轮廓通常呈短条状或弯曲带状。"
    if class_id == 2:
        return f"模型将目标识别为铜颗粒；局部边缘清晰度为 {sharp:.1f}，常呈片状或不规则块状。"
    if class_id == 3:
        return f"模型将目标识别为浅色碳颗粒；目标对比度较弱，区域平均灰度为 {gray:.1f}。"
    if class_id == 4:
        return f"模型将目标识别为深色碳颗粒；区域平均灰度为 {gray:.1f}，通常表现为暗色碎片或团块。"
    if class_id == 5:
        return f"模型将目标识别为长纤维素；其长宽比为 {ar:.2f}，轮廓通常呈细长、弯曲或带状。"
    return "该目标已被检测到，但模型对六个明确类别均缺乏足够把握，因此标为“其他/待复核”，建议查看 ROI 后人工确认。"


def particle_risk_hint(class_id: int, morph: dict) -> str:
    eq = morph.get("eq_diameter_px", 0) * 0.7
    if class_id in (1, 5):
        return "纤维素颗粒可能来源于绝缘纸老化、脱落或检修污染；长纤维在电场中更易取向和搭桥，应结合水分、介损和重复取样复核。"
    if class_id in (3, 4):
        return "碳颗粒可能与局部过热、放电或油纸裂解有关；尺寸较大或数量持续增加时，应结合 DGA 与局放结果排查。"
    if class_id == 2:
        return "铜颗粒可能来源于制造残留、机械磨损或导电部件异常，建议人工复核典型目标。"
    if class_id == 0:
        return "气泡可能来自取样扰动、析气或放电产气；应先排除制样与流路引入的伪影。"
    if class_id == 6:
        return f"该目标等效直径约 {eq:.1f} μm，当前类别不确定，不宜直接用于故障归因，建议人工复核或补充标注后再训练。"
    return f"该目标等效直径约 {eq:.1f} μm，建议结合连续图像统计和其他油化试验综合判断。"


def sample_summary(detections: list[dict]) -> dict:
    total = len(detections)
    class_counts = {str(cid): 0 for cid in CLASS_NAMES}
    diameters: list[float] = []
    for det in detections:
        cid = int(det["class_id"])
        class_counts[str(cid)] = class_counts.get(str(cid), 0) + 1
        diameters.append(float(det.get("eq_diameter_um", 0)))
    ratios = {cid: (count / total if total else 0.0) for cid, count in class_counts.items()}
    dominant_id = max(class_counts, key=class_counts.get) if total else "0"
    dvals = np.asarray(diameters or [0.0], dtype=np.float32)
    fiber_ratio = ratios.get("1", 0) + ratios.get("5", 0)
    carbon_ratio = ratios.get("3", 0) + ratios.get("4", 0)
    copper_ratio = ratios.get("2", 0)
    other_ratio = ratios.get("6", 0)
    hints = []
    if carbon_ratio > 0.45:
        hints.append("碳颗粒占比较高，建议排查过热、放电或油纸裂解。")
    if fiber_ratio > 0.35:
        hints.append("纤维素颗粒占比较高，建议关注绝缘纸老化、脱落或取样污染。")
    if copper_ratio > 0.12:
        hints.append("铜颗粒占比较高，建议关注金属磨损或制造残留。")
    if other_ratio > 0.15:
        hints.append("其他/待复核目标占比较高，本批次分类可靠性不足，建议人工复核并补充训练样本。")
    if not hints:
        hints.append("当前单图未显示明显的单一主导风险；结论应以完整连续采集批次为准。")
    high_risk = sum(
        1
        for det in detections
        if det["class_id"] in (2, 4, 5) or det.get("eq_diameter_um", 0) > 45
    )
    score = (2 if high_risk > 12 else 1 if high_risk > 4 else 0) + (1 if total > 80 else 0)
    overall = "高" if score >= 3 else "中" if score >= 1 else "低"
    return {
        "total_particles": total,
        "class_counts": class_counts,
        "class_ratios": ratios,
        "d10": float(np.percentile(dvals, 10)),
        "d50": float(np.percentile(dvals, 50)),
        "d90": float(np.percentile(dvals, 90)),
        "max_eq_diameter": float(dvals.max()),
        "high_risk_particle_count": high_risk,
        "dominant_particle_type": CLASS_CN.get(int(dominant_id), "-"),
        "overall_risk_level": overall,
        "risk_summary": "".join(hints),
        "recommended_actions": [
            "使用完整 4096×3000 原始帧组成连续采集批次后再计算颗粒浓度。",
            "对其他/待复核、长纤维、铜颗粒和聚集颗粒进行人工复核，并结合 DGA、水分、介损和局放结果判断。",
        ],
    }
