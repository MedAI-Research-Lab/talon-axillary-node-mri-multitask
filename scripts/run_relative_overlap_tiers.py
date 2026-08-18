"""Relative GT-coverage/prediction-purity component sensitivity analysis.

This is a post-hoc descriptive sensitivity analysis from frozen test arrays;
it performs no training, threshold tuning, or model inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT.parent
OUTPUT = PROJECT / "outputs" / "benchmark_comparison" / "relative_overlap_analysis"
MODELS = {
    "HYBRID_TALON": PROJECT / "outputs" / "runs" / "HYBRID_TALON" / "seed_42" / "evaluation",
    "UNET_MTL_MASK_GUIDED": PARENT / "outputs" / "runs" / "UNET_MTL_MASK_GUIDED" / "seed_42" / "evaluation",
}
MINIMUM_AREAS = (1, 10, 25)
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def tier(coverage: float, purity: float, intersects: bool) -> str:
    if not intersects:
        return "no_overlap"
    quality = min(float(coverage), float(purity))
    if quality < 0.25:
        return "partial_below_25"
    if quality < 0.50:
        return "low_25_to_50"
    if quality < 0.75:
        return "moderate_50_to_75"
    return "strong_75_plus"


def labels(mask: np.ndarray, minimum_area: int = 1) -> tuple[np.ndarray, int]:
    raw, count = ndimage.label(mask.astype(bool), structure=STRUCTURE)
    cleaned = np.zeros_like(raw, dtype=np.int32)
    next_id = 0
    for component_id in range(1, count + 1):
        component = raw == component_id
        if int(component.sum()) < int(minimum_area):
            continue
        next_id += 1
        cleaned[component] = next_id
    return cleaned, next_id


def analyze_slice(model: str, row: pd.Series, array_root: Path, segmentation_threshold: float, minimum_area: int):
    with np.load(array_root / str(row["array_path"])) as stored:
        probability = stored["segmentation_probability"]
        ground_truth = stored["target_mask"] > 0.5
    # Prefer the exact locked binary mask saved during evaluation. This avoids
    # floating-point boundary ambiguity for pixels exactly equal to the stored
    # decimal representation of the validation-selected threshold.
    prediction_path = (
        array_root / "prediction_artifacts" / str(row["SubjectID"]) / str(row["CaseID"])
        / Path(str(row["ImageName"])).stem / "pred_mask.png"
    )
    prediction = np.asarray(Image.open(prediction_path).convert("L")) > 0 if prediction_path.exists() else probability >= segmentation_threshold
    pred_labels, pred_count = labels(prediction, minimum_area)
    gt_labels, gt_count = labels(ground_truth, 1)
    pred_records, gt_records = [], []

    # Prediction-component view: choose the GT pair with the highest balanced
    # overlap quality q=min(coverage,purity), with IoU as a deterministic tie-break.
    for pred_id in range(1, pred_count + 1):
        pred_component = pred_labels == pred_id
        pred_area = int(pred_component.sum())
        candidates = []
        for gt_id in range(1, gt_count + 1):
            gt_component = gt_labels == gt_id
            intersection = int((pred_component & gt_component).sum())
            if not intersection:
                continue
            gt_area = int(gt_component.sum())
            coverage = intersection / max(gt_area, 1)
            purity = intersection / max(pred_area, 1)
            union = int((pred_component | gt_component).sum())
            iou = intersection / max(union, 1)
            candidates.append((min(coverage, purity), iou, gt_id, coverage, purity, intersection))
        if candidates:
            quality, iou, gt_id, coverage, purity, intersection = max(candidates)
            category = tier(coverage, purity, True)
            intersecting_gt_count = len(candidates)
        else:
            quality = iou = coverage = purity = 0.0
            gt_id = None
            intersection = 0
            category = "off_target_fp"
            intersecting_gt_count = 0
        pred_records.append({
            "model": model, "minimum_area_px": minimum_area,
            "SubjectID": row["SubjectID"], "CaseID": row["CaseID"], "ClassName": row["ClassName"],
            "ImageName": row["ImageName"], "pred_component_id": pred_id,
            "best_gt_component_id": gt_id, "category": category, "area_px": pred_area,
            "intersection_px": intersection, "gt_coverage": coverage,
            "prediction_purity": purity, "quality_min": quality, "iou": iou,
            "intersecting_gt_count": intersecting_gt_count, "is_merge": intersecting_gt_count > 1,
        })

    # Unique-GT view: union all prediction components touching a GT. This avoids
    # counting a fragmented target more than once while retaining fragmentation.
    for gt_id in range(1, gt_count + 1):
        gt_component = gt_labels == gt_id
        gt_area = int(gt_component.sum())
        touching_ids = []
        for pred_id in range(1, pred_count + 1):
            if bool(((pred_labels == pred_id) & gt_component).any()):
                touching_ids.append(pred_id)
        if touching_ids:
            prediction_union = np.isin(pred_labels, touching_ids)
            intersection = int((prediction_union & gt_component).sum())
            prediction_area = int(prediction_union.sum())
            coverage = intersection / max(gt_area, 1)
            purity = intersection / max(prediction_area, 1)
            category = tier(coverage, purity, True)
        else:
            intersection = prediction_area = 0
            coverage = purity = 0.0
            category = "no_overlap"
        gt_records.append({
            "model": model, "minimum_area_px": minimum_area,
            "SubjectID": row["SubjectID"], "CaseID": row["CaseID"], "ClassName": row["ClassName"],
            "ImageName": row["ImageName"], "gt_component_id": gt_id,
            "category": category, "gt_area_px": gt_area, "intersection_px": intersection,
            "prediction_union_area_px": prediction_area, "gt_coverage": coverage,
            "prediction_purity": purity, "quality_min": min(coverage, purity),
            "touching_prediction_count": len(touching_ids), "is_fragmented": len(touching_ids) > 1,
        })
    return pred_records, gt_records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_pred, all_gt = [], []
    for model, evaluation_root in MODELS.items():
        thresholds = json.loads((evaluation_root / "reproducibility" / "locked_thresholds.json").read_text(encoding="utf-8"))
        segmentation_threshold = float(thresholds["segmentation_threshold"])
        array_root = evaluation_root / "predictions" / "test"
        manifest = pd.read_csv(array_root / "prediction_manifest.csv")
        for minimum_area in MINIMUM_AREAS:
            for _, row in manifest.iterrows():
                pred_records, gt_records = analyze_slice(model, row, array_root, segmentation_threshold, minimum_area)
                all_pred.extend(pred_records)
                all_gt.extend(gt_records)

    pred = pd.DataFrame(all_pred)
    gt = pd.DataFrame(all_gt)
    pred.to_csv(OUTPUT / "prediction_component_relative_overlap_records.csv", index=False)
    gt.to_csv(OUTPUT / "unique_gt_relative_overlap_records.csv", index=False)

    gt_categories = ["no_overlap", "partial_below_25", "low_25_to_50", "moderate_50_to_75", "strong_75_plus"]
    pred_categories = ["off_target_fp", "partial_below_25", "low_25_to_50", "moderate_50_to_75", "strong_75_plus"]
    gt_summary = gt.groupby(["model", "minimum_area_px", "category"], observed=True).size().unstack(fill_value=0)
    pred_summary = pred.groupby(["model", "minimum_area_px", "category"], observed=True).size().unstack(fill_value=0)
    for category in gt_categories:
        if category not in gt_summary:
            gt_summary[category] = 0
    for category in pred_categories:
        if category not in pred_summary:
            pred_summary[category] = 0
    gt_summary = gt_summary[gt_categories].reset_index()
    pred_summary = pred_summary[pred_categories].reset_index()
    gt_summary["gt_total"] = gt_summary[gt_categories].sum(axis=1)
    pred_summary["prediction_component_total"] = pred_summary[pred_categories].sum(axis=1)
    gt_summary.to_csv(OUTPUT / "unique_gt_tier_summary.csv", index=False)
    pred_summary.to_csv(OUTPUT / "prediction_component_tier_summary.csv", index=False)

    case_gt = gt.groupby(["model", "minimum_area_px", "SubjectID", "CaseID", "ClassName", "category"], observed=True).size().unstack(fill_value=0)
    for category in gt_categories:
        if category not in case_gt:
            case_gt[category] = 0
    case_gt = case_gt[gt_categories].reset_index()
    case_gt["gt_total"] = case_gt[gt_categories].sum(axis=1)
    case_gt.to_csv(OUTPUT / "case_unique_gt_tier_counts.csv", index=False)

    metadata = {
        "status": "completed", "training_performed": False, "model_inference_performed": False,
        "source": "frozen locked-test probability arrays", "connectivity": 8,
        "tier_variable": "min(gt_coverage, prediction_purity)",
        "tiers": {
            "partial_below_25": "overlap exists but q < 0.25",
            "low_25_to_50": "0.25 <= q < 0.50",
            "moderate_50_to_75": "0.50 <= q < 0.75",
            "strong_75_plus": "q >= 0.75",
        },
        "minimum_area_sensitivity_px": list(MINIMUM_AREAS),
        "note": "GT-level tiers use the union of every prediction component touching that GT, so fragmented predictions do not multiply the GT count.",
    }
    (OUTPUT / "relative_overlap_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(gt_summary.to_csv(index=False))
    print(pred_summary.to_csv(index=False))


if __name__ == "__main__":
    main()
