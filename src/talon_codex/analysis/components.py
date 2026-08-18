"""2D connected-component error taxonomy for lesion segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


@dataclass
class ComponentAnalysis:
    predicted_records: list[dict[str, Any]]
    ground_truth_records: list[dict[str, Any]]
    masks: dict[str, np.ndarray]


def connected_labels(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    return ndimage.label(mask.astype(bool), structure=structure)


def _component_geometry(component: np.ndarray) -> dict[str, Any]:
    coordinates = np.argwhere(component)
    if not len(coordinates):
        return {"area_px": 0, "centroid_y": np.nan, "centroid_x": np.nan, "bbox": None}
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    cy, cx = coordinates.mean(axis=0)
    return {"area_px": int(len(coordinates)), "centroid_y": float(cy), "centroid_x": float(cx), "bbox": f"{x0},{y0},{x1},{y1}"}


def _remove_small(labels: np.ndarray, count: int, minimum_area: int) -> tuple[np.ndarray, int]:
    cleaned = np.zeros_like(labels, dtype=np.int32)
    next_label = 0
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) >= minimum_area:
            next_label += 1
            cleaned[component] = next_label
    return cleaned, next_label


def analyze_components(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    minimum_area_px: int,
    connectivity: int = 8,
    poor_match_iou: float = 0.10,
    poor_match_purity: float = 0.10,
    pixel_spacing_mm: tuple[float, float] | None = None,
) -> ComponentAnalysis:
    """Classify predicted components as matched, poorly matched or off-target FP.

    These are 2D slice occurrences, not proof of unique 3D lesions. Case totals
    should therefore be described as component occurrences unless 3D linkage is added.
    """
    pred_labels, pred_count = connected_labels(predicted_mask, connectivity)
    gt_labels, gt_count = connected_labels(ground_truth_mask, connectivity)
    pred_labels, pred_count = _remove_small(pred_labels, pred_count, minimum_area_px)
    gt_labels, gt_count = _remove_small(gt_labels, gt_count, 1)
    off_target = np.zeros_like(predicted_mask, dtype=np.uint8)
    poor = np.zeros_like(predicted_mask, dtype=np.uint8)
    matched_excess = np.zeros_like(predicted_mask, dtype=np.uint8)
    predicted_records: list[dict[str, Any]] = []
    match_matrix = np.zeros((pred_count, gt_count), dtype=bool)
    spacing_area = None if pixel_spacing_mm is None else float(pixel_spacing_mm[0] * pixel_spacing_mm[1])
    if pred_count == 0:
        predicted_records.append({
            "pred_component_id": 0, "category": "empty_prediction", "closest_gt_component_id": None,
            "matched_gt_component_id": None, "max_iou": 0.0, "purity": 0.0,
            "intersecting_gt_count": 0, "matched_gt_count": 0, "is_merge": False,
            "minimum_area_px": int(minimum_area_px), "area_px": 0, "centroid_y": np.nan,
            "centroid_x": np.nan, "bbox": None, "matched_excess_area_px": 0,
            "area_mm2": 0.0 if spacing_area is not None else np.nan,
        })
    for pred_id in range(1, pred_count + 1):
        component = pred_labels == pred_id
        area = int(component.sum())
        best_gt = 0
        best_iou = best_purity = 0.0
        intersecting_gt: list[int] = []
        for gt_id in range(1, gt_count + 1):
            gt_component = gt_labels == gt_id
            intersection = int((component & gt_component).sum())
            if intersection == 0:
                continue
            intersecting_gt.append(gt_id)
            union = int((component | gt_component).sum())
            iou = intersection / max(union, 1)
            purity = intersection / max(area, 1)
            if iou >= poor_match_iou and purity >= poor_match_purity:
                match_matrix[pred_id - 1, gt_id - 1] = True
            if iou > best_iou:
                best_gt, best_iou, best_purity = gt_id, iou, purity
        if not intersecting_gt:
            category = "off_target_fp"
            off_target[component] = 1
        elif best_iou < poor_match_iou or best_purity < poor_match_purity:
            category = "poorly_matched"
            poor[component] = 1
        else:
            category = "matched"
            if best_gt:
                matched_excess[component & ~(gt_labels == best_gt)] = 1
        accepted_gt_count = int(match_matrix[pred_id - 1].sum()) if gt_count else 0
        record = {
            "pred_component_id": pred_id, "category": category, "closest_gt_component_id": best_gt or None,
            "matched_gt_component_id": best_gt if category == "matched" else None,
            "max_iou": float(best_iou), "purity": float(best_purity), "intersecting_gt_count": len(intersecting_gt),
            "matched_gt_count": accepted_gt_count, "is_merge": accepted_gt_count > 1,
            "matched_excess_area_px": int((component & ~(gt_labels == best_gt)).sum()) if category == "matched" and best_gt else 0,
            "minimum_area_px": int(minimum_area_px),
            **_component_geometry(component),
        }
        record["area_mm2"] = np.nan if spacing_area is None else record["area_px"] * spacing_area
        predicted_records.append(record)
    ground_truth_records: list[dict[str, Any]] = []
    for gt_id in range(1, gt_count + 1):
        component = gt_labels == gt_id
        matched_predictions = np.flatnonzero(match_matrix[:, gt_id - 1]) + 1 if pred_count else np.array([], dtype=int)
        record = {
            "gt_component_id": gt_id, "matched_prediction_count": int(len(matched_predictions)),
            "is_missed": len(matched_predictions) == 0, "is_fragmented": len(matched_predictions) > 1,
            **_component_geometry(component),
        }
        record["area_mm2"] = np.nan if spacing_area is None else record["area_px"] * spacing_area
        ground_truth_records.append(record)
    false_negative = ((ground_truth_mask > 0) & ~(predicted_mask > 0)).astype(np.uint8)
    true_positive = ((ground_truth_mask > 0) & (predicted_mask > 0)).astype(np.uint8)
    return ComponentAnalysis(
        predicted_records, ground_truth_records,
        {"true_positive": true_positive, "off_target_fp": off_target, "poorly_matched": poor, "matched_excess": matched_excess, "false_negative": false_negative, "component_labels": pred_labels.astype(np.int32)},
    )


def component_sensitivity_analysis(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    minimum_areas_px: Iterable[int] = (1, 10, 25),
    **kwargs: Any,
) -> dict[int, ComponentAnalysis]:
    return {int(area): analyze_components(predicted_mask, ground_truth_mask, int(area), **kwargs) for area in minimum_areas_px}


def select_component_thresholds(
    validation_rows,
    segmentation_threshold: float,
    iou_grid: Iterable[float],
    purity_grid: Iterable[float],
    connectivity: int = 8,
) -> tuple[dict[str, float], "Any"]:
    """Lock component matching rules using validation masks only."""
    import pandas as pd
    records = []
    for iou in iou_grid:
        for purity in purity_grid:
            matched = off_target = poor = missed = gt_total = 0
            for row in validation_rows.itertuples(index=False):
                analysis = analyze_components(
                    row.seg_probability_map >= segmentation_threshold, row.target_mask,
                    minimum_area_px=1, connectivity=connectivity,
                    poor_match_iou=float(iou), poor_match_purity=float(purity),
                )
                matched += sum(record["category"] == "matched" for record in analysis.predicted_records)
                off_target += sum(record["category"] == "off_target_fp" for record in analysis.predicted_records)
                poor += sum(record["category"] == "poorly_matched" for record in analysis.predicted_records)
                missed += sum(bool(record["is_missed"]) for record in analysis.ground_truth_records)
                gt_total += len(analysis.ground_truth_records)
            precision = matched / max(matched + off_target + poor, 1)
            recall = (gt_total - missed) / max(gt_total, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            records.append({"poor_match_iou": float(iou), "poor_match_purity": float(purity), "component_precision": precision, "component_recall": recall, "component_f1": f1})
    table = pd.DataFrame(records).sort_values(["component_f1", "poor_match_iou", "poor_match_purity"], ascending=[False, True, True])
    best = table.iloc[0]
    return {"poor_match_iou": float(best["poor_match_iou"]), "poor_match_purity": float(best["poor_match_purity"])}, table.reset_index(drop=True)
