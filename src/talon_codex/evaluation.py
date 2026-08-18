"""Locked-test evaluation, breast/case aggregation, component analysis and XAI export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import importlib.metadata
import json
import platform

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .analysis.components import component_sensitivity_analysis, select_component_thresholds
from .analysis.statistics import (
    apply_probability_calibrator, calibration_bins, classification_metrics, curve_tables,
    fit_probability_calibrator, patient_cluster_bootstrap, segmentation_metrics,
    select_segmentation_threshold,
)
from .analysis.xai import GradCAM, xai_overlap_metrics
from .config import ResearchConfig
from .reporting import render_evaluation_figures, write_q1_workbook


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def collect_raw_predictions(model: nn.Module, loader: Iterable[Mapping[str, Any]], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        outputs = model(batch["image"])
        probabilities = torch.softmax(outputs["cls_logits"], dim=1)[:, 1].cpu().numpy()
        segmentation = torch.sigmoid(outputs["lesion_logits"]).cpu().numpy()[:, 0]
        targets = batch["mask"].cpu().numpy()[:, 0]
        for index in range(len(probabilities)):
            rows.append({
                "SubjectID": raw_batch["SubjectID"][index], "CaseID": raw_batch["CaseID"][index],
                "MaskedPatientName": raw_batch["MaskedPatientName"][index], "ImageName": raw_batch["ImageName"][index],
                "ClassName": raw_batch["ClassName"][index], "RoiSizeBin": raw_batch["RoiSizeBin"][index],
                "ResolvedImagePath": raw_batch["ResolvedImagePath"][index],
                "ResolvedRoiPath": raw_batch["ResolvedRoiPath"][index],
                "PixelSpacingXmm": float(raw_batch["PixelSpacingXmm"][index]),
                "PixelSpacingYmm": float(raw_batch["PixelSpacingYmm"][index]),
                "SliceThicknessMm": float(raw_batch["SliceThicknessMm"][index]),
                "class_id": int(batch["class_id"][index].cpu()), "class_probability": float(probabilities[index]),
                "seg_probability_map": segmentation[index].astype(np.float32), "target_mask": targets[index].astype(np.uint8),
                "predicted_lesion_fraction": float(segmentation[index].mean()),
            })
    return pd.DataFrame(rows)


def case_aggregation(slice_rows: pd.DataFrame, strategy: str, top_k: int = 5) -> pd.DataFrame:
    """Aggregate slices within a breast/case while retaining the underlying subject cluster."""
    records: list[dict[str, Any]] = []
    for case_id, group in slice_rows.groupby("CaseID", sort=True):
        if group["class_id"].nunique() != 1:
            raise ValueError(f"Case {case_id} contains multiple class labels.")
        if group["SubjectID"].nunique() != 1:
            raise ValueError(f"Case {case_id} maps to multiple subjects.")
        if strategy == "mean_all":
            selected = group
        elif strategy == "topk_confidence":
            selected = group.assign(_rank=(group["class_probability"] - 0.5).abs()).nlargest(top_k, "_rank")
        elif strategy == "topk_lesion":
            selected = group.nlargest(top_k, "predicted_lesion_fraction")
        else:
            raise KeyError(f"Unknown case aggregation: {strategy}")
        records.append({
            "SubjectID": group["SubjectID"].iloc[0], "CaseID": case_id,
            "MaskedPatientName": group["MaskedPatientName"].iloc[0], "ClassName": group["ClassName"].iloc[0],
            "class_id": int(group["class_id"].iloc[0]), "class_probability": float(selected["class_probability"].mean()),
            "n_slices_total": int(len(group)), "n_slices_aggregated": int(len(selected)), "aggregation": strategy,
        })
    return pd.DataFrame(records)


def _safe_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in Path(str(value)).stem)


def _save_mask(path: Path, mask: np.ndarray, probability: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(mask, 0, 1)
    encoded = (array * (65535 if probability else 255)).astype(np.uint16 if probability else np.uint8)
    cv2.imwrite(str(path), encoded)


def _save_label_map(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(labels, dtype=np.uint16))


def _save_prediction_overlay(path: Path, image_path: str, target: np.ndarray, prediction: np.ndarray) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_AREA)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image)
    if target.any(): axis.contour(target, levels=[0.5], colors=["lime"], linewidths=1.2)
    if prediction.any(): axis.contour(prediction, levels=[0.5], colors=["red"], linewidths=1.0)
    axis.axis("off"); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_prediction_arrays(
    rows: pd.DataFrame,
    evaluation_dir: Path,
    split: str,
    dataset_root: Path,
) -> pd.DataFrame:
    """Persist model-independent numeric arrays needed to rebuild test visuals."""
    store_root = evaluation_dir / "predictions" / split
    arrays_root = store_root / "arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows.itertuples(index=False), start=1):
        stem = f"{ordinal:05d}_{_safe_stem(row.ImageName)}"
        relative_array = Path("arrays") / str(row.SubjectID) / str(row.CaseID) / f"{stem}.npz"
        array_path = store_root / relative_array
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            array_path,
            segmentation_probability=np.asarray(row.seg_probability_map, dtype=np.float16),
            target_mask=np.asarray(row.target_mask, dtype=np.uint8),
            class_probability=np.asarray(row.class_probability, dtype=np.float32),
            class_id=np.asarray(row.class_id, dtype=np.int8),
        )
        source_image = Path(str(row.ResolvedImagePath)).resolve()
        source_mask = Path(str(row.ResolvedRoiPath)).resolve()
        try:
            source_image_reference = source_image.relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            source_image_reference = source_image.name
        try:
            source_mask_reference = source_mask.relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            source_mask_reference = source_mask.name
        records.append({
            "SubjectID": row.SubjectID, "CaseID": row.CaseID, "MaskedPatientName": row.MaskedPatientName,
            "ImageName": row.ImageName, "ClassName": row.ClassName, "RoiSizeBin": row.RoiSizeBin,
            "class_id": int(row.class_id), "class_probability": float(row.class_probability),
            "predicted_lesion_fraction": float(row.predicted_lesion_fraction),
            "PixelSpacingXmm": float(row.PixelSpacingXmm), "PixelSpacingYmm": float(row.PixelSpacingYmm),
            "SliceThicknessMm": float(row.SliceThicknessMm),
            "array_path": relative_array.as_posix(),
            "array_sha256": _file_sha256(array_path),
            "source_image_reference": source_image_reference,
            "source_mask_reference": source_mask_reference,
            "source_image_sha256": _file_sha256(source_image),
            "source_mask_sha256": _file_sha256(source_mask),
        })
    manifest = pd.DataFrame(records)
    manifest.to_csv(store_root / "prediction_manifest.csv", index=False)
    return manifest


def write_artifact_manifest(evaluation_dir: Path, config: ResearchConfig, experiment: str, seed: int) -> pd.DataFrame:
    """Hash stored numeric artifacts so regenerated figures have traceable inputs."""
    reproducibility_dir = evaluation_dir / "reproducibility"
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = evaluation_dir.parent / "checkpoints" / "best_overall.pt"
    package_names = [
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy",
        "opencv-python", "albumentations", "matplotlib", "seaborn", "openpyxl",
    ]
    package_versions = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = None
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": package_versions,
    }
    (reproducibility_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    config.snapshot(reproducibility_dir / "resolved_config.json", {
        "experiment": experiment, "seed": int(seed), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "checkpoint_sha256": _file_sha256(checkpoint) if checkpoint.exists() else None,
        "metadata_sha256": _file_sha256(config.metadata_path) if config.metadata_path.exists() else None,
    })
    records = []
    for path in sorted(evaluation_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        records.append({
            "relative_path": path.relative_to(evaluation_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        })
    manifest = pd.DataFrame(records)
    manifest.to_csv(reproducibility_dir / "artifact_manifest.csv", index=False)
    return manifest


def analyze_and_export_segmentation(
    rows: pd.DataFrame,
    threshold: float,
    output_dir: Path,
    component_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slice_metrics: list[dict[str, Any]] = []
    predicted_components: list[dict[str, Any]] = []
    gt_components: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        probability = row.seg_probability_map
        target = row.target_mask.astype(bool)
        prediction = probability >= threshold
        identifiers = {
            "SubjectID": row.SubjectID, "CaseID": row.CaseID, "MaskedPatientName": row.MaskedPatientName,
            "ImageName": row.ImageName, "ClassName": row.ClassName, "RoiSizeBin": row.RoiSizeBin,
        }
        spacing_x, spacing_y = float(row.PixelSpacingXmm), float(row.PixelSpacingYmm)
        pixel_area_mm2 = spacing_x * spacing_y if np.isfinite(spacing_x) and np.isfinite(spacing_y) else np.nan
        slice_metrics.append({**identifiers, **segmentation_metrics(prediction, target)})
        artifact_dir = output_dir / "prediction_artifacts" / row.SubjectID / row.CaseID
        stem = _safe_stem(row.ImageName)
        slice_dir = artifact_dir / stem
        _save_mask(slice_dir / "probability_map.png", probability, probability=True)
        np.savez_compressed(slice_dir / "probability_map.npz", probability=np.asarray(probability, dtype=np.float16))
        _save_mask(slice_dir / "pred_mask.png", prediction)
        _save_mask(slice_dir / "gt_mask.png", target)
        _save_prediction_overlay(slice_dir / "visual_overlay.png", row.ResolvedImagePath, target, prediction)
        analyses = component_sensitivity_analysis(
            prediction, target, minimum_areas_px=component_config["minimum_areas_px"],
            connectivity=int(component_config["connectivity"]), poor_match_iou=float(component_config["poor_match_iou"]),
            poor_match_purity=float(component_config["poor_match_purity"]),
        )
        for minimum_area, analysis in analyses.items():
            predicted_components.extend([{
                **identifiers, **record,
                "area_mm2": float(record.get("area_px", np.nan) * pixel_area_mm2) if np.isfinite(pixel_area_mm2) else np.nan,
                "matched_excess_area_mm2": float(record.get("matched_excess_area_px", np.nan) * pixel_area_mm2) if np.isfinite(pixel_area_mm2) else np.nan,
            } for record in analysis.predicted_records])
            gt_components.extend([{
                **identifiers, "minimum_area_px": minimum_area, **record,
                "area_mm2": float(record.get("area_px", np.nan) * pixel_area_mm2) if np.isfinite(pixel_area_mm2) else np.nan,
            } for record in analysis.ground_truth_records])
            for mask_name, mask in analysis.masks.items():
                exact_names = {
                    "true_positive": "tp_mask.png", "false_negative": "fn_mask.png",
                    "off_target_fp": "fp_off_target_mask.png", "matched_excess": "matched_excess_mask.png",
                    "poorly_matched": "poorly_matched_mask.png", "component_labels": "component_label_map.png",
                }
                destination = slice_dir / f"min_area_{minimum_area}" / exact_names[mask_name]
                if mask_name == "component_labels":
                    _save_label_map(destination, mask)
                else:
                    _save_mask(destination, mask / max(float(mask.max()), 1.0))
    return pd.DataFrame(slice_metrics), pd.DataFrame(predicted_components), pd.DataFrame(gt_components)


def _classification_ci_table(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    stats_cfg = config.section("statistics")
    records = []
    point = classification_metrics(frame["class_id"], frame["class_probability"], float(config.section("selection")["classification_threshold"]), int(stats_cfg["ece_bins"]))
    for metric in ("auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "macro_f1", "brier", "ece", "calibration_slope", "calibration_intercept"):
        result = patient_cluster_bootstrap(
            frame,
            lambda sampled, name=metric: float(classification_metrics(sampled["class_id"], sampled["class_probability"], float(config.section("selection")["classification_threshold"]), int(stats_cfg["ece_bins"]))[name]),
            iterations=int(stats_cfg["bootstrap_iterations"]), confidence_level=float(stats_cfg["confidence_level"]), seed=config.seed,
        )
        records.append({"metric": metric, **result, "point_check": point[metric]})
    return pd.DataFrame(records)


def _segmentation_ci_table(slice_metrics: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    stats_cfg = config.section("statistics")
    records = []
    for metric in ("dice", "iou", "pixel_sensitivity", "pixel_specificity", "pixel_precision", "gt_coverage", "prediction_purity", "empty_prediction", "missed_target", "pred_gt_area_ratio"):
        result = patient_cluster_bootstrap(
            slice_metrics, lambda sampled, name=metric: float(sampled[name].mean()),
            iterations=int(stats_cfg["bootstrap_iterations"]), confidence_level=float(stats_cfg["confidence_level"]), seed=config.seed,
        )
        records.append({"metric": metric, **result})
    return pd.DataFrame(records)


def register_final_test_access(config: ResearchConfig, experiment: str, seed: int) -> Path:
    """Prevent accidental repeated peeking at the locked test set for one run."""
    registry = config.output_root / "test_access_registry" / experiment / f"seed_{seed}.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    enforce = bool(config.section("test_access").get("enforce_single_evaluation_per_run", True))
    if enforce and registry.exists():
        raise RuntimeError(f"Locked test was already evaluated for this run: {registry}")
    registry.write_text(json.dumps({"experiment": experiment, "seed": int(seed), "purpose": "prespecified final evaluation"}, indent=2), encoding="utf-8")
    return registry


def _component_bootstrap_table(case_report: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    stats_cfg = config.section("statistics")
    metrics = [
        "off_target_component_occurrences", "poorly_matched_occurrences",
        "missed_target_occurrences", "empty_prediction_slices", "off_target_area_px_total",
    ]
    rows = []
    for metric in metrics:
        result = patient_cluster_bootstrap(
            case_report, lambda sampled, name=metric: float(sampled[name].mean()),
            iterations=int(stats_cfg["bootstrap_iterations"]), confidence_level=float(stats_cfg["confidence_level"]), seed=config.seed,
        )
        rows.append({"metric": metric, **result})
    return pd.DataFrame(rows)


def segmentation_group_summary(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dice", "iou", "pixel_precision", "pixel_sensitivity", "gt_coverage", "prediction_purity", "pred_gt_area_ratio", "empty_prediction", "missed_target"]
    records = []
    group_specs = [("overall", []), ("class", ["ClassName"]), ("roi_size", ["RoiSizeBin"]), ("class_roi_size", ["ClassName", "RoiSizeBin"])]
    for group_type, columns in group_specs:
        groups = [((), slice_metrics)] if not columns else slice_metrics.groupby(columns, observed=True)
        for key, group in groups:
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(columns, key))
            for metric in metrics:
                values = group[metric].astype(float)
                records.append({
                    "group_type": group_type, **labels, "metric": metric, "n": len(values),
                    "mean": values.mean(), "sd": values.std(ddof=1), "median": values.median(),
                    "q1": values.quantile(0.25), "q3": values.quantile(0.75),
                })
    return pd.DataFrame(records)


def build_case_error_report(
    slice_metrics: pd.DataFrame,
    predicted_components: pd.DataFrame,
    gt_components: pd.DataFrame,
    case_predictions: pd.DataFrame,
) -> pd.DataFrame:
    report = slice_metrics.groupby(["SubjectID", "CaseID", "MaskedPatientName", "ClassName"], observed=True).agg(
        slice_count=("ImageName", "size"), mean_dice=("dice", "mean"), median_dice=("dice", "median"),
        mean_iou=("iou", "mean"), empty_prediction_slices=("empty_prediction", "sum"),
        total_false_positive_area_px=("false_positive_area_px", "sum"),
        roi_size_bins=("RoiSizeBin", lambda values: "|".join(sorted(set(map(str, values))))),
    ).reset_index()
    primary_pred = predicted_components[predicted_components["minimum_area_px"] == 1].copy()
    off = primary_pred[primary_pred["category"] == "off_target_fp"]
    off_summary = off.groupby(["SubjectID", "CaseID"], observed=True).agg(
        fp_slice_count=("ImageName", "nunique"), off_target_component_occurrences=("pred_component_id", "size"),
        off_target_area_px_total=("area_px", "sum"), off_target_area_px_largest=("area_px", "max"),
        off_target_area_mm2_total=("area_mm2", lambda values: values.sum(min_count=1)), off_target_area_mm2_largest=("area_mm2", "max"),
    ).reset_index()
    poor = primary_pred[primary_pred["category"] == "poorly_matched"].groupby(["SubjectID", "CaseID"], observed=True).size().rename("poorly_matched_occurrences").reset_index()
    excess = primary_pred.groupby(["SubjectID", "CaseID"], observed=True)["matched_excess_area_px"].sum().rename("matched_excess_area_px").reset_index()
    excess_mm2 = primary_pred.groupby(["SubjectID", "CaseID"], observed=True)["matched_excess_area_mm2"].agg(lambda values: values.sum(min_count=1)).rename("matched_excess_area_mm2").reset_index()
    gt_primary = gt_components[gt_components["minimum_area_px"] == 1]
    gt_summary = gt_primary.groupby(["SubjectID", "CaseID"], observed=True).agg(
        missed_target_occurrences=("is_missed", "sum"), fragmented_target_occurrences=("is_fragmented", "sum"),
    ).reset_index()
    report = report.merge(off_summary, on=["SubjectID", "CaseID"], how="left").merge(poor, on=["SubjectID", "CaseID"], how="left").merge(excess, on=["SubjectID", "CaseID"], how="left").merge(excess_mm2, on=["SubjectID", "CaseID"], how="left").merge(gt_summary, on=["SubjectID", "CaseID"], how="left")
    report = report.merge(case_predictions[["SubjectID", "CaseID", "class_id", "class_probability"]], on=["SubjectID", "CaseID"], how="left")
    report["classification_prediction"] = (report["class_probability"] >= 0.5).astype(int)
    report["classification_correct"] = (report["classification_prediction"] == report["class_id"]).astype(int)
    report["classification_confidence"] = np.maximum(report["class_probability"], 1 - report["class_probability"])
    count_columns = ["fp_slice_count", "off_target_component_occurrences", "off_target_area_px_total", "off_target_area_px_largest", "poorly_matched_occurrences", "matched_excess_area_px", "missed_target_occurrences", "fragmented_target_occurrences"]
    report[count_columns] = report[count_columns].fillna(0)
    return report


def evaluate_model(
    model: nn.Module,
    loaders: Mapping[str, Iterable[Mapping[str, Any]]],
    config: ResearchConfig,
    experiment: str,
    seed: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Tune all thresholds/calibration on validation, then access locked test once."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_dir = config.run_dir(experiment, seed) / "evaluation"
    predictions_dir, metrics_dir = evaluation_dir / "predictions", evaluation_dir / "metrics"
    components_dir, curves_dir = evaluation_dir / "component_analysis", evaluation_dir / "curves"
    figures_dir, reproducibility_dir = evaluation_dir / "figures", evaluation_dir / "reproducibility"
    for directory in (predictions_dir, metrics_dir, components_dir, curves_dir, figures_dir, reproducibility_dir): directory.mkdir(parents=True, exist_ok=True)

    validation_raw = collect_raw_predictions(model, loaders["validation"], device)
    segmentation_threshold, threshold_table = select_segmentation_threshold(validation_raw, config.section("selection")["segmentation_thresholds"])
    threshold_table.to_csv(curves_dir / "validation_segmentation_threshold_selection.csv", index=False)
    component_cfg = dict(config.section("components"))
    approved = component_cfg.get("radiologist_approved_thresholds")
    selected_components, component_threshold_table = select_component_thresholds(
        validation_raw, segmentation_threshold, component_cfg["validation_iou_grid"], component_cfg["validation_purity_grid"], int(component_cfg["connectivity"]),
    )
    component_threshold_table.to_csv(curves_dir / "validation_component_threshold_selection.csv", index=False)
    if approved:
        selected_components = {"poor_match_iou": float(approved["poor_match_iou"]), "poor_match_purity": float(approved["poor_match_purity"])}
    component_cfg.update(selected_components)

    calibration_method = str(config.section("statistics")["calibration_method"])
    validation_cases = case_aggregation(validation_raw, str(config.section("selection")["case_aggregation_primary"]), int(config.section("selection")["top_k"]))
    slice_calibrator = fit_probability_calibrator(validation_raw["class_id"], validation_raw["class_probability"], calibration_method)
    case_calibrator = fit_probability_calibrator(validation_cases["class_id"], validation_cases["class_probability"], calibration_method)
    joblib.dump(slice_calibrator, reproducibility_dir / "slice_calibrator.joblib")
    joblib.dump(case_calibrator, reproducibility_dir / "case_calibrator.joblib")
    locked = {
        "segmentation_threshold": segmentation_threshold,
        "classification_threshold": float(config.section("selection")["classification_threshold"]),
        **selected_components, "component_threshold_source": "radiologist_approved" if approved else "validation_locked_pending_radiologist_review",
        "calibration_method": calibration_method, "selected_on": "validation",
    }
    (reproducibility_dir / "locked_thresholds.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")

    register_final_test_access(config, experiment, seed)
    test_raw = collect_raw_predictions(model, loaders["test"], device)
    test_raw["calibrated_probability"] = apply_probability_calibrator(slice_calibrator, test_raw["class_probability"], calibration_method)
    validation_raw["calibrated_probability"] = apply_probability_calibrator(slice_calibrator, validation_raw["class_probability"], calibration_method)
    validation_manifest = store_prediction_arrays(validation_raw, evaluation_dir, "validation", config.dataset_root)
    test_manifest = store_prediction_arrays(test_raw, evaluation_dir, "test", config.dataset_root)
    scalar_columns = [column for column in test_raw.columns if column not in ("seg_probability_map", "target_mask")]
    test_raw[scalar_columns].to_csv(predictions_dir / "test_slice_classification_predictions.csv", index=False)
    validation_raw[[column for column in validation_raw.columns if column not in ("seg_probability_map", "target_mask")]].to_csv(predictions_dir / "validation_slice_classification_predictions.csv", index=False)

    slice_metrics, predicted_components, gt_components = analyze_and_export_segmentation(test_raw, segmentation_threshold, predictions_dir / "test", component_cfg)
    slice_metrics.to_csv(metrics_dir / "test_slice_segmentation_metrics.csv", index=False)
    predicted_components.to_csv(components_dir / "test_predicted_components.csv", index=False)
    gt_components.to_csv(components_dir / "test_ground_truth_components.csv", index=False)
    segmentation_groups = segmentation_group_summary(slice_metrics)
    segmentation_groups.to_csv(metrics_dir / "segmentation_metrics_by_class_and_roi_size.csv", index=False)

    case_keys = test_raw[["SubjectID", "CaseID"]].drop_duplicates().sort_values(["SubjectID", "CaseID"])
    component_index = pd.MultiIndex.from_tuples([(row.SubjectID, row.CaseID, int(area)) for row in case_keys.itertuples(index=False) for area in component_cfg["minimum_areas_px"]], names=["SubjectID", "CaseID", "minimum_area_px"])
    case_component_summary = predicted_components.groupby(["SubjectID", "CaseID", "minimum_area_px", "category"], observed=True).size().unstack(fill_value=0).reindex(component_index, fill_value=0) if not predicted_components.empty else pd.DataFrame(index=component_index)
    for category in ("matched", "poorly_matched", "off_target_fp", "empty_prediction"):
        if category not in case_component_summary: case_component_summary[category] = 0
    case_component_summary = case_component_summary.reset_index()
    case_component_summary.to_csv(components_dir / "test_case_component_counts.csv", index=False)
    case_gt_error_summary = gt_components.groupby(["SubjectID", "CaseID", "minimum_area_px"], observed=True).agg(gt_component_occurrences=("gt_component_id", "size"), missed_gt_occurrences=("is_missed", "sum"), fragmented_gt_occurrences=("is_fragmented", "sum")).reset_index()
    case_gt_error_summary.to_csv(components_dir / "test_case_gt_component_errors.csv", index=False)

    case_segmentation = slice_metrics.groupby(["SubjectID", "CaseID", "MaskedPatientName", "ClassName"], observed=True).agg(
        n_slices=("dice", "size"), mean_dice=("dice", "mean"), sd_dice=("dice", "std"),
        median_dice=("dice", "median"), q1_dice=("dice", lambda x: x.quantile(.25)), q3_dice=("dice", lambda x: x.quantile(.75)),
        mean_iou=("iou", "mean"), mean_pixel_precision=("pixel_precision", "mean"),
        mean_pixel_sensitivity=("pixel_sensitivity", "mean"), mean_pixel_specificity=("pixel_specificity", "mean"),
        mean_gt_coverage=("gt_coverage", "mean"), mean_prediction_purity=("prediction_purity", "mean"),
        mean_pred_gt_area_ratio=("pred_gt_area_ratio", "mean"), empty_prediction_rate=("empty_prediction", "mean"),
        missed_target_rate=("missed_target", "mean"), total_false_positive_area_px=("false_positive_area_px", "sum"),
    ).reset_index()
    case_segmentation.to_csv(metrics_dir / "test_case_segmentation_summary.csv", index=False)
    aggregation_outputs: dict[str, pd.DataFrame] = {}
    strategies = [config.section("selection")["case_aggregation_primary"], *config.section("selection")["case_aggregation_secondary"]]
    for strategy in strategies:
        case_rows = case_aggregation(test_raw, str(strategy), int(config.section("selection")["top_k"]))
        case_rows.to_csv(predictions_dir / f"test_case_predictions_{strategy}.csv", index=False)
        aggregation_outputs[str(strategy)] = case_rows
    primary = aggregation_outputs[str(config.section("selection")["case_aggregation_primary"])]
    primary["calibrated_probability"] = apply_probability_calibrator(case_calibrator, primary["class_probability"], calibration_method)
    primary.to_csv(predictions_dir / f"test_case_predictions_{config.section('selection')['case_aggregation_primary']}.csv", index=False)

    case_report = build_case_error_report(slice_metrics, predicted_components, gt_components, primary)
    case_report.to_csv(metrics_dir / "test_case_error_report.csv", index=False)
    component_ci = _component_bootstrap_table(case_report, config)
    component_ci.to_csv(metrics_dir / "component_error_metrics_with_95ci.csv", index=False)
    slice_ci = _classification_ci_table(test_raw, config); slice_ci.insert(0, "calibration", "raw"); slice_ci.insert(0, "level", "slice_subject_clustered")
    calibrated_slice = test_raw.copy(); calibrated_slice["class_probability"] = calibrated_slice["calibrated_probability"]
    slice_ci_cal = _classification_ci_table(calibrated_slice, config); slice_ci_cal.insert(0, "calibration", "calibrated"); slice_ci_cal.insert(0, "level", "slice_subject_clustered")
    case_ci = _classification_ci_table(primary, config); case_ci.insert(0, "calibration", "raw"); case_ci.insert(0, "level", "case_subject_clustered")
    calibrated_case = primary.copy(); calibrated_case["class_probability"] = calibrated_case["calibrated_probability"]
    case_ci_cal = _classification_ci_table(calibrated_case, config); case_ci_cal.insert(0, "calibration", "calibrated"); case_ci_cal.insert(0, "level", "case_subject_clustered")
    classification_ci = pd.concat([slice_ci, slice_ci_cal, case_ci, case_ci_cal], ignore_index=True)
    classification_ci.to_csv(metrics_dir / "classification_metrics_with_95ci.csv", index=False)
    segmentation_ci = _segmentation_ci_table(slice_metrics, config); segmentation_ci.to_csv(metrics_dir / "segmentation_metrics_with_95ci.csv", index=False)

    ece_bins = int(config.section("statistics")["ece_bins"])
    for level, raw_frame, calibrated_frame in (("case", primary, calibrated_case), ("slice", test_raw, calibrated_slice)):
        for calibration_name, frame in (("raw", raw_frame), ("calibrated", calibrated_frame)):
            roc, pr = curve_tables(frame["class_id"], frame["class_probability"])
            roc.to_csv(curves_dir / f"{level}_{calibration_name}_roc_curve.csv", index=False)
            pr.to_csv(curves_dir / f"{level}_{calibration_name}_precision_recall_curve.csv", index=False)
            calibration_bins(frame["class_id"].to_numpy(), frame["class_probability"].to_numpy(), ece_bins).to_csv(curves_dir / f"{level}_{calibration_name}_calibration_bins.csv", index=False)
            render_evaluation_figures(frame, figures_dir / calibration_name, float(config.section("selection")["classification_threshold"]), ece_bins, level=level)

    q1_table_dir = config.output_root / "q1_report" / "tables" / experiment / f"seed_{seed}"
    write_q1_workbook({"case_predictions": primary, "case_error_report": case_report, "slice_segmentation": slice_metrics, "case_segmentation": case_segmentation, "segmentation_subgroups": segmentation_groups, "classification_95ci": classification_ci, "segmentation_95ci": segmentation_ci, "component_95ci": component_ci, "pred_components": predicted_components, "gt_components": gt_components, "case_fp_counts": case_component_summary, "case_gt_errors": case_gt_error_summary}, q1_table_dir / "q1_results_tables.xlsx")
    return {"segmentation_threshold": segmentation_threshold, "slice_rows": test_raw, "slice_metrics": slice_metrics, "case_rows": primary, "case_error_report": case_report, "predicted_components": predicted_components, "validation_manifest": validation_manifest, "test_manifest": test_manifest, "output_dir": evaluation_dir}


def generate_gradcam_artifacts(model: nn.Module, loader: Iterable[Mapping[str, Any]], output_dir: Path, device: torch.device) -> pd.DataFrame:
    """Generate classification Grad-CAM for TALON and the multi-task U-Net baseline."""
    target_layer = getattr(model, "neck", None) or getattr(model, "bottleneck", None)
    if target_layer is None:
        raise AttributeError("No supported Grad-CAM target layer found.")
    records: list[dict[str, Any]] = []
    model.eval()
    with GradCAM(model, target_layer) as cam:
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            heatmaps, logits = cam(batch["image"])
            with torch.no_grad():
                segmentation_probability = torch.sigmoid(model(batch["image"])["lesion_logits"])
            for index in range(len(raw_batch["SubjectID"])):
                heatmap = heatmaps[index, 0].detach().cpu().numpy()
                mask = batch["mask"][index, 0].detach().cpu().numpy()
                prediction = (segmentation_probability[index, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
                image = batch["image"][index].detach().cpu().permute(1, 2, 0).numpy()
                identifiers = {"SubjectID": raw_batch["SubjectID"][index], "CaseID": raw_batch["CaseID"][index], "ImageName": raw_batch["ImageName"][index]}
                records.append({**identifiers, **xai_overlap_metrics(heatmap, mask, prediction), "predicted_class": int(logits[index].argmax())})
                slice_dir = output_dir / "gradcam" / identifiers["SubjectID"] / identifiers["CaseID"] / _safe_stem(identifiers["ImageName"])
                _save_mask(slice_dir / "gt_mask.png", mask)
                _save_mask(slice_dir / "pred_mask.png", prediction)
                _save_mask(slice_dir / "gradcam.png", heatmap)
                figure, axes = plt.subplots(1, 6, figsize=(18, 3.4))
                axes[0].imshow(np.clip(image, 0, 1)); axes[0].set_title("Original")
                axes[1].imshow(mask, cmap="gray"); axes[1].set_title("GT")
                axes[2].imshow(prediction, cmap="gray"); axes[2].set_title("Prediction")
                axes[3].imshow(heatmap, cmap="jet", vmin=0, vmax=1); axes[3].set_title("Grad-CAM")
                axes[4].imshow(np.clip(image, 0, 1)); axes[4].imshow(heatmap, cmap="jet", alpha=.45); axes[4].contour(mask, levels=[.5], colors=["lime"]); axes[4].set_title("Grad-CAM + GT")
                axes[5].imshow(np.clip(image, 0, 1)); axes[5].imshow(heatmap, cmap="jet", alpha=.45)
                if prediction.any(): axes[5].contour(prediction, levels=[.5], colors=["red"])
                axes[5].set_title("Grad-CAM + prediction")
                for axis in axes: axis.axis("off")
                figure.tight_layout(); slice_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(slice_dir / "xai_composite.png", dpi=220, bbox_inches="tight"); plt.close(figure)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "gradcam_metrics.csv", index=False)
    return frame


def evaluate_external_dataset(
    model: nn.Module,
    external_loader: Iterable[Mapping[str, Any]],
    locked_segmentation_threshold: float,
    locked_classification_threshold: float,
    config: ResearchConfig,
    output_dir: Path,
    device: torch.device,
    calibrator: Any | None = None,
    case_calibrator: Any | None = None,
    calibration_method: str | None = None,
) -> dict[str, Any]:
    """External-validation entry point; no threshold or parameter is retuned here."""
    raw = collect_raw_predictions(model, external_loader, device)
    if calibrator is not None and calibration_method is not None:
        raw["calibrated_probability"] = apply_probability_calibrator(calibrator, raw["class_probability"], calibration_method)
    slice_metrics, predicted_components, gt_components = analyze_and_export_segmentation(raw, locked_segmentation_threshold, output_dir, config.section("components"))
    cases = case_aggregation(raw, str(config.section("selection")["case_aggregation_primary"]), int(config.section("selection")["top_k"]))
    calibrated_cases = None
    if case_calibrator is not None and calibration_method is not None:
        calibrated_cases = cases.copy()
        calibrated_cases["class_probability"] = apply_probability_calibrator(case_calibrator, cases["class_probability"], calibration_method)
    return {
        "classification": classification_metrics(cases["class_id"], cases["class_probability"], locked_classification_threshold, int(config.section("statistics")["ece_bins"])),
        "classification_calibrated": None if calibrated_cases is None else classification_metrics(calibrated_cases["class_id"], calibrated_cases["class_probability"], locked_classification_threshold, int(config.section("statistics")["ece_bins"])),
        "case_rows": cases,
        "calibrated_case_rows": calibrated_cases,
        "raw_rows": raw,
        "slice_metrics": slice_metrics, "predicted_components": predicted_components, "ground_truth_components": gt_components,
    }
