"""Paired statistical comparison of TALON-Net, baseline and ablation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import experiment_run_directory

from .statistics import holm_adjust, mcnemar_exact, paired_classification_metric_bootstrap, paired_patient_bootstrap_difference, paired_wilcoxon


def compare_two_runs(run_a: Path, run_b: Path, name_a: str, name_b: str, iterations: int, seed: int) -> pd.DataFrame:
    eval_a, eval_b = run_a / "evaluation", run_b / "evaluation"
    cases_a = pd.read_csv(eval_a / "predictions" / "test_case_predictions_mean_all.csv")
    cases_b = pd.read_csv(eval_b / "predictions" / "test_case_predictions_mean_all.csv")
    paired_cases = cases_a[["SubjectID", "CaseID", "class_id", "class_probability"]].merge(
        cases_b[["SubjectID", "CaseID", "class_id", "class_probability"]], on=["SubjectID", "CaseID", "class_id"], suffixes=("_a", "_b"), validate="one_to_one"
    )
    threshold = 0.5
    paired_cases["correct_a"] = ((paired_cases["class_probability_a"] >= threshold).astype(int) == paired_cases["class_id"]).astype(float)
    paired_cases["correct_b"] = ((paired_cases["class_probability_b"] >= threshold).astype(int) == paired_cases["class_id"]).astype(float)
    accuracy = paired_patient_bootstrap_difference(paired_cases, "correct_a", "correct_b", iterations=iterations, seed=seed)
    mcnemar = mcnemar_exact(paired_cases["class_id"], paired_cases["class_probability_a"] >= threshold, paired_cases["class_probability_b"] >= threshold)
    auc = paired_classification_metric_bootstrap(paired_cases, "class_probability_a", "class_probability_b", "auroc", iterations=iterations, seed=seed)
    auprc = paired_classification_metric_bootstrap(paired_cases, "class_probability_a", "class_probability_b", "auprc", iterations=iterations, seed=seed)
    brier = paired_classification_metric_bootstrap(paired_cases, "class_probability_a", "class_probability_b", "brier", iterations=iterations, seed=seed)
    seg_a = pd.read_csv(eval_a / "metrics" / "test_slice_segmentation_metrics.csv")
    seg_b = pd.read_csv(eval_b / "metrics" / "test_slice_segmentation_metrics.csv")
    keys = ["SubjectID", "CaseID", "ImageName"]
    paired_seg = seg_a[keys + ["dice", "iou"]].merge(seg_b[keys + ["dice", "iou"]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    dice_w = paired_wilcoxon(paired_seg["dice_a"], paired_seg["dice_b"])
    dice_ci = paired_patient_bootstrap_difference(paired_seg, "dice_a", "dice_b", iterations=iterations, seed=seed)
    iou_w = paired_wilcoxon(paired_seg["iou_a"], paired_seg["iou_b"])
    iou_ci = paired_patient_bootstrap_difference(paired_seg, "iou_a", "iou_b", iterations=iterations, seed=seed)
    case_error_a = pd.read_csv(eval_a / "metrics" / "test_case_error_report.csv")
    case_error_b = pd.read_csv(eval_b / "metrics" / "test_case_error_report.csv")
    paired_error = case_error_a[["SubjectID", "CaseID", "off_target_component_occurrences"]].merge(case_error_b[["SubjectID", "CaseID", "off_target_component_occurrences"]], on=["SubjectID", "CaseID"], suffixes=("_a", "_b"), validate="one_to_one")
    off_target = paired_patient_bootstrap_difference(paired_error, "off_target_component_occurrences_a", "off_target_component_occurrences_b", iterations=iterations, seed=seed)
    off_target_w = paired_wilcoxon(paired_error["off_target_component_occurrences_a"], paired_error["off_target_component_occurrences_b"])
    xai_a = pd.read_csv(eval_a / "xai" / "gradcam_metrics.csv")
    xai_b = pd.read_csv(eval_b / "xai" / "gradcam_metrics.csv")
    xai = xai_a[["SubjectID", "CaseID", "ImageName", "xai_energy_inside_gt"]].merge(xai_b[["SubjectID", "CaseID", "ImageName", "xai_energy_inside_gt"]], on=["SubjectID", "CaseID", "ImageName"], suffixes=("_a", "_b"), validate="one_to_one")
    xai_ci = paired_patient_bootstrap_difference(xai, "xai_energy_inside_gt_a", "xai_energy_inside_gt_b", iterations=iterations, seed=seed)
    def row(endpoint, result, model_a, model_b, test):
        return {"comparison": f"{name_a} - {name_b}", "endpoint": endpoint, "model_a_result": model_a, "model_b_result": model_b, "effect": result["difference"], "ci_lower": result["lower"], "ci_upper": result["upper"], "p_value": result["p_two_sided"], "test": test}
    rows = [
        row("case_AUROC", auc, auc["model_a"], auc["model_b"], "paired subject-clustered bootstrap"),
        row("case_AUPRC", auprc, auprc["model_a"], auprc["model_b"], "paired subject-clustered bootstrap"),
        row("case_Brier", brier, brier["model_a"], brier["model_b"], "paired subject-clustered bootstrap"),
        row("case_accuracy", accuracy, paired_cases["correct_a"].mean(), paired_cases["correct_b"].mean(), f"paired bootstrap; exact McNemar p={mcnemar['p_value']:.6g}"),
        row("slice_Dice", dice_ci, paired_seg["dice_a"].mean(), paired_seg["dice_b"].mean(), f"paired bootstrap; Wilcoxon p={dice_w['p_value']:.6g}"),
        row("slice_IoU", iou_ci, paired_seg["iou_a"].mean(), paired_seg["iou_b"].mean(), f"paired bootstrap; Wilcoxon p={iou_w['p_value']:.6g}"),
        row("case_off_target_components", off_target, paired_error["off_target_component_occurrences_a"].mean(), paired_error["off_target_component_occurrences_b"].mean(), f"paired bootstrap; Wilcoxon p={off_target_w['p_value']:.6g}"),
        row("slice_GradCAM_energy_inside_GT", xai_ci, xai["xai_energy_inside_gt_a"].mean(), xai["xai_energy_inside_gt_b"].mean(), "paired subject-clustered bootstrap"),
    ]
    return pd.DataFrame(rows)


def compare_suite(output_root: Path, experiments: list[str], seeds: list[int], iterations: int) -> pd.DataFrame:
    records = []
    reference = experiments[0]
    for seed in seeds:
        for challenger in experiments[1:]:
            table = compare_two_runs(
                experiment_run_directory(output_root, reference, seed),
                experiment_run_directory(output_root, challenger, seed),
                reference, challenger, iterations, seed,
            )
            table.insert(0, "seed", seed)
            records.append(table)
    result = pd.concat(records, ignore_index=True)
    adjusted = holm_adjust({str(index): value for index, value in result["p_value"].items() if np.isfinite(value)})
    result["p_holm"] = [adjusted.get(str(index), np.nan) for index in result.index]
    return result
