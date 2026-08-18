"""Patient-aware performance estimates, confidence intervals and comparisons."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if selected.any():
            ece += selected.mean() * abs(float(y_true[selected].mean()) - float(probability[selected].mean()))
    return float(ece)


def calibration_bins(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    records = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        records.append({
            "bin": index + 1, "lower": lower, "upper": upper, "n": int(selected.sum()),
            "mean_predicted": float(probability[selected].mean()) if selected.any() else np.nan,
            "observed_fraction": float(y_true[selected].mean()) if selected.any() else np.nan,
        })
    return pd.DataFrame(records)


def classification_metrics(y_true: Iterable[int], probability: Iterable[float], threshold: float = 0.5, ece_bins: int = 10) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)), "threshold": float(threshold), "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted), "sensitivity": recall_score(y, predicted, zero_division=0),
        "specificity": tn / max(tn + fp, 1), "precision": precision_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0), "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p, ece_bins),
    }
    result["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    result["auprc"] = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    result.update(calibration_slope_intercept(y, p))
    return {key: float(value) if isinstance(value, (np.floating, float)) else value for key, value in result.items()}


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    true = target.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    pred_area, gt_area = int(pred.sum()), int(true.sum())
    return {
        "dice": dice, "iou": (tp + 1e-6) / (tp + fp + fn + 1e-6),
        "pixel_sensitivity": (tp + 1e-6) / (tp + fn + 1e-6),
        "pixel_specificity": (tn + 1e-6) / (tn + fp + 1e-6),
        "pixel_precision": (tp + 1e-6) / (tp + fp + 1e-6),
        "gt_coverage": (tp + 1e-6) / (gt_area + 1e-6),
        "prediction_purity": (tp + 1e-6) / (pred_area + 1e-6),
        "empty_prediction": float(pred_area == 0), "missed_target": float(tp == 0),
        "prediction_area_px": pred_area, "ground_truth_area_px": gt_area,
        "pred_gt_area_ratio": pred_area / max(gt_area, 1), "false_positive_area_px": fp,
    }


def calibration_slope_intercept(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probability), dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan}
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
    return {"calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}


def fit_probability_calibrator(y_true: Iterable[int], probability: Iterable[float], method: str = "platt"):
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probability), dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration requires both classes in validation data.")
    if method == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    else:
        raise KeyError(f"Unsupported calibration method: {method}")
    return model


def apply_probability_calibrator(model, probability: Iterable[float], method: str) -> np.ndarray:
    p = np.clip(np.asarray(list(probability), dtype=float), 1e-6, 1 - 1e-6)
    if method == "platt":
        return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    return np.asarray(model.predict(p), dtype=float)


def curve_tables(y_true: Iterable[int], probability: Iterable[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    fpr, tpr, roc_threshold = roc_curve(y, p)
    precision, recall, pr_threshold = precision_recall_curve(y, p)
    roc = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_threshold})
    pr = pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.r_[pr_threshold, np.nan]})
    return roc, pr


def select_segmentation_threshold(validation_rows: pd.DataFrame, thresholds: Iterable[float]) -> tuple[float, pd.DataFrame]:
    """Choose threshold on validation probabilities only; callers must keep test locked."""
    records = []
    for threshold in thresholds:
        scores = [segmentation_metrics(prob >= threshold, target)["dice"] for prob, target in zip(validation_rows["seg_probability_map"], validation_rows["target_mask"])]
        records.append({"threshold": float(threshold), "mean_dice": float(np.mean(scores))})
    table = pd.DataFrame(records)
    best = float(table.sort_values(["mean_dice", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    return best, table


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    subject_column: str = "SubjectID",
    iterations: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    subjects = frame[subject_column].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for bootstrap_id, subject in enumerate(sampled):
            part = frame.loc[frame[subject_column] == subject].copy()
            part["_bootstrap_subject"] = bootstrap_id
            pieces.append(part)
        value = float(statistic(pd.concat(pieces, ignore_index=True)))
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return {"estimate": float(statistic(frame)), "lower": np.nan, "upper": np.nan, "valid_iterations": 0}
    alpha = (1 - confidence_level) / 2
    return {
        "estimate": float(statistic(frame)), "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1 - alpha)), "valid_iterations": len(estimates),
    }


def paired_patient_bootstrap_difference(
    paired_frame: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    subject_column: str = "SubjectID",
    iterations: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired row-level effect with resampling at the underlying-subject level."""
    frame = paired_frame.reset_index(drop=True)
    subjects = frame[subject_column].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(iterations):
        sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
        sampled_rows = pd.concat([frame.loc[frame[subject_column] == subject] for subject in sampled_subjects], ignore_index=True)
        boot.append(float((sampled_rows[metric_a] - sampled_rows[metric_b]).mean()))
    boot = np.asarray(boot)
    alpha = (1 - confidence_level) / 2
    point = float((frame[metric_a] - frame[metric_b]).mean())
    return {"difference": point, "lower": float(np.quantile(boot, alpha)), "upper": float(np.quantile(boot, 1 - alpha)), "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))}


def paired_auc_bootstrap_difference(
    paired_case_frame: pd.DataFrame,
    probability_a: str,
    probability_b: str,
    label_column: str = "class_id",
    subject_column: str = "SubjectID",
    iterations: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Case-level AUROC difference with paired underlying-subject resampling."""
    rng = np.random.default_rng(seed)
    frame = paired_case_frame.reset_index(drop=True)
    subjects = frame[subject_column].drop_duplicates().to_numpy()
    differences: list[float] = []
    for _ in range(iterations):
        sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
        sample = pd.concat([frame.loc[frame[subject_column] == subject] for subject in sampled_subjects], ignore_index=True)
        if sample[label_column].nunique() < 2:
            continue
        differences.append(float(roc_auc_score(sample[label_column], sample[probability_a]) - roc_auc_score(sample[label_column], sample[probability_b])))
    point = float(roc_auc_score(frame[label_column], frame[probability_a]) - roc_auc_score(frame[label_column], frame[probability_b]))
    if not differences:
        return {"difference": point, "lower": np.nan, "upper": np.nan, "p_two_sided": np.nan, "valid_iterations": 0}
    values = np.asarray(differences)
    alpha = (1 - confidence_level) / 2
    return {
        "difference": point, "lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1 - alpha)),
        "p_two_sided": float(min(1.0, 2 * min((values <= 0).mean(), (values >= 0).mean()))), "valid_iterations": len(values),
    }


def paired_classification_metric_bootstrap(
    paired_frame: pd.DataFrame,
    probability_a: str,
    probability_b: str,
    metric: str,
    label_column: str = "class_id",
    subject_column: str = "SubjectID",
    iterations: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    functions = {
        "auroc": roc_auc_score,
        "auprc": average_precision_score,
        "brier": brier_score_loss,
    }
    if metric not in functions:
        raise KeyError(metric)
    function = functions[metric]
    frame = paired_frame.reset_index(drop=True)
    subjects = frame[subject_column].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        sample = pd.concat([frame.loc[frame[subject_column] == subject] for subject in sampled], ignore_index=True)
        if sample[label_column].nunique() < 2 and metric != "brier":
            continue
        values.append(float(function(sample[label_column], sample[probability_a]) - function(sample[label_column], sample[probability_b])))
    point_a = float(function(frame[label_column], frame[probability_a])); point_b = float(function(frame[label_column], frame[probability_b]))
    array = np.asarray(values)
    alpha = (1 - confidence_level) / 2
    return {"model_a": point_a, "model_b": point_b, "difference": point_a - point_b, "lower": float(np.quantile(array, alpha)), "upper": float(np.quantile(array, 1 - alpha)), "p_two_sided": float(min(1.0, 2 * min((array <= 0).mean(), (array >= 0).mean())))}


def mcnemar_exact(y_true: Iterable[int], prediction_a: Iterable[int], prediction_b: Iterable[int]) -> dict[str, float]:
    y = np.asarray(list(y_true))
    a_correct = np.asarray(list(prediction_a)) == y
    b_correct = np.asarray(list(prediction_b)) == y
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    p_value = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only, "p_value": float(p_value)}


def paired_wilcoxon(values_a: Iterable[float], values_b: Iterable[float]) -> dict[str, float]:
    a, b = np.asarray(list(values_a), dtype=float), np.asarray(list(values_b), dtype=float)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted
