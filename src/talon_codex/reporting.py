"""Publication-ready tables and figures generated from saved numeric results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .analysis.statistics import calibration_bins, curve_tables
from .config import experiment_run_directory


def render_evaluation_figures(classification_rows: pd.DataFrame, output_dir: Path, classification_threshold: float, ece_bins: int, level: str = "case") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = classification_rows["class_id"].to_numpy(dtype=int)
    probability = classification_rows["class_probability"].to_numpy(dtype=float)
    prediction = (probability >= classification_threshold).astype(int)
    roc, pr = curve_tables(y, probability)
    calibration = calibration_bins(y, probability, ece_bins).dropna()
    figures: list[tuple[str, plt.Figure]] = []
    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.plot(roc["false_positive_rate"], roc["true_positive_rate"], linewidth=2)
    axis.plot([0, 1], [0, 1], "--", color="0.6")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{level.title()}-level ROC curve", xlim=(0, 1), ylim=(0, 1))
    figures.append((f"{level}_roc_curve", fig))
    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.plot(pr["recall"], pr["precision"], linewidth=2)
    axis.set(xlabel="Recall", ylabel="Precision", title=f"{level.title()}-level precision–recall curve", xlim=(0, 1), ylim=(0, 1))
    figures.append((f"{level}_precision_recall_curve", fig))
    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.plot(calibration["mean_predicted"], calibration["observed_fraction"], marker="o")
    axis.plot([0, 1], [0, 1], "--", color="0.6")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction", title=f"{level.title()}-level calibration", xlim=(0, 1), ylim=(0, 1))
    figures.append((f"{level}_calibration_plot", fig))
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(xlabel="Predicted class", ylabel="True class", title=f"{level.title()}-level confusion matrix")
    figures.append((f"{level}_confusion_matrix", fig))
    for name, figure in figures:
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)


def render_training_history(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("total_loss", "dice", "classification_accuracy", "lesion_recall", "checkpoint_score"):
        if metric not in history:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for split, group in history.groupby("split"):
            axis.plot(group["epoch"], group[metric], label=split)
        axis.set(xlabel="Epoch", ylabel=metric.replace("_", " ").title(), title=f"Training history: {metric.replace('_', ' ')}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"history_{metric}.png", dpi=250, bbox_inches="tight")
        plt.close(figure)


def write_q1_workbook(tables: Mapping[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            safe_name = name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)


def render_component_summary(component_counts: pd.DataFrame, output_dir: Path) -> Path:
    """Rebuild the connected-component sensitivity figure from a stored CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [name for name in ("matched", "poorly_matched", "off_target_fp") if name in component_counts]
    summary = component_counts.groupby("minimum_area_px", observed=True)[categories].mean().reset_index()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(summary))
    width = 0.8 / max(len(categories), 1)
    for index, category in enumerate(categories):
        axis.bar(x + (index - (len(categories) - 1) / 2) * width, summary[category], width=width, label=category)
    axis.set_xticks(x, [f"≥{int(value)} px" for value in summary["minimum_area_px"]])
    axis.set(xlabel="Minimum component area", ylabel="Mean count per case", title="Connected-component sensitivity analysis")
    axis.legend()
    figure.tight_layout()
    destination = output_dir / "case_component_sensitivity.png"
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return destination


def render_prediction_overlays_from_store(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    segmentation_threshold: float,
    maximum_items: int | None = 24,
) -> pd.DataFrame:
    """Rebuild deterministic overlays from stored NPZ predictions without model inference."""
    manifest = pd.read_csv(manifest_path)
    selected = manifest.sort_values(["SubjectID", "CaseID", "ImageName"])
    if maximum_items is not None:
        selected = selected.head(int(maximum_items))
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    store_root = manifest_path.parent
    for row in selected.itertuples(index=False):
        array_path = store_root / str(row.array_path)
        with np.load(array_path) as stored:
            probability = stored["segmentation_probability"].astype(np.float32)
            target = stored["target_mask"].astype(bool)
        image_path = dataset_root / Path(str(row.source_image_reference))
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Source image required for overlay regeneration: {image_path}")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (probability.shape[1], probability.shape[0]), interpolation=cv2.INTER_AREA)
        prediction = probability >= float(segmentation_threshold)
        figure, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(image); axes[0].set_title("Source image")
        axes[1].imshow(target, cmap="gray"); axes[1].set_title("Ground truth")
        axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1); axes[2].set_title("Probability")
        axes[3].imshow(image)
        axes[3].contour(target, levels=[0.5], colors=["lime"], linewidths=1.2)
        axes[3].contour(prediction, levels=[0.5], colors=["red"], linewidths=1.0)
        axes[3].set_title("GT (green) / prediction (red)")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{row.SubjectID} | {row.CaseID} | {row.ImageName}")
        figure.tight_layout()
        destination = output_dir / f"{row.SubjectID}_{row.CaseID}_{Path(str(row.ImageName)).stem}_overlay.png"
        figure.savefig(destination, dpi=250, bbox_inches="tight")
        plt.close(figure)
        records.append({
            "SubjectID": row.SubjectID, "CaseID": row.CaseID, "ImageName": row.ImageName,
            "overlay_path": destination.as_posix(), "array_path": array_path.as_posix(),
        })
    result = pd.DataFrame(records)
    result.to_csv(output_dir / "regenerated_overlay_manifest.csv", index=False)
    return result


def regenerate_q1_outputs(
    output_root: Path,
    dataset_root: Path,
    experiment: str,
    seed: int,
    ece_bins: int,
    maximum_overlays: int | None = 24,
) -> dict[str, Any]:
    """Recreate Q1 figures/tables only from stored numeric test artifacts."""
    run_dir = experiment_run_directory(output_root, experiment, seed)
    evaluation_dir = run_dir / "evaluation"
    predictions_dir = evaluation_dir / "predictions"
    metrics_dir = evaluation_dir / "metrics"
    components_dir = evaluation_dir / "component_analysis"
    thresholds = pd.read_json(evaluation_dir / "reproducibility" / "locked_thresholds.json", typ="series")
    classification_threshold = float(thresholds["classification_threshold"])
    segmentation_threshold = float(thresholds["segmentation_threshold"])
    q1_root = output_root / "q1_report"
    figure_dir = q1_root / "figures" / experiment / f"seed_{seed}"
    table_dir = q1_root / "tables" / experiment / f"seed_{seed}"
    supplementary_dir = q1_root / "supplementary" / experiment / f"seed_{seed}"
    for directory in (figure_dir, table_dir, supplementary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    case_predictions = pd.read_csv(predictions_dir / "test_case_predictions_mean_all.csv")
    slice_predictions = pd.read_csv(predictions_dir / "test_slice_classification_predictions.csv")
    render_evaluation_figures(case_predictions, figure_dir, classification_threshold, ece_bins, level="case")
    render_evaluation_figures(slice_predictions, figure_dir, classification_threshold, ece_bins, level="slice")
    calibrated_case = None
    calibrated_slice = None
    if "calibrated_probability" in case_predictions:
        calibrated_case = case_predictions.copy()
        calibrated_case["raw_probability"] = calibrated_case["class_probability"]
        calibrated_case["class_probability"] = calibrated_case["calibrated_probability"]
        render_evaluation_figures(calibrated_case, figure_dir / "calibrated", classification_threshold, ece_bins, level="case_calibrated")
    if "calibrated_probability" in slice_predictions:
        calibrated_slice = slice_predictions.copy()
        calibrated_slice["raw_probability"] = calibrated_slice["class_probability"]
        calibrated_slice["class_probability"] = calibrated_slice["calibrated_probability"]
        render_evaluation_figures(calibrated_slice, figure_dir / "calibrated", classification_threshold, ece_bins, level="slice_calibrated")
    history = pd.read_csv(run_dir / "training_history" / "training_history.csv")
    render_training_history(history, figure_dir / "training")
    component_counts = pd.read_csv(components_dir / "test_case_component_counts.csv")
    render_component_summary(component_counts, figure_dir)
    overlays = render_prediction_overlays_from_store(
        predictions_dir / "test" / "prediction_manifest.csv", dataset_root,
        supplementary_dir / "prediction_overlays", segmentation_threshold, maximum_overlays,
    )
    tables = {
        "case_predictions": case_predictions,
        "slice_predictions": slice_predictions,
        "classification_95ci": pd.read_csv(metrics_dir / "classification_metrics_with_95ci.csv"),
        "segmentation_95ci": pd.read_csv(metrics_dir / "segmentation_metrics_with_95ci.csv"),
        "case_segmentation": pd.read_csv(metrics_dir / "test_case_segmentation_summary.csv"),
        "case_fp_counts": component_counts,
        # Evaluation exports the canonical subgroup/error table under this
        # filename.  The previous reporting-only name was never produced and
        # caused Q1 regeneration to fail after all locked-test artifacts had
        # already been written.
        "roi_error": pd.read_csv(metrics_dir / "segmentation_metrics_by_class_and_roi_size.csv"),
        "overlay_manifest": overlays,
    }
    if calibrated_case is not None:
        tables["case_predictions_calibrated"] = calibrated_case
    if calibrated_slice is not None:
        tables["slice_predictions_calibrated"] = calibrated_slice
    optional_tables = {
        "case_error_report": metrics_dir / "test_case_error_report.csv",
        "segmentation_subgroups": metrics_dir / "segmentation_metrics_by_class_and_roi_size.csv",
        "component_95ci": metrics_dir / "component_error_metrics_with_95ci.csv",
        "xai_metrics": evaluation_dir / "xai" / "gradcam_metrics.csv",
        "teacher_checkpoints": run_dir / "teacher_checkpoint_comparison" / "teacher_checkpoint_validation_comparison.csv",
    }
    for table_name, table_path in optional_tables.items():
        if table_path.exists():
            tables[table_name] = pd.read_csv(table_path)
    workbook = table_dir / "q1_results_tables_regenerated.xlsx"
    write_q1_workbook(tables, workbook)
    narratives = write_bilingual_result_summaries(tables, q1_root / "reports" / experiment / f"seed_{seed}")
    return {"figure_dir": figure_dir, "table_dir": table_dir, "supplementary_dir": supplementary_dir, "workbook": workbook, **narratives}


def write_bilingual_result_summaries(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    """Create traceable Turkish-thesis and English-manuscript result drafts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classification = tables["classification_95ci"]
    segmentation = tables["segmentation_95ci"]
    case_raw = classification[(classification["level"] == "case_subject_clustered") & (classification["calibration"] == "raw")]
    def value(table: pd.DataFrame, metric: str, column: str = "estimate") -> float:
        rows = table[table["metric"] == metric]
        return float(rows.iloc[0][column]) if len(rows) else np.nan
    auroc, auprc = value(case_raw, "auroc"), value(case_raw, "auprc")
    dice, iou = value(segmentation, "dice"), value(segmentation, "iou")
    tr = output_dir / "tez_sonuclar_ozeti_tr.md"
    en = output_dir / "manuscript_results_summary_en.md"
    tr.write_text(f"""# Tez sonuç özeti — otomatik taslak

Hasta/vaka düzeyinde AUROC {auroc:.3f}, AUPRC {auprc:.3f} olarak hesaplandı. Dilim düzeyinde ortalama Dice {dice:.3f} ve IoU {iou:.3f} bulundu. Güven aralıkları ve alt grup sonuçları ilgili Q1 çalışma kitabından aktarılmalıdır.

> Bu metin yalnız sayısal çıktılardan üretilmiştir. Klinik tanımlamalar, etik bilgiler ve 107→103 hasta akışı doğrulanmadan nihai tez metni olarak kullanılmamalıdır.
""", encoding="utf-8")
    en.write_text(f"""# Manuscript results summary — automated draft

At the case level, the model achieved an AUROC of {auroc:.3f} and an AUPRC of {auprc:.3f}. At the slice level, the mean Dice score was {dice:.3f} and the mean IoU was {iou:.3f}. Confidence intervals and prespecified subgroup findings should be transferred from the Q1 results workbook.

> This draft is generated only from numeric artifacts. It must not be finalized before clinical definitions, ethics information, and the 107-to-103 participant flow are verified.
""", encoding="utf-8")
    return {"turkish_results_summary": tr, "english_results_summary": en}
