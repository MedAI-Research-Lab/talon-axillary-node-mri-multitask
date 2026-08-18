"""Controlled execution entry point for the TALON-Net Jupyter deliverable.

This runner never writes to the legacy ``talon_outputs`` tree.  It exists so
long-running notebook stages can be logged and restarted from a terminal while
the notebooks remain the human-readable record of the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch

from talon_codex.config import load_config
from talon_codex.data import build_data_bundle
from talon_codex.evaluation import (
    _classification_ci_table,
    _segmentation_ci_table,
    evaluate_model,
    generate_gradcam_artifacts,
    write_artifact_manifest,
)
from talon_codex.analysis.statistics import calibration_bins, curve_tables
from talon_codex.losses import MultiTaskCriterion
from talon_codex.models import (
    build_capacity_matched_baseline,
    build_talon,
    build_training_spatial_prior,
)
from talon_codex.pipeline import prepare_data
from talon_codex.reporting import (
    regenerate_q1_outputs,
    render_evaluation_figures,
    render_training_history,
    write_q1_workbook,
)
from talon_codex.training import audit_auxiliary_gradients, train_experiment, training_weights


CONFIG_PATH = PROJECT_ROOT / "configs" / "publication_config.local.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PROJECT_ROOT / "configs" / "publication_config.frozen.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXECUTION_ROOT = OUTPUT_ROOT / "execution"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def assert_output_contract(config) -> None:
    resolved = config.output_root.resolve()
    required = (PROJECT_ROOT / "outputs").resolve()
    forbidden = (WORKSPACE_ROOT / "talon_outputs").resolve()
    if resolved != required:
        raise RuntimeError(f"Output root contract violation: {resolved} != {required}")
    if resolved == forbidden or forbidden in resolved.parents:
        raise RuntimeError("Legacy talon_outputs is write-protected for this execution.")


def environment_manifest() -> dict[str, object]:
    package_names = [
        "torch", "torchvision", "numpy", "pandas", "openpyxl", "Pillow",
        "opencv-python", "albumentations", "scikit-learn", "scipy",
        "matplotlib", "seaborn", "joblib", "python-docx",
    ]
    versions: dict[str, str | None] = {}
    try:
        from importlib.metadata import version
        for name in package_names:
            try:
                versions[name] = version(name)
            except Exception:
                versions[name] = None
    except Exception:
        pass
    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        gpu["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        gpu["nvidia_smi_error"] = repr(exc)
    return {
        "created_utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "packages": versions,
        "gpu": gpu,
        "pid": os.getpid(),
        "project_root": str(PROJECT_ROOT),
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
    }


def stage_environment(config) -> dict[str, object]:
    manifest = environment_manifest()
    write_json(EXECUTION_ROOT / "environment_manifest.json", manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the planned full training run.")
    return manifest


def stage_audit(config) -> dict[str, object]:
    started = time.time()
    prepared = prepare_data(config, seed=config.seed, hash_audit=True)
    audits = prepared["audits"]
    frames = prepared["bundle"].frames
    summary = {
        "status": "passed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "slices": int(sum(len(frame) for frame in frames.values())),
        "subjects": int(pd.concat(frames.values(), ignore_index=True)["SubjectID"].nunique()),
        "cases": int(pd.concat(frames.values(), ignore_index=True)["CaseID"].nunique()),
        "split_slices": {name: int(len(frame)) for name, frame in frames.items()},
        "empty_masks": int(audits["mask_audit"]["IsEmptyAfterThreshold"].sum()),
        "shape_mismatches": int((~audits["mask_audit"]["ImageMaskShapeMatch"]).sum()),
        "duplicate_image_groups": int(audits["duplicate_images"]["ImagePixelSHA256"].nunique()) if not audits["duplicate_images"].empty else 0,
        "cross_split_duplicate_rows": int((audits["duplicate_images"]["DuplicateRisk"] == "cross_split_leakage").sum()) if not audits["duplicate_images"].empty else 0,
        "cleaned_masks": int(len(prepared["cleaned_mask_manifest"])),
        "source_files": {
            "mask_audit": str(OUTPUT_ROOT / "audit" / "mask_checks" / "mask_audit.csv"),
            "duplicate_images": str(OUTPUT_ROOT / "audit" / "duplicate_checks" / "duplicate_images.csv"),
            "split_assignments": str(OUTPUT_ROOT / "audit" / "split_assignments" / f"split_assignments_seed_{config.seed}.csv"),
            "cleaned_mask_manifest": str(OUTPUT_ROOT / "cleaned_masks" / "cleaned_mask_manifest.csv"),
        },
    }
    write_json(EXECUTION_ROOT / "data_audit_status.json", summary)
    return summary


def stage_model_audit(config) -> dict[str, object]:
    started = time.time()
    bundle = build_data_bundle(config, seed=config.seed)
    prior = build_training_spatial_prior(
        [Path(path) for path in bundle.frames["train"]["ResolvedRoiPath"]],
        int(config.get("mask_threshold", 127)),
    )
    talon = build_talon(config.section("model"), prior, "TALON_FULL")
    baseline, capacity_rows = build_capacity_matched_baseline(config.section("model"), talon)
    talon_count = sum(p.numel() for p in talon.parameters() if p.requires_grad)
    baseline_count = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    capacity = pd.DataFrame(capacity_rows)
    capacity_path = EXECUTION_ROOT / "model_capacity_audit.csv"
    capacity.to_csv(capacity_path, index=False)

    device = torch.device("cuda")
    talon.to(device)
    class_weights, positive_weight = training_weights(bundle.frames["train"], config)
    criterion = MultiTaskCriterion(config.section("loss"), class_weights, positive_weight, set()).to(device)
    phase = str(config.section("training")["teacher_phases"][0]["name"])
    gradient = audit_auxiliary_gradients(talon, criterion, bundle.loaders["train"], device, phase, set())
    gradient_path = EXECUTION_ROOT / "preflight_auxiliary_gradient_audit.csv"
    gradient.to_csv(gradient_path, index=False)
    if not bool(gradient["passes"].all()):
        raise RuntimeError(f"Auxiliary gradient audit failed: {gradient_path}")
    summary = {
        "status": "passed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0),
        "talon_trainable_parameters": talon_count,
        "baseline_trainable_parameters": baseline_count,
        "absolute_parameter_difference": abs(talon_count - baseline_count),
        "relative_parameter_difference": abs(talon_count - baseline_count) / max(talon_count, 1),
        "capacity_audit": str(capacity_path),
        "gradient_audit": str(gradient_path),
    }
    write_json(EXECUTION_ROOT / "model_preflight_status.json", summary)
    return summary


def stage_amp_smoke(config) -> dict[str, object]:
    """Verify that a real configured batch has finite outputs/losses under AMP."""
    from talon_codex.training import _move_batch
    bundle = build_data_bundle(config, seed=config.seed)
    prior = build_training_spatial_prior(
        [Path(path) for path in bundle.frames["train"]["ResolvedRoiPath"]],
        int(config.get("mask_threshold", 127)),
    )
    model = build_talon(config.section("model"), prior, "TALON_FULL").to("cuda").train()
    class_weights, positive_weight = training_weights(bundle.frames["train"], config)
    criterion = MultiTaskCriterion(config.section("loss"), class_weights, positive_weight, set()).to("cuda")
    batch = _move_batch(next(iter(bundle.loaders["train"])), torch.device("cuda"))
    phase = str(config.section("training")["teacher_phases"][0]["name"])
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(batch["image"])
        losses = criterion(outputs, batch, phase, talon=True)
    output_finite = {key: bool(torch.isfinite(value).all()) for key, value in outputs.items() if torch.is_tensor(value)}
    loss_values = {key: float(value.detach()) for key, value in losses.items()}
    loss_finite = {key: bool(torch.isfinite(value).all()) for key, value in losses.items()}
    result = {
        "status": "passed" if all(output_finite.values()) and all(loss_finite.values()) else "failed",
        "created_utc": utc_now(),
        "batch_size": int(batch["image"].shape[0]),
        "output_finite": output_finite,
        "loss_finite": loss_finite,
        "loss_values": loss_values,
    }
    write_json(EXECUTION_ROOT / "amp_batch_smoke_status.json", result)
    if result["status"] != "passed":
        raise RuntimeError("AMP batch smoke test produced a non-finite tensor.")
    return result


def stage_train(config, experiment: str, seed: int) -> dict[str, object]:
    started = time.time()
    bundle = build_data_bundle(config, seed=seed)
    model, history, checkpoint = train_experiment(config, bundle, experiment, seed)
    run_dir = config.run_dir(experiment, seed)
    render_training_history(history, run_dir / "figures" / "training")
    summary = {
        "status": "completed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "experiment": experiment,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "history": str(run_dir / "training_history" / "training_history.csv"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    write_json(run_dir / "execution_training_status.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def _load_trained_model(config, experiment: str, seed: int, bundle):
    from talon_codex.models import MaskGuidedMultiTaskUNet, baseline_width_from_checkpoint
    run_dir = config.run_dir(experiment, seed)
    checkpoint_path = run_dir / "checkpoints" / "best_overall.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prior = build_training_spatial_prior(
        [Path(path) for path in bundle.frames["train"]["ResolvedRoiPath"]],
        int(config.get("mask_threshold", 127)),
    )
    if experiment == "UNET_MTL_MASK_GUIDED":
        model = MaskGuidedMultiTaskUNet(
            baseline_width_from_checkpoint(checkpoint, config.section("model")),
            float(config.section("model")["classifier_dropout"]),
        )
    else:
        model = build_talon(config.section("model"), prior, experiment)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(torch.device("cuda")), checkpoint_path


def stage_evaluate(config, experiment: str, seed: int) -> dict[str, object]:
    started = time.time()
    bundle = build_data_bundle(config, seed=seed)
    model, checkpoint = _load_trained_model(config, experiment, seed, bundle)
    device = torch.device("cuda")
    evaluation = evaluate_model(model, bundle.loaders, config, experiment, seed, device)
    evaluation_dir = Path(evaluation["output_dir"])
    xai = generate_gradcam_artifacts(model, bundle.loaders["test"], evaluation_dir / "xai", device)
    manifest = write_artifact_manifest(evaluation_dir, config, experiment, seed)
    regenerated = regenerate_q1_outputs(
        config.output_root,
        config.dataset_root,
        experiment,
        seed,
        int(config.section("statistics")["ece_bins"]),
        24,
    )
    summary = {
        "status": "completed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "experiment": experiment,
        "seed": seed,
        "split": "locked_test",
        "checkpoint": str(checkpoint),
        "evaluation_dir": str(evaluation_dir),
        "artifact_manifest": str(evaluation_dir / "reproducibility" / "artifact_manifest.csv"),
        "xai_rows": int(len(xai)),
        "artifact_rows": int(len(manifest)),
        "q1_outputs": {key: str(value) for key, value in regenerated.items()},
        "raw_vs_calibrated": "both",
        "ci_method": "SubjectID-clustered bootstrap, configured 5000 iterations",
    }
    write_json(config.run_dir(experiment, seed) / "execution_evaluation_status.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_resume_evaluate(config, experiment: str, seed: int) -> dict[str, object]:
    """Resume an interrupted evaluation without repeating locked-test metrics.

    The locked test predictions and component tables must already exist from the
    original registered access.  Bootstrap tables, curves, figures and the Q1
    workbook are rebuilt exclusively from those frozen CSV files.  The model is
    loaded only to finish a missing Grad-CAM export; no classification or
    segmentation metric inference is repeated.
    """
    started = time.time()
    run_dir = config.run_dir(experiment, seed)
    evaluation_dir = run_dir / "evaluation"
    predictions_dir = evaluation_dir / "predictions"
    metrics_dir = evaluation_dir / "metrics"
    components_dir = evaluation_dir / "component_analysis"
    curves_dir = evaluation_dir / "curves"
    figures_dir = evaluation_dir / "figures"
    registry = config.output_root / "test_access_registry" / experiment / f"seed_{seed}.json"
    checkpoint = run_dir / "checkpoints" / "best_overall.pt"
    status_path = run_dir / "execution_evaluation_status.json"
    resume_audit_path = run_dir / "evaluation_resume_audit.json"

    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8-sig"))
        if str(existing.get("status", "")).lower() == "completed":
            return existing
    if not registry.exists():
        raise FileNotFoundError("Cannot resume without the original locked-test access registry: " + str(registry))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    required = {
        "locked_thresholds": evaluation_dir / "reproducibility" / "locked_thresholds.json",
        "test_slice_predictions": predictions_dir / "test_slice_classification_predictions.csv",
        "validation_slice_predictions": predictions_dir / "validation_slice_classification_predictions.csv",
        "test_case_predictions": predictions_dir / "test_case_predictions_mean_all.csv",
        "test_prediction_manifest": predictions_dir / "test" / "prediction_manifest.csv",
        "validation_prediction_manifest": predictions_dir / "validation" / "prediction_manifest.csv",
        "slice_segmentation": metrics_dir / "test_slice_segmentation_metrics.csv",
        "case_segmentation": metrics_dir / "test_case_segmentation_summary.csv",
        "segmentation_subgroups": metrics_dir / "segmentation_metrics_by_class_and_roi_size.csv",
        "case_error_report": metrics_dir / "test_case_error_report.csv",
        "component_ci": metrics_dir / "component_error_metrics_with_95ci.csv",
        "pred_components": components_dir / "test_predicted_components.csv",
        "gt_components": components_dir / "test_ground_truth_components.csv",
        "case_fp_counts": components_dir / "test_case_component_counts.csv",
        "case_gt_errors": components_dir / "test_case_gt_component_errors.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Interrupted evaluation lacks frozen artifacts required for safe resume: " + "; ".join(missing))

    thresholds = json.loads(required["locked_thresholds"].read_text(encoding="utf-8"))
    if thresholds.get("selected_on") != "validation":
        raise RuntimeError("Stored thresholds are not explicitly validation-selected.")
    registry_payload = json.loads(registry.read_text(encoding="utf-8-sig"))
    if registry_payload.get("experiment") != experiment or int(registry_payload.get("seed", -1)) != int(seed):
        raise RuntimeError("Locked-test registry identity does not match the interrupted run.")

    test_rows = pd.read_csv(required["test_slice_predictions"])
    validation_rows = pd.read_csv(required["validation_slice_predictions"])
    primary = pd.read_csv(required["test_case_predictions"])
    slice_metrics = pd.read_csv(required["slice_segmentation"])
    test_manifest = pd.read_csv(required["test_prediction_manifest"])
    validation_manifest = pd.read_csv(required["validation_prediction_manifest"])
    if not (len(test_rows) == len(slice_metrics) == len(test_manifest)):
        raise RuntimeError("Frozen test slice rows, segmentation rows and prediction manifest have different lengths.")
    if len(validation_rows) != len(validation_manifest):
        raise RuntimeError("Frozen validation rows and prediction manifest have different lengths.")
    key = ["SubjectID", "CaseID", "ImageName"]
    if not test_rows[key].astype(str).equals(slice_metrics[key].astype(str)):
        raise RuntimeError("Frozen test classification and segmentation row identities do not match.")
    if "calibrated_probability" not in test_rows or "calibrated_probability" not in primary:
        raise RuntimeError("Frozen calibrated probabilities are absent; calibration will not be refit on resume.")

    classification_ci_path = metrics_dir / "classification_metrics_with_95ci.csv"
    segmentation_ci_path = metrics_dir / "segmentation_metrics_with_95ci.csv"
    if classification_ci_path.exists():
        classification_ci = pd.read_csv(classification_ci_path)
    else:
        slice_ci = _classification_ci_table(test_rows, config)
        slice_ci.insert(0, "calibration", "raw"); slice_ci.insert(0, "level", "slice_subject_clustered")
        calibrated_slice = test_rows.copy(); calibrated_slice["class_probability"] = calibrated_slice["calibrated_probability"]
        slice_ci_cal = _classification_ci_table(calibrated_slice, config)
        slice_ci_cal.insert(0, "calibration", "calibrated"); slice_ci_cal.insert(0, "level", "slice_subject_clustered")
        case_ci = _classification_ci_table(primary, config)
        case_ci.insert(0, "calibration", "raw"); case_ci.insert(0, "level", "case_subject_clustered")
        calibrated_case = primary.copy(); calibrated_case["class_probability"] = calibrated_case["calibrated_probability"]
        case_ci_cal = _classification_ci_table(calibrated_case, config)
        case_ci_cal.insert(0, "calibration", "calibrated"); case_ci_cal.insert(0, "level", "case_subject_clustered")
        classification_ci = pd.concat([slice_ci, slice_ci_cal, case_ci, case_ci_cal], ignore_index=True)
        classification_ci.to_csv(classification_ci_path, index=False)
    if segmentation_ci_path.exists():
        segmentation_ci = pd.read_csv(segmentation_ci_path)
    else:
        segmentation_ci = _segmentation_ci_table(slice_metrics, config)
        segmentation_ci.to_csv(segmentation_ci_path, index=False)

    calibrated_slice = test_rows.copy(); calibrated_slice["class_probability"] = calibrated_slice["calibrated_probability"]
    calibrated_case = primary.copy(); calibrated_case["class_probability"] = calibrated_case["calibrated_probability"]
    ece_bins = int(config.section("statistics")["ece_bins"])
    class_threshold = float(thresholds["classification_threshold"])
    curves_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for level, raw_frame, calibrated_frame in (("case", primary, calibrated_case), ("slice", test_rows, calibrated_slice)):
        for calibration_name, frame in (("raw", raw_frame), ("calibrated", calibrated_frame)):
            roc, pr = curve_tables(frame["class_id"], frame["class_probability"])
            roc.to_csv(curves_dir / f"{level}_{calibration_name}_roc_curve.csv", index=False)
            pr.to_csv(curves_dir / f"{level}_{calibration_name}_precision_recall_curve.csv", index=False)
            calibration_bins(frame["class_id"].to_numpy(), frame["class_probability"].to_numpy(), ece_bins).to_csv(
                curves_dir / f"{level}_{calibration_name}_calibration_bins.csv", index=False
            )
            render_evaluation_figures(frame, figures_dir / calibration_name, class_threshold, ece_bins, level=level)

    tables = {
        "case_predictions": primary,
        "case_error_report": pd.read_csv(required["case_error_report"]),
        "slice_segmentation": slice_metrics,
        "case_segmentation": pd.read_csv(required["case_segmentation"]),
        "segmentation_subgroups": pd.read_csv(required["segmentation_subgroups"]),
        "classification_95ci": classification_ci,
        "segmentation_95ci": segmentation_ci,
        "component_95ci": pd.read_csv(required["component_ci"]),
        "pred_components": pd.read_csv(required["pred_components"]),
        "gt_components": pd.read_csv(required["gt_components"]),
        "case_fp_counts": pd.read_csv(required["case_fp_counts"]),
        "case_gt_errors": pd.read_csv(required["case_gt_errors"]),
    }
    q1_table_dir = config.output_root / "q1_report" / "tables" / experiment / f"seed_{seed}"
    write_q1_workbook(tables, q1_table_dir / "q1_results_tables.xlsx")

    bundle = build_data_bundle(config, seed=seed)
    if len(bundle.frames["test"]) != len(test_rows):
        raise RuntimeError("Current test split no longer matches the frozen test prediction row count.")
    model, loaded_checkpoint = _load_trained_model(config, experiment, seed, bundle)
    if loaded_checkpoint.resolve() != checkpoint.resolve():
        raise RuntimeError("Resume loaded a different checkpoint path.")
    device = torch.device("cuda")
    xai_path = evaluation_dir / "xai" / "gradcam_metrics.csv"
    if xai_path.exists() and len(pd.read_csv(xai_path)) == len(test_rows):
        xai = pd.read_csv(xai_path)
        xai_inference_performed = False
    else:
        xai = generate_gradcam_artifacts(model, bundle.loaders["test"], evaluation_dir / "xai", device)
        xai_inference_performed = True
    del model
    torch.cuda.empty_cache()
    if len(xai) != len(test_rows):
        raise RuntimeError(f"Grad-CAM row count {len(xai)} does not match frozen test rows {len(test_rows)}.")

    audit = {
        "status": "completed",
        "created_utc": utc_now(),
        "experiment": experiment,
        "seed": int(seed),
        "recovery_type": "interrupted_evaluation_resume_from_frozen_predictions",
        "original_test_registry_preserved": True,
        "locked_test_metric_inference_repeated": False,
        "threshold_selection_repeated": False,
        "calibration_refit_repeated": False,
        "xai_inference_performed": xai_inference_performed,
        "checkpoint_sha256": _sha256(checkpoint),
        "frozen_test_rows": int(len(test_rows)),
        "frozen_validation_rows": int(len(validation_rows)),
        "stored_thresholds": thresholds,
        "original_test_access_record": str(registry),
    }
    write_json(resume_audit_path, audit)
    manifest = write_artifact_manifest(evaluation_dir, config, experiment, seed)
    regenerated = regenerate_q1_outputs(config.output_root, config.dataset_root, experiment, seed, ece_bins, 24)
    summary = {
        **audit,
        "elapsed_seconds": time.time() - started,
        "split": "locked_test_interrupted_run_continuation",
        "checkpoint": str(checkpoint),
        "evaluation_dir": str(evaluation_dir),
        "artifact_manifest": str(evaluation_dir / "reproducibility" / "artifact_manifest.csv"),
        "xai_rows": int(len(xai)),
        "artifact_rows": int(len(manifest)),
        "q1_outputs": {key: str(value) for key, value in regenerated.items()},
        "raw_vs_calibrated": "both",
        "ci_method": "SubjectID-clustered bootstrap, configured 5000 iterations",
    }
    write_json(status_path, summary)
    return summary


def stage_report_existing(config, experiment: str, seed: int) -> dict[str, object]:
    """Finish reporting from an already completed frozen evaluation.

    This stage performs no model loading, validation/test inference, threshold
    selection or calibration fitting.  It is specifically for recovery when a
    reporting-only failure occurred after locked-test artifacts were stored.
    """
    started = time.time()
    run_dir = config.run_dir(experiment, seed)
    evaluation_dir = run_dir / "evaluation"
    required = [
        run_dir / "checkpoints" / "best_overall.pt",
        evaluation_dir / "reproducibility" / "locked_thresholds.json",
        evaluation_dir / "reproducibility" / "artifact_manifest.csv",
        evaluation_dir / "predictions" / "test_slice_classification_predictions.csv",
        evaluation_dir / "predictions" / "test_case_predictions_mean_all.csv",
        evaluation_dir / "predictions" / "test" / "prediction_manifest.csv",
        evaluation_dir / "metrics" / "classification_metrics_with_95ci.csv",
        evaluation_dir / "metrics" / "segmentation_metrics_with_95ci.csv",
        evaluation_dir / "metrics" / "segmentation_metrics_by_class_and_roi_size.csv",
        evaluation_dir / "metrics" / "test_case_error_report.csv",
        evaluation_dir / "xai" / "gradcam_metrics.csv",
        config.output_root / "test_access_registry" / experiment / f"seed_{seed}.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Stored frozen evaluation is incomplete: " + "; ".join(missing))
    thresholds = json.loads((evaluation_dir / "reproducibility" / "locked_thresholds.json").read_text(encoding="utf-8"))
    if thresholds.get("selected_on") != "validation":
        raise RuntimeError("Stored thresholds are not explicitly validation-selected.")
    regenerated = regenerate_q1_outputs(
        config.output_root,
        config.dataset_root,
        experiment,
        seed,
        int(config.section("statistics")["ece_bins"]),
        24,
    )
    audit = {
        "status": "completed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "experiment": experiment,
        "seed": seed,
        "recovery_type": "reporting_only_from_stored_frozen_artifacts",
        "model_inference_performed": False,
        "locked_test_reaccessed": False,
        "threshold_selection_performed": False,
        "calibration_refit_performed": False,
        "checkpoint_changed": False,
        "stored_thresholds": thresholds,
        "original_test_access_record": str(config.output_root / "test_access_registry" / experiment / f"seed_{seed}.json"),
        "source_artifact_manifest": str(evaluation_dir / "reproducibility" / "artifact_manifest.csv"),
        "q1_outputs": {key: str(value) for key, value in regenerated.items()},
        "reporting_fix": "Use canonical segmentation_metrics_by_class_and_roi_size.csv instead of an unproduced legacy filename.",
        "raw_vs_calibrated": "both stored frozen outputs",
        "ci_method": "SubjectID-clustered bootstrap, configured 5000 iterations",
    }
    write_json(run_dir / "evaluation" / "reproducibility" / "reporting_recovery_audit.json", audit)
    write_json(run_dir / "execution_evaluation_status.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["environment", "audit", "model-audit", "amp-smoke", "train", "evaluate", "resume-evaluate", "report-existing"])
    parser.add_argument("--experiment", default="TALON_FULL")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    assert_output_contract(config)
    status_path = EXECUTION_ROOT / f"last_{args.stage.replace('-', '_')}_attempt.json"
    attempt = {"stage": args.stage, "experiment": args.experiment, "seed": args.seed, "started_utc": utc_now(), "status": "running"}
    write_json(status_path, attempt)
    try:
        if args.stage == "environment":
            result = stage_environment(config)
        elif args.stage == "audit":
            result = stage_audit(config)
        elif args.stage == "model-audit":
            result = stage_model_audit(config)
        elif args.stage == "amp-smoke":
            result = stage_amp_smoke(config)
        elif args.stage == "train":
            result = stage_train(config, args.experiment, args.seed)
        elif args.stage == "report-existing":
            result = stage_report_existing(config, args.experiment, args.seed)
        elif args.stage == "resume-evaluate":
            result = stage_resume_evaluate(config, args.experiment, args.seed)
        else:
            result = stage_evaluate(config, args.experiment, args.seed)
        attempt.update({"status": "completed", "finished_utc": utc_now(), "result": result})
        write_json(status_path, attempt)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    except Exception as exc:
        attempt.update({
            "status": "failed", "finished_utc": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        write_json(status_path, attempt)
        raise


if __name__ == "__main__":
    main()
