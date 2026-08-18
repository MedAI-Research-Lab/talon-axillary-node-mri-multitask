"""Executable Hybrid TALON benchmark pipeline.

Only the TALON model builder is replaced. Data loading, locked grouped split,
losses, five sequential-teacher phases, optimizer resets, scheduler, early
stopping, threshold selection, calibration, test evaluation and Q1 reporting
are reused from the audited TALON Jupyter pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_PROJECT = PROJECT_ROOT
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
BASE_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_pipeline_stage.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "publication_config.local.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PROJECT_ROOT / "configs" / "publication_config.frozen.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXECUTION_ROOT = OUTPUT_ROOT / "execution"
LOCKED_PARENT_OUTPUT = PROJECT_ROOT / "outputs"
EXPERIMENT = "HYBRID_TALON"
EXPECTED_PARAMETERS = 4_930_849

sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from talon_publication.hybrid_talon import build_hybrid_talon
from talon_codex.config import load_config
from talon_codex.data import build_data_bundle
from talon_codex.losses import MultiTaskCriterion
from talon_codex.models import build_training_spatial_prior
from talon_codex.models.talon import TalonVariant
import talon_codex.training as training_module
from talon_codex.training import audit_auxiliary_gradients, training_weights


def _load_base_runner():
    spec = importlib.util.spec_from_file_location("locked_talon_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = PROJECT_ROOT
    module.WORKSPACE_ROOT = WORKSPACE_ROOT
    module.SOURCE_ROOT = SOURCE_ROOT
    module.CONFIG_PATH = CONFIG_PATH
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.EXECUTION_ROOT = EXECUTION_ROOT
    return module


BASE = _load_base_runner()
ORIGINAL_BUILD_TALON = training_module.build_talon


def hybrid_builder(model_config, prior, experiment_name):
    if experiment_name == EXPERIMENT:
        return build_hybrid_talon(model_config, prior)
    return ORIGINAL_BUILD_TALON(model_config, prior, experiment_name)


def install_hybrid_dispatch() -> None:
    """Patch the single model-construction seam used by the locked pipeline."""
    training_module.build_talon = hybrid_builder
    training_module.ABLATION_VARIANTS[EXPERIMENT] = TalonVariant()
    BASE.build_talon = hybrid_builder
    BASE.train_experiment = training_module.train_experiment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def assert_output_contract(config) -> None:
    actual = config.output_root.resolve()
    expected = OUTPUT_ROOT.resolve()
    forbidden = (WORKSPACE_ROOT / "talon_outputs").resolve()
    if actual != expected:
        raise RuntimeError(f"Hybrid output root mismatch: {actual} != {expected}")
    if actual == forbidden or forbidden in actual.parents:
        raise RuntimeError("Legacy talon_outputs is write-protected.")


def stage_structure_audit() -> dict[str, object]:
    required_directories = [
        "configs", "docs", "src", "notebooks", "scripts", "reports",
        "outputs/audit", "outputs/cleaned_masks", "outputs/execution",
        "outputs/runs", "outputs/q1_report", "outputs/test_access_registry",
        "outputs/statistical_comparisons", "outputs/cross_validation",
        "outputs/external_validation", "outputs/reports",
    ]
    required_files = [
        "configs/publication_config.frozen.json",
        "configs/publication_config.example.json",
        "configs/repeated_holdout_design.json",
        "src/talon_publication/hybrid_talon.py",
        "scripts/audit_architecture.py",
        "scripts/run_publication_pipeline.py",
        "notebooks/00_repository_overview.ipynb",
        "notebooks/01_architecture_and_configuration.ipynb",
        "notebooks/02_five_seed_results.ipynb",
        "notebooks/03_reproduce_paper_outputs.ipynb",
    ]
    for relative in required_directories:
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
    missing = [relative for relative in required_files if not (PROJECT_ROOT / relative).exists()]
    result = {
        "status": "passed" if not missing else "failed",
        "created_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "required_directories": required_directories,
        "required_files": required_files,
        "missing": missing,
    }
    write_json(EXECUTION_ROOT / "folder_structure_audit.json", result)
    if missing:
        raise FileNotFoundError("Hybrid benchmark structure is incomplete: " + ", ".join(missing))
    return result


def stage_data_and_split_audit(config) -> dict[str, object]:
    prepared = BASE.prepare_data(config, seed=config.seed, hash_audit=True)
    local_path = OUTPUT_ROOT / "audit" / "split_assignments" / f"split_assignments_seed_{config.seed}.csv"
    locked_path = LOCKED_PARENT_OUTPUT / "audit" / "split_assignments" / f"split_assignments_seed_{config.seed}.csv"
    local = pd.read_csv(local_path).sort_values(list(pd.read_csv(local_path, nrows=0).columns)).reset_index(drop=True)
    locked = pd.read_csv(locked_path).sort_values(list(pd.read_csv(locked_path, nrows=0).columns)).reset_index(drop=True)
    same_columns = list(local.columns) == list(locked.columns)
    exact_split_match = same_columns and local.equals(locked)
    architecture_path = PROJECT_ROOT / "reports" / "architecture_audit.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    result = {
        "status": "passed" if exact_split_match and architecture.get("all_required_audits_pass") else "failed",
        "created_utc": utc_now(),
        "local_split": str(local_path),
        "locked_reference_split": str(locked_path),
        "exact_split_match": exact_split_match,
        "split_rows": len(local),
        "architecture_audit": str(architecture_path),
        "architecture_audit_pass": bool(architecture.get("all_required_audits_pass")),
        "segmentation_state_entries": architecture["segmentation_checkpoint_shape_audit"]["left_entries"],
        "legacy_classifier_state_entries": architecture["legacy_classifier_shape_audit"]["left_entries"],
        "prepared_slices": int(sum(len(frame) for frame in prepared["bundle"].frames.values())),
    }
    write_json(EXECUTION_ROOT / "data_split_architecture_audit.json", result)
    if result["status"] != "passed":
        raise RuntimeError("Exact split or architecture parity audit failed.")
    return result


def stage_model_preflight(config) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Hybrid TALON training.")
    bundle = build_data_bundle(config, seed=config.seed)
    prior = build_training_spatial_prior(
        [Path(path) for path in bundle.frames["train"]["ResolvedRoiPath"]],
        int(config.get("mask_threshold", 127)),
    )
    model = build_hybrid_talon(config.section("model"), prior).to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Hybrid parameter mismatch: {parameter_count} != {EXPECTED_PARAMETERS}")
    class_weights, positive_weight = training_weights(bundle.frames["train"], config)
    criterion = MultiTaskCriterion(config.section("loss"), class_weights, positive_weight, set()).to("cuda")
    phase = str(config.section("training")["teacher_phases"][0]["name"])
    gradient_path = EXECUTION_ROOT / "hybrid_auxiliary_gradient_audit.csv"
    if gradient_path.exists():
        gradient = pd.read_csv(gradient_path)
    else:
        gradient = audit_auxiliary_gradients(model, criterion, bundle.loaders["train"], torch.device("cuda"), phase, set())
        gradient.to_csv(gradient_path, index=False)
    if not bool(gradient["passes"].all()):
        raise RuntimeError(f"Hybrid auxiliary gradient audit failed: {gradient_path}")

    from talon_codex.training import _move_batch
    model.train()
    batch = _move_batch(next(iter(bundle.loaders["train"])), torch.device("cuda"))
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(batch["image"])
        losses = criterion(outputs, batch, phase, talon=True)
    total = losses["total"]
    total.backward()
    output_finite = {key: bool(torch.isfinite(value).all()) for key, value in outputs.items() if torch.is_tensor(value)}
    loss_finite = {key: bool(torch.isfinite(value).all()) for key, value in losses.items()}
    required_outputs = {"lesion_logits", "cls_logits", "dataset_prior", "image_location_logits", "descriptors", "legacy_descriptors", "mask_quality"}
    missing_outputs = sorted(required_outputs - set(outputs))
    result = {
        "status": "passed" if all(output_finite.values()) and all(loss_finite.values()) and not missing_outputs else "failed",
        "created_utc": utc_now(),
        "device": torch.cuda.get_device_name(0),
        "trainable_parameters": parameter_count,
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "output_finite": output_finite,
        "loss_finite": loss_finite,
        "missing_outputs": missing_outputs,
        "gradient_audit": str(gradient_path),
        "teacher_phases": config.section("training")["teacher_phases"],
    }
    write_json(EXECUTION_ROOT / "hybrid_model_preflight.json", result)
    del model, batch, outputs, losses, total
    torch.cuda.empty_cache()
    if result["status"] != "passed":
        raise RuntimeError("Hybrid real-batch preflight failed.")
    return result


def stage_train(config, seed: int) -> dict[str, object]:
    return BASE.stage_train(config, EXPERIMENT, seed)


def stage_evaluate(config, seed: int) -> dict[str, object]:
    return BASE.stage_evaluate(config, EXPERIMENT, seed)


def stage_compare(config, seed: int) -> dict[str, object]:
    """Paired comparison against the already frozen capacity-matched U-Net."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, confusion_matrix, roc_auc_score,
    )
    from talon_codex.analysis.statistics import (
        mcnemar_exact, paired_classification_metric_bootstrap,
        paired_patient_bootstrap_difference, paired_wilcoxon,
    )

    destination = OUTPUT_ROOT / "benchmark_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    roots = {
        "HYBRID_TALON": config.run_dir(EXPERIMENT, seed) / "evaluation",
        "UNET_MTL_MASK_GUIDED": LOCKED_PARENT_OUTPUT / "runs" / "UNET_MTL_MASK_GUIDED" / f"seed_{seed}" / "evaluation",
    }
    required = []
    for root in roots.values():
        required.extend([
            root / "predictions" / "test_case_predictions_mean_all.csv",
            root / "metrics" / "test_slice_segmentation_metrics.csv",
            root / "metrics" / "test_case_error_report.csv",
            root / "reproducibility" / "locked_thresholds.json",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Stored comparison artifacts missing: " + "; ".join(missing))

    case_tables = {}
    segmentation_tables = {}
    thresholds = {}
    summary_rows = []
    confusion_rows = []
    for name, root in roots.items():
        cases = pd.read_csv(root / "predictions" / "test_case_predictions_mean_all.csv")
        segmentation = pd.read_csv(root / "metrics" / "test_slice_segmentation_metrics.csv")
        error = pd.read_csv(root / "metrics" / "test_case_error_report.csv")
        locked = json.loads((root / "reproducibility" / "locked_thresholds.json").read_text(encoding="utf-8"))
        threshold = float(locked["classification_threshold"])
        for calibration, probability_column in (("raw", "class_probability"), ("calibrated", "calibrated_probability")):
            probability = cases[probability_column]
            prediction = (probability >= threshold).astype(int)
            matrix = confusion_matrix(cases["class_id"], prediction, labels=[0, 1])
            tn, fp, fn, tp = matrix.ravel()
            summary_rows.append({
                "model": name, "calibration": calibration, "cases": len(cases),
                "auroc": roc_auc_score(cases["class_id"], probability),
                "auprc": average_precision_score(cases["class_id"], probability),
                "accuracy": accuracy_score(cases["class_id"], prediction),
                "balanced_accuracy": balanced_accuracy_score(cases["class_id"], prediction),
                "brier": brier_score_loss(cases["class_id"], probability),
                "mean_dice": segmentation["dice"].mean(), "mean_iou": segmentation["iou"].mean(),
                "mean_off_target_components_per_case": error["off_target_component_occurrences"].mean(),
                "classification_threshold": threshold,
                "segmentation_threshold": float(locked["segmentation_threshold"]),
            })
            confusion_rows.append({"model": name, "calibration": calibration, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        case_tables[name] = cases
        segmentation_tables[name] = segmentation
        thresholds[name] = locked

    keys = ["SubjectID", "CaseID"]
    columns = keys + ["class_id", "class_probability", "calibrated_probability"]
    paired_cases = case_tables["HYBRID_TALON"][columns].merge(
        case_tables["UNET_MTL_MASK_GUIDED"][columns], on=keys, suffixes=("_hybrid", "_unet"), validate="one_to_one",
    )
    if not (paired_cases["class_id_hybrid"] == paired_cases["class_id_unet"]).all():
        raise RuntimeError("Hybrid and U-Net case labels differ.")
    paired_cases["class_id"] = paired_cases["class_id_hybrid"]
    paired_cases.to_csv(destination / "paired_case_predictions.csv", index=False)

    paired_rows = []
    for calibration, prefix in (("raw", "class_probability"), ("calibrated", "calibrated_probability")):
        a, b = f"{prefix}_hybrid", f"{prefix}_unet"
        for metric in ("auroc", "auprc", "brier"):
            result = paired_classification_metric_bootstrap(
                paired_cases, a, b, metric, iterations=int(config.section("statistics")["bootstrap_iterations"]), seed=seed,
            )
            paired_rows.append({"endpoint": f"case_{calibration}_{metric}", "method": "paired SubjectID-clustered bootstrap", **result})
        pred_a = (paired_cases[a] >= float(thresholds["HYBRID_TALON"]["classification_threshold"])).astype(int)
        pred_b = (paired_cases[b] >= float(thresholds["UNET_MTL_MASK_GUIDED"]["classification_threshold"])).astype(int)
        mc = mcnemar_exact(paired_cases["class_id"], pred_a, pred_b)
        paired_rows.append({"endpoint": f"case_{calibration}_accuracy", "method": "exact McNemar", "model_a": (pred_a == paired_cases["class_id"]).mean(), "model_b": (pred_b == paired_cases["class_id"]).mean(), "difference": (pred_a == paired_cases["class_id"]).mean() - (pred_b == paired_cases["class_id"]).mean(), "p_two_sided": mc["p_value"], "lower": float("nan"), "upper": float("nan")})

    seg_keys = ["SubjectID", "CaseID", "ImageName"]
    paired_seg = segmentation_tables["HYBRID_TALON"][seg_keys + ["dice", "iou"]].merge(
        segmentation_tables["UNET_MTL_MASK_GUIDED"][seg_keys + ["dice", "iou"]],
        on=seg_keys, suffixes=("_hybrid", "_unet"), validate="one_to_one",
    )
    paired_seg.to_csv(destination / "paired_slice_segmentation_metrics.csv", index=False)
    for endpoint in ("dice", "iou"):
        bootstrap = paired_patient_bootstrap_difference(
            paired_seg, f"{endpoint}_hybrid", f"{endpoint}_unet",
            iterations=int(config.section("statistics")["bootstrap_iterations"]), seed=seed,
        )
        wilcoxon = paired_wilcoxon(paired_seg[f"{endpoint}_hybrid"], paired_seg[f"{endpoint}_unet"])
        paired_rows.append({"endpoint": f"slice_{endpoint}", "method": "paired SubjectID-clustered bootstrap + Wilcoxon", "model_a": paired_seg[f"{endpoint}_hybrid"].mean(), "model_b": paired_seg[f"{endpoint}_unet"].mean(), **bootstrap, "wilcoxon_p": wilcoxon["p_value"]})

    pd.DataFrame(summary_rows).to_csv(destination / "hybrid_vs_unet_summary.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(destination / "case_confusion_matrices.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(destination / "paired_statistical_comparisons.csv", index=False)
    result = {
        "status": "completed", "created_utc": utc_now(), "seed": seed,
        "hybrid_evaluation": str(roots["HYBRID_TALON"]),
        "locked_unet_evaluation": str(roots["UNET_MTL_MASK_GUIDED"]),
        "paired_cases": len(paired_cases), "paired_slices": len(paired_seg),
        "summary": str(destination / "hybrid_vs_unet_summary.csv"),
        "paired_statistics": str(destination / "paired_statistical_comparisons.csv"),
        "unet_retrained": False,
    }
    write_json(destination / "comparison_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["structure-audit", "audit", "preflight", "train", "evaluate", "compare"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    install_hybrid_dispatch()
    config = load_config(CONFIG_PATH)
    assert_output_contract(config)
    status_path = EXECUTION_ROOT / f"last_{args.stage.replace('-', '_')}_attempt.json"
    attempt = {"stage": args.stage, "seed": args.seed, "started_utc": utc_now(), "status": "running"}
    write_json(status_path, attempt)
    try:
        if args.stage == "structure-audit":
            result = stage_structure_audit()
        elif args.stage == "audit":
            result = stage_data_and_split_audit(config)
        elif args.stage == "preflight":
            result = stage_model_preflight(config)
        elif args.stage == "train":
            result = stage_train(config, args.seed)
        elif args.stage == "compare":
            result = stage_compare(config, args.seed)
        else:
            result = stage_evaluate(config, args.seed)
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
