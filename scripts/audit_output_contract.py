"""Verify that a completed seed has the same artifact schema as seed 42."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_common import MODELS, NEW_TRAINING_SEEDS, OUTPUT_ROOT, run_directory, write_json


DYNAMIC_PREFIXES = (
    "evaluation/predictions/test/arrays/",
    "evaluation/predictions/test/prediction_artifacts/",
    "evaluation/predictions/validation/arrays/",
    "evaluation/xai/gradcam/",
    "teacher_checkpoint_comparison/",
)

# Recovery/provenance records document how an interrupted run was completed.
# They are deliberately seed-specific metadata and do not alter the scientific
# output schema being compared with the immutable seed-42 reference.
PROVENANCE_FILES = {
    "evaluation_resume_audit.json",
}


def stable_file_schema(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(prefix) for prefix in DYNAMIC_PREFIXES):
            continue
        if relative in PROVENANCE_FILES:
            continue
        result.add(relative)
    return result


def csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def audit(seed: int) -> dict[str, object]:
    if seed == 42:
        raise ValueError("Seed 42 is the reference schema, not a candidate.")
    model_results: dict[str, object] = {}
    all_pass = True
    for model in MODELS:
        reference = run_directory(model, 42)
        candidate = run_directory(model, seed)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        reference_schema = stable_file_schema(reference)
        candidate_schema = stable_file_schema(candidate)
        missing = sorted(reference_schema - candidate_schema)
        extra = sorted(candidate_schema - reference_schema)
        checkpoint_reference = sorted(path.name for path in (reference / "checkpoints").glob("*.pt"))
        checkpoint_candidate = sorted(path.name for path in (candidate / "checkpoints").glob("*.pt"))
        csv_contracts = (
            "training_history/training_history.csv",
            "evaluation/metrics/classification_metrics_with_95ci.csv",
            "evaluation/metrics/segmentation_metrics_with_95ci.csv",
            "evaluation/metrics/test_slice_segmentation_metrics.csv",
            "evaluation/predictions/test_case_predictions_mean_all.csv",
            "evaluation/component_analysis/test_predicted_components.csv",
        )
        column_checks = {
            relative: csv_columns(reference / relative) == csv_columns(candidate / relative)
            for relative in csv_contracts
        }
        teacher_ok = True
        teacher_column_contract = True
        teacher_phase_contract = True
        if model == "HYBRID_TALON":
            reference_teacher_csv = reference / "teacher_checkpoint_comparison" / "teacher_checkpoint_validation_comparison.csv"
            teacher_csv = candidate / "teacher_checkpoint_comparison" / "teacher_checkpoint_validation_comparison.csv"
            teacher_ok = reference_teacher_csv.exists() and teacher_csv.exists() and len(pd.read_csv(teacher_csv)) == 5
            teacher_column_contract = teacher_ok and csv_columns(reference_teacher_csv) == csv_columns(teacher_csv)
            teacher_phase_contract = all(
                (candidate / "teacher_checkpoint_comparison" / f"t{index:02d}" / "phase_summary.json").exists()
                for index in range(1, 6)
            )
        passed = (
            not missing
            and not extra
            and checkpoint_reference == checkpoint_candidate
            and all(column_checks.values())
            and teacher_ok
            and teacher_column_contract
            and teacher_phase_contract
        )
        all_pass = all_pass and passed
        model_results[model] = {
            "passed": passed,
            "missing_static_files": missing,
            "extra_static_files": extra,
            "checkpoint_names_identical": checkpoint_reference == checkpoint_candidate,
            "checkpoint_names": checkpoint_candidate,
            "csv_column_contracts": column_checks,
            "hybrid_teacher_five_rows": teacher_ok if model == "HYBRID_TALON" else "not_applicable",
            "hybrid_teacher_columns_identical": teacher_column_contract if model == "HYBRID_TALON" else "not_applicable",
            "hybrid_teacher_phase_directories_complete": teacher_phase_contract if model == "HYBRID_TALON" else "not_applicable",
        }
    result = {
        "status": "passed" if all_pass else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "reference_seed": 42,
        "note": "Dynamic patient/image artifact paths and row counts may differ; the scientific artifact schema must not.",
        "models": model_results,
    }
    destination = OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / f"output_contract_seed_{seed}.json"
    write_json(destination, result)
    if not all_pass:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=NEW_TRAINING_SEEDS)
    args = parser.parse_args()
    print(json.dumps(audit(int(args.seed)), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
