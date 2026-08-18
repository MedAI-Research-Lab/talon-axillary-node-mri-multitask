"""Run approved repeated-holdout seeds without changing the seed-42 protocol.

This module is intentionally inert on import.  Training is permitted only for
the predeclared new-training seeds.  Seed 42 remains a copied immutable
reference run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_common import (
    DATA_SPLIT_ROOT,
    EXECUTION_ROOT,
    MODELS,
    NEW_TRAINING_SEEDS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    config_for_seed,
    ensure_new_training_seed,
    load_canonical_hybrid_pipeline,
    scientific_config_hash,
    seed_split_directory,
    write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_seed(seed: int, pipeline) -> dict[str, object]:
    """Create and verify one stratified SubjectID-grouped split."""
    ensure_new_training_seed(seed)
    config = config_for_seed(seed)
    reference_hash = scientific_config_hash(config_for_seed(42).raw)
    runtime_hash = scientific_config_hash(config.raw)
    if runtime_hash != reference_hash:
        raise RuntimeError("Scientific configuration differs from the seed-42 reference.")
    split_dir = seed_split_directory(seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot(
        split_dir / f"runtime_config_seed_{seed}.json",
        extra={"scientific_config_hash": scientific_config_hash(config.raw)},
    )

    prepared = pipeline.BASE.prepare_data(config, seed=seed, hash_audit=True)
    duplicate_table = prepared["audits"]["duplicate_images"]
    cross_split_duplicates = (
        int((duplicate_table["DuplicateRisk"] == "cross_split_leakage").sum())
        if not duplicate_table.empty and "DuplicateRisk" in duplicate_table
        else 0
    )
    if cross_split_duplicates:
        raise RuntimeError(f"Detected {cross_split_duplicates} cross-split duplicate-image rows.")
    generated = OUTPUT_ROOT / "audit" / "split_assignments"
    expected_files = (
        f"split_assignments_seed_{seed}.csv",
        f"split_counts_seed_{seed}.csv",
        f"roi_edges_train_only_seed_{seed}.json",
    )
    for name in expected_files:
        source = generated / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, split_dir / name)

    assignments = pd.read_csv(split_dir / f"split_assignments_seed_{seed}.csv")
    required = {"SubjectID", "CaseID", "ClassName", "Split"}
    if not required.issubset(assignments.columns):
        raise RuntimeError(f"Split columns missing: {sorted(required - set(assignments.columns))}")
    subject_leakage = assignments.groupby("SubjectID")["Split"].nunique().gt(1)
    if bool(subject_leakage.any()):
        raise RuntimeError("A SubjectID appears in more than one split.")
    case_leakage = assignments.groupby("CaseID")["Split"].nunique().gt(1)
    if bool(case_leakage.any()):
        raise RuntimeError("A CaseID appears in more than one split.")
    class_by_split = assignments.groupby(["Split", "ClassName"]).size().unstack(fill_value=0)
    if set(class_by_split.index) != {"train", "validation", "test"}:
        raise RuntimeError("Train/validation/test split is incomplete.")
    if (class_by_split == 0).any().any():
        raise RuntimeError("At least one split is missing one diagnostic class.")

    reference_path = DATA_SPLIT_ROOT / "seed_42" / "split_assignments_seed_42.csv"
    reference = pd.read_csv(reference_path)[["CaseID", "Split"]]
    candidate = assignments[["CaseID", "Split"]]
    comparison = reference.merge(candidate, on="CaseID", suffixes=("_42", f"_{seed}"), validate="one_to_one")
    changed_cases = int((comparison["Split_42"] != comparison[f"Split_{seed}"]).sum())
    if changed_cases == 0:
        raise RuntimeError(f"Seed {seed} reproduced the seed-42 split; a distinct test set was required.")

    result = {
        "status": "passed",
        "seed": seed,
        "split_seed": config.get("split_seed"),
        "split_unit": "SubjectID",
        "reporting_unit": "CaseID",
        "subject_leakage": False,
        "case_leakage": False,
        "cross_split_duplicate_rows": cross_split_duplicates,
        "same_split_used_by_both_models": True,
        "changed_case_assignments_vs_seed_42": changed_cases,
        "counts": class_by_split.to_dict(),
        "slices": {name: int(len(frame)) for name, frame in prepared["bundle"].frames.items()},
        "scientific_config_hash": runtime_hash,
        "split_directory": str(split_dir),
    }
    write_json(split_dir / "split_integrity_audit.json", result)
    return result


def train_model(seed: int, model: str, pipeline) -> dict[str, object]:
    ensure_new_training_seed(seed)
    config = config_for_seed(seed)
    if model == "HYBRID_TALON":
        return pipeline.stage_train(config, seed)
    if model == "UNET_MTL_MASK_GUIDED":
        # The canonical trainer constructs the same capacity-matched U-Net used
        # by seed 42; Hybrid dispatch does not alter its TALON_FULL reference.
        return pipeline.BASE.stage_train(config, model, seed)
    raise ValueError(model)


def evaluate_model(seed: int, model: str, pipeline) -> dict[str, object]:
    ensure_new_training_seed(seed)
    config = config_for_seed(seed)
    if model == "HYBRID_TALON":
        return pipeline.stage_evaluate(config, seed)
    if model == "UNET_MTL_MASK_GUIDED":
        run_dir = config.run_dir(model, seed)
        registry = config.output_root / "test_access_registry" / model / f"seed_{seed}.json"
        completed = run_dir / "execution_evaluation_status.json"
        if registry.exists() and not completed.exists():
            return pipeline.BASE.stage_resume_evaluate(config, model, seed)
        return pipeline.BASE.stage_evaluate(config, model, seed)
    raise ValueError(model)


def teacher_comparison(seed: int) -> dict[str, object]:
    ensure_new_training_seed(seed)
    script = PROJECT_ROOT / "scripts" / "run_teacher_checkpoint_comparison.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--seed", str(seed)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Teacher checkpoint comparison failed for seed {seed}.")
    status_path = OUTPUT_ROOT / "runs" / "HYBRID_TALON" / f"seed_{seed}" / "teacher_checkpoint_comparison" / "teacher_checkpoint_validation_comparison_status.json"
    return json.loads(status_path.read_text(encoding="utf-8"))


def compare_models(seed: int, pipeline) -> dict[str, object]:
    ensure_new_training_seed(seed)
    result = pipeline.stage_compare(config_for_seed(seed), seed)
    temporary = OUTPUT_ROOT / "benchmark_comparison"
    destination = OUTPUT_ROOT / "repeated_holdout_comparison" / "per_seed" / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    for source in temporary.iterdir():
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    return {**result, "archived_per_seed": str(destination)}


def relative_overlap(seed: int) -> dict[str, object]:
    script = PROJECT_ROOT / "scripts" / "run_relative_overlap.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--seed", str(seed)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Relative-overlap analysis failed for seed {seed}.")
    path = OUTPUT_ROOT / "repeated_holdout_comparison" / "per_seed" / f"seed_{seed}" / "relative_overlap_analysis" / "relative_overlap_analysis_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def output_contract(seed: int) -> dict[str, object]:
    script = PROJECT_ROOT / "scripts" / "audit_output_contract.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--seed", str(seed)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Seed-42 output-schema parity failed for seed {seed}.")
    path = OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / f"output_contract_seed_{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def execute_stage(seed: int, stage: str, pipeline) -> dict[str, object]:
    if stage == "prepare":
        return prepare_seed(seed, pipeline)
    if stage == "train-hybrid":
        return train_model(seed, "HYBRID_TALON", pipeline)
    if stage == "train-unet":
        return train_model(seed, "UNET_MTL_MASK_GUIDED", pipeline)
    if stage == "evaluate-hybrid":
        return evaluate_model(seed, "HYBRID_TALON", pipeline)
    if stage == "evaluate-unet":
        return evaluate_model(seed, "UNET_MTL_MASK_GUIDED", pipeline)
    if stage == "teacher-hybrid":
        return teacher_comparison(seed)
    if stage == "compare":
        return compare_models(seed, pipeline)
    if stage == "relative-overlap":
        return relative_overlap(seed)
    if stage == "output-contract":
        return output_contract(seed)
    raise ValueError(stage)


FULL_SEQUENCE = (
    "prepare",
    "train-hybrid",
    "train-unet",
    "evaluate-hybrid",
    "evaluate-unet",
    "teacher-hybrid",
    "compare",
    "relative-overlap",
    "output-contract",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(*FULL_SEQUENCE, "full"))
    parser.add_argument("--seed", type=int, required=True, choices=NEW_TRAINING_SEEDS)
    args = parser.parse_args()
    seed = int(args.seed)
    pipeline = load_canonical_hybrid_pipeline()
    stages = FULL_SEQUENCE if args.stage == "full" else (args.stage,)
    status_path = EXECUTION_ROOT / f"seed_{seed}_sequence_status.json"
    status = {
        "status": "running",
        "started_utc": utc_now(),
        "seed": seed,
        "split_seed": seed,
        "stages": {name: {"status": "pending"} for name in FULL_SEQUENCE},
        "seed_42_retrained": False,
    }
    write_json(status_path, status)
    try:
        for stage in stages:
            started = time.time()
            status["current_stage"] = stage
            status["stages"][stage] = {"status": "running", "started_utc": utc_now()}
            write_json(status_path, status)
            result = execute_stage(seed, stage, pipeline)
            status["stages"][stage] = {
                "status": "completed",
                "finished_utc": utc_now(),
                "elapsed_seconds": time.time() - started,
                "result": result,
            }
            write_json(status_path, status)
        status.update({"status": "completed", "current_stage": None, "finished_utc": utc_now()})
        write_json(status_path, status)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
