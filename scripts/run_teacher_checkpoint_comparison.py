"""Validation-only comparison of five saved Hybrid TALON teacher checkpoints.

No optimizer, backward pass, weight update, training loader iteration, or test
loader access occurs in this script.  It is resume-safe at checkpoint level.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_common import OUTPUT_ROOT, SEEDS, config_for_seed, run_directory, sha256, write_json

from models.hybrid_talon import build_hybrid_talon
from talon_codex.analysis.statistics import classification_metrics, select_segmentation_threshold
from talon_codex.data import build_data_bundle
from talon_codex.evaluation import (
    GradCAM,
    _to_device,
    analyze_and_export_segmentation,
    case_aggregation,
    collect_raw_predictions,
    xai_overlap_metrics,
)
from talon_codex.models import build_training_spatial_prior
from talon_codex.reporting import write_q1_workbook


EXPECTED_CHECKPOINTS = (
    "phase_01_teacher1_anatomy_normal_best.pt",
    "phase_02_teacher2_candidate_search_best.pt",
    "phase_03_teacher3_doctor_ball_objectness_best.pt",
    "phase_04_teacher4_seg_cls_coupling_best.pt",
    "phase_05_teacher5_report_ready_polish_best.pt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gradcam_metrics_only(model, validation_loader, output_dir: Path, device: torch.device) -> pd.DataFrame:
    """Compute the seed-42 teacher XAI endpoints without test access."""
    target_layer = getattr(model, "neck", None) or getattr(model, "bottleneck", None)
    if target_layer is None:
        raise AttributeError("No supported Hybrid TALON Grad-CAM target layer found.")
    rows: list[dict[str, object]] = []
    model.eval()
    with GradCAM(model, target_layer) as cam:
        for raw_batch in validation_loader:
            batch = _to_device(raw_batch, device)
            heatmaps, logits = cam(batch["image"])
            with torch.no_grad():
                segmentation_probability = torch.sigmoid(model(batch["image"])["lesion_logits"])
            for index in range(len(raw_batch["SubjectID"])):
                heatmap = heatmaps[index, 0].detach().cpu().numpy()
                mask = batch["mask"][index, 0].detach().cpu().numpy()
                prediction = (
                    segmentation_probability[index, 0].detach().cpu().numpy() >= 0.5
                ).astype(np.uint8)
                rows.append(
                    {
                        "SubjectID": raw_batch["SubjectID"][index],
                        "CaseID": raw_batch["CaseID"][index],
                        "ImageName": raw_batch["ImageName"][index],
                        **xai_overlap_metrics(heatmap, mask, prediction),
                        "predicted_class": int(logits[index].argmax()),
                    }
                )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "gradcam_metrics.csv", index=False)
    return frame


def compare(seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    started = time.time()
    config = config_for_seed(seed)
    run_dir = run_directory("HYBRID_TALON", seed)
    checkpoint_dir = run_dir / "checkpoints"
    output_dir = run_dir / "teacher_checkpoint_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [checkpoint_dir / name for name in EXPECTED_CHECKPOINTS]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Hybrid teacher checkpoints: " + "; ".join(missing))

    # build_data_bundle is configured with seed == split_seed.  Only the train
    # frame is used to reconstruct the spatial prior; only validation is scored.
    bundle = build_data_bundle(config, seed=seed)
    prior = build_training_spatial_prior(
        [Path(path) for path in bundle.frames["train"]["ResolvedRoiPath"]],
        int(config.get("mask_threshold", 127)),
    )
    validation_loader = bundle.loaders["validation"]
    validation_rows = len(bundle.frames["validation"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, object]] = []

    for teacher_index, checkpoint in enumerate(checkpoints, start=1):
        phase_dir = output_dir / f"t{teacher_index:02d}"
        summary_path = phase_dir / "phase_summary.json"
        checkpoint_hash = sha256(checkpoint)
        if summary_path.exists():
            existing = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "completed"
                and existing.get("checkpoint_sha256") == checkpoint_hash
                and existing.get("evaluation_split") == "validation"
            ):
                records.append(existing["metrics"])
                continue

        phase_dir.mkdir(parents=True, exist_ok=True)
        model = build_hybrid_talon(config.section("model"), prior).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        raw = collect_raw_predictions(model, validation_loader, device)
        threshold, sweep = select_segmentation_threshold(
            raw, config.section("selection")["segmentation_thresholds"]
        )
        sweep.to_csv(phase_dir / "validation_threshold_sweep.csv", index=False)
        slice_metrics, predicted_components, _ = analyze_and_export_segmentation(
            raw, threshold, phase_dir, config.section("components")
        )
        slice_metrics.to_csv(phase_dir / "validation_slice_segmentation_metrics.csv", index=False)
        predicted_components.to_csv(phase_dir / "validation_predicted_components.csv", index=False)
        cases = case_aggregation(
            raw,
            str(config.section("selection")["case_aggregation_primary"]),
            int(config.section("selection")["top_k"]),
        )
        cases.to_csv(phase_dir / "validation_case_predictions.csv", index=False)
        classification = classification_metrics(
            cases["class_id"],
            cases["class_probability"],
            float(config.section("selection")["classification_threshold"]),
            int(config.section("statistics")["ece_bins"]),
        )
        xai = gradcam_metrics_only(model, validation_loader, phase_dir / "xai", device)
        if len(xai) != validation_rows:
            raise RuntimeError(f"Teacher {teacher_index} XAI rows {len(xai)} != validation rows {validation_rows}")

        metrics = {
            "teacher": teacher_index,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "segmentation_threshold": float(threshold),
            "validation_slices": int(len(slice_metrics)),
            "validation_cases": int(len(cases)),
            "mean_dice": float(slice_metrics["dice"].mean()),
            "mean_iou": float(slice_metrics["iou"].mean()),
            "off_target_fp_occurrences": int((predicted_components["category"] == "off_target_fp").sum()),
            "poorly_matched_occurrences": int((predicted_components["category"] == "poorly_matched").sum()),
            "xai_energy_inside_gt": float(xai["xai_energy_inside_gt"].mean()),
            "xai_off_target_attention": float(xai["xai_off_target_attention"].mean()),
            "xai_metrics_source": str(phase_dir / "xai" / "gradcam_metrics.csv"),
            "xai_reused_from_completed_artifacts": False,
            **classification,
        }
        records.append(metrics)
        write_json(
            summary_path,
            {
                "status": "completed",
                "created_utc": utc_now(),
                "training_performed": False,
                "optimizer_created": False,
                "backward_called": False,
                "weight_updates": False,
                "evaluation_split": "validation",
                "locked_test_accessed": False,
                "checkpoint_sha256": checkpoint_hash,
                "metrics": metrics,
            },
        )
        pd.DataFrame(records).to_csv(
            output_dir / "teacher_checkpoint_validation_partial.csv", index=False
        )
        del model, payload, raw, slice_metrics, predicted_components, cases, xai
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.DataFrame(records).sort_values("teacher").reset_index(drop=True)
    if len(table) != 5 or list(table["checkpoint"]) != list(EXPECTED_CHECKPOINTS):
        raise RuntimeError("Teacher comparison did not produce the five expected ordered rows.")
    csv_path = output_dir / "teacher_checkpoint_validation_comparison.csv"
    workbook_path = output_dir / "teacher_checkpoint_validation_comparison.xlsx"
    table.to_csv(csv_path, index=False)
    write_q1_workbook(
        {
            "teacher_validation": table,
            "analysis_notes": pd.DataFrame(
                [
                    {"item": "model", "value": "HYBRID_TALON"},
                    {"item": "seed", "value": seed},
                    {"item": "selection_split", "value": "validation only"},
                    {"item": "training_performed", "value": False},
                    {"item": "locked_test_accessed", "value": False},
                    {"item": "threshold", "value": "selected independently on validation for each checkpoint"},
                ]
            ),
        },
        workbook_path,
    )
    manifest = {
        "status": "completed",
        "created_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "model": "HYBRID_TALON",
        "seed": seed,
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "weight_updates": False,
        "evaluation_split": "validation",
        "locked_test_accessed": False,
        "checkpoints": len(table),
        "outputs": {"csv": str(csv_path), "workbook": str(workbook_path)},
        "output_sha256": {"csv": sha256(csv_path), "workbook": sha256(workbook_path)},
    }
    write_json(output_dir / "teacher_checkpoint_validation_comparison_status.json", manifest)
    return table, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    table, manifest = compare(int(args.seed))
    print(table.to_csv(index=False), flush=True)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
