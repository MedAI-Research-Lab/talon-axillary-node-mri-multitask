"""Reproducible training loops, sequential teachers and checkpoint discipline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from .config import ResearchConfig
from .data import DataBundle, read_binary_mask, seed_everything
from .losses import MultiTaskCriterion, estimate_training_weights, phase_activation, soft_dice
from .models import ABLATION_VARIANTS, build_capacity_matched_baseline, build_talon, build_training_spatial_prior


@dataclass
class EpochSummary:
    epoch: int
    phase: str
    split: str
    total_loss: float
    dice: float
    classification_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    lesion_recall: float
    segmentation_precision: float
    empty_prediction_rate: float
    missed_target_rate: float
    pred_gt_area_ratio: float
    localization_hit_rate: float
    box_iou: float
    checkpoint_score: float
    learning_rate: float
    loss_components: dict[str, float]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def training_weights(train_frame: pd.DataFrame, config: ResearchConfig) -> tuple[Tensor, Tensor]:
    counts = train_frame["class_id"].value_counts().astype(int).to_dict()
    foreground = 0
    total = 0
    for path in train_frame["ResolvedRoiPath"]:
        mask = read_binary_mask(Path(path), int(config.get("mask_threshold", 127)))
        foreground += int(mask.sum())
        total += int(mask.size)
    cap = float(config.section("loss")["segmentation"]["positive_weight_cap"])
    return estimate_training_weights(counts, foreground, total - foreground, cap)


def common_checkpoint_score(summary: EpochSummary, weights: Mapping[str, float]) -> float:
    """Model-agnostic final TALON/baseline checkpoint criterion."""
    return (
        float(weights["dice"]) * summary.dice
        + float(weights["balanced_accuracy"]) * summary.balanced_accuracy
        + float(weights["macro_f1"]) * summary.macro_f1
        + float(weights["segmentation_recall"]) * summary.lesion_recall
        - float(weights["empty_prediction_penalty"]) * summary.empty_prediction_rate
        - float(weights["missed_target_penalty"]) * summary.missed_target_rate
    )


def talon_phase_scores(summary: EpochSummary) -> dict[str, float]:
    """Original TALON phase-specific validation scores."""
    ratio = max(summary.pred_gt_area_ratio, 1e-6)
    ratio_score = float(np.exp(-abs(np.log(ratio))))
    dice_cls_hmean = 2 * summary.dice * summary.balanced_accuracy / max(summary.dice + summary.balanced_accuracy, 1e-6)
    seg_pr_hmean = 2 * summary.segmentation_precision * summary.lesion_recall / max(summary.segmentation_precision + summary.lesion_recall, 1e-6)
    localization = 0.6 * summary.localization_hit_rate + 0.4 * summary.box_iou
    loss = summary.loss_components
    empty, missed = summary.empty_prediction_rate, summary.missed_target_rate
    return {
        "teacher1_stability": 0.25 * summary.dice + 0.22 * summary.lesion_recall + 0.18 * summary.balanced_accuracy + 0.15 * ratio_score + 0.10 * seg_pr_hmean + 0.10 * summary.macro_f1 - 0.06 * empty - 0.06 * missed,
        "recall_safe": 0.48 * summary.lesion_recall + 0.24 * summary.dice + 0.18 * ratio_score + 0.10 * dice_cls_hmean - 0.07 * empty - 0.07 * missed - 0.03 * loss.get("candidate_recall", 0.0),
        "doctor_ball": 0.30 * summary.dice + 0.24 * summary.lesion_recall + 0.16 * seg_pr_hmean + 0.12 * summary.balanced_accuracy + 0.10 * ratio_score + 0.08 * dice_cls_hmean + 0.14 * localization - 0.07 * empty - 0.07 * missed - 0.04 * loss.get("doctor_ball", 0.0),
        "dual_strong": 0.58 * dice_cls_hmean + 0.18 * summary.lesion_recall + 0.14 * ratio_score + 0.10 * summary.macro_f1 + 0.10 * localization - 0.06 * empty - 0.06 * missed - 0.03 * loss.get("segmentation_classification_coupling", 0.0),
        "final_polish": 0.40 * dice_cls_hmean + 0.18 * summary.balanced_accuracy + 0.16 * summary.dice + 0.12 * summary.lesion_recall + 0.10 * localization + 0.06 * ratio_score + 0.04 * seg_pr_hmean - 0.06 * empty - 0.06 * missed,
        "joint_score": 0.32 * dice_cls_hmean + 0.14 * summary.dice + 0.22 * summary.balanced_accuracy + 0.08 * summary.macro_f1 + 0.12 * summary.lesion_recall + 0.16 * localization + 0.04 * seg_pr_hmean + 0.04 * ratio_score - 0.12 * empty - 0.14 * missed,
    }


def _box_iou(prediction: Tensor, target: Tensor) -> float:
    scores = []
    for pred, true in zip(prediction, target):
        pred_points, true_points = torch.nonzero(pred[0], as_tuple=False), torch.nonzero(true[0], as_tuple=False)
        if not len(pred_points) or not len(true_points):
            scores.append(0.0)
            continue
        py0, px0 = pred_points.min(0).values; py1, px1 = pred_points.max(0).values
        ty0, tx0 = true_points.min(0).values; ty1, tx1 = true_points.max(0).values
        iy0, ix0 = torch.maximum(py0, ty0), torch.maximum(px0, tx0)
        iy1, ix1 = torch.minimum(py1, ty1), torch.minimum(px1, tx1)
        intersection = max(0, int(iy1 - iy0 + 1)) * max(0, int(ix1 - ix0 + 1))
        pred_area = int((py1 - py0 + 1) * (px1 - px0 + 1)); true_area = int((ty1 - ty0 + 1) * (tx1 - tx0 + 1))
        scores.append(intersection / max(pred_area + true_area - intersection, 1))
    return float(np.mean(scores)) if scores else 0.0


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: MultiTaskCriterion,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    phase_name: str,
    talon: bool,
    epoch: int,
    clip_norm: float,
    amp: bool,
) -> EpochSummary:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = correct = 0
    dice_sum = recall_sum = precision_sum = 0.0
    empty_count = missed_count = 0
    pred_area_sum = gt_area_sum = localization_hits = box_iou_sum = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(batch["image"])
            losses = criterion(outputs, batch, phase_name, talon)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(batch["image"].shape[0])
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        probability = torch.sigmoid(outputs["lesion_logits"])
        prediction = probability >= 0.5
        target = batch["mask"] > 0.5
        intersection = (prediction & target).sum((1, 2, 3)).float()
        pred_area = prediction.sum((1, 2, 3)).float()
        gt_area = target.sum((1, 2, 3)).float()
        dice_sum += float(((2 * intersection + 1e-6) / (prediction.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).sum())
        recall_sum += float(((intersection + 1e-6) / (gt_area + 1e-6)).sum())
        precision_sum += float(((intersection + 1e-6) / (pred_area + 1e-6)).sum())
        empty_count += int((pred_area == 0).sum())
        missed_count += int((intersection == 0).sum())
        pred_area_sum += float(pred_area.sum()); gt_area_sum += float(gt_area.sum())
        localization_hits += float((intersection > 0).sum())
        box_iou_sum += _box_iou(prediction, target) * batch_size
        class_prediction = outputs["cls_logits"].argmax(1)
        correct += int((class_prediction == batch["class_id"]).sum())
        labels.extend(batch["class_id"].detach().cpu().tolist())
        predictions.extend(class_prediction.detach().cpu().tolist())
    denominator = max(sample_count, 1)
    averaged = {name: value / denominator for name, value in totals.items()}
    dice = dice_sum / denominator
    recall = recall_sum / denominator
    precision = precision_sum / denominator
    accuracy = correct / denominator
    balanced = balanced_accuracy_score(labels, predictions) if labels else 0.0
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0) if labels else 0.0
    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    return EpochSummary(
        epoch, phase_name, "train" if training else "validation", averaged.get("total", np.nan),
        dice, accuracy, balanced, macro_f1, recall, precision,
        empty_count / denominator, missed_count / denominator, pred_area_sum / max(gt_area_sum, 1.0),
        localization_hits / denominator, box_iou_sum / denominator, 0.0, lr, averaged,
    )


def _checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, summary: EpochSummary, experiment: str, seed: int) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "summary": asdict(summary), "experiment": experiment, "seed": seed,
        "torch_version": torch.__version__,
        "model_metadata": {"base_channels": getattr(model, "base_channels", None)},
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint transactionally so an interrupted write is recoverable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def audit_auxiliary_gradients(
    model: nn.Module,
    criterion: MultiTaskCriterion,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    phase_name: str,
    disabled_auxiliary: set[str],
) -> pd.DataFrame:
    """Verify that each enabled TALON auxiliary objective reaches model parameters."""
    batch = _move_batch(next(iter(loader)), device)
    model.train()
    outputs = model(batch["image"])
    losses = criterion(outputs, batch, phase_name, talon=True)
    auxiliary_names = [
        "focus", "descriptor", "centroid", "location", "low_prior", "outside_body",
        "doctor_ball", "candidate_recall", "segmentation_classification_coupling",
    ]
    records = []
    activation = phase_activation(phase_name)
    for name in auxiliary_names:
        model.zero_grad(set_to_none=True)
        value = losses.get(name)
        if value is None:
            records.append({"loss": name, "enabled": False, "gradient_norm": 0.0, "passes": True})
            continue
        value.backward(retain_graph=True)
        grad_norm = float(sum(parameter.grad.detach().norm().item() for parameter in model.parameters() if parameter.grad is not None))
        enabled = name not in disabled_auxiliary and float(activation.get(name, 0.0)) > 0
        records.append({"loss": name, "enabled": enabled, "gradient_norm": grad_norm, "passes": (grad_norm > 0) if enabled else True})
    model.zero_grad(set_to_none=True)
    return pd.DataFrame(records)


def train_experiment(config: ResearchConfig, bundle: DataBundle, experiment: str, seed: int) -> tuple[nn.Module, pd.DataFrame, Path]:
    """Train one experiment; this function is defined but is not run on import."""
    seed_everything(seed)
    device = _device()
    train_frame = bundle.frames["train"]
    prior = build_training_spatial_prior(
        [Path(path) for path in train_frame["ResolvedRoiPath"]], int(config.get("mask_threshold", 127))
    )
    capacity_audit: list[dict[str, float | int | bool]] = []
    if experiment == "UNET_MTL_MASK_GUIDED":
        reference_talon = build_talon(config.section("model"), prior, "TALON_FULL")
        model, capacity_audit = build_capacity_matched_baseline(config.section("model"), reference_talon)
        del reference_talon
        talon = False
        phases = [{"name": "standard", "max_epochs": config.section("training")["baseline_epoch_budget"], "patience": config.section("training")["baseline_patience"], "lr_scale": 1.0}]
    else:
        model = build_talon(config.section("model"), prior, experiment)
        talon = True
        if ABLATION_VARIANTS[experiment].sequential_teacher:
            phases = config.section("training")["teacher_phases"]
        else:
            maximum = sum(int(item["max_epochs"]) for item in config.section("training")["teacher_phases"])
            phases = [{"name": "standard", "max_epochs": maximum, "patience": config.section("training")["baseline_patience"], "lr_scale": 1.0}]
    model.to(device)
    class_weights, positive_weight = training_weights(train_frame, config)
    disabled_by_experiment = {
        "TALON_NO_DOCTOR_DESCRIPTOR": {"descriptor", "centroid"},
        "TALON_NO_SPATIAL_PRIOR": {"low_prior"},
    }
    disabled_auxiliary = disabled_by_experiment.get(experiment, set())
    criterion = MultiTaskCriterion(config.section("loss"), class_weights, positive_weight, disabled_auxiliary).to(device)
    optimizer_cfg = config.section("optimizer")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(optimizer_cfg["amp"]) and device.type == "cuda")
    run_dir = config.run_dir(experiment, seed)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if capacity_audit:
        pd.DataFrame(capacity_audit).to_csv(run_dir / "baseline_capacity_matching.csv", index=False)
    history: list[dict[str, Any]] = []
    exposure_records: list[pd.DataFrame] = []
    phase_records: list[dict[str, Any]] = []
    best_common_score = -np.inf
    best_global_path = checkpoint_dir / "best_overall.pt"
    best_common_path = checkpoint_dir / "best_common_validation.pt"
    resume_path = checkpoint_dir / "resume_latest.pt"
    talon_checkpoint_scores = {name: -np.inf for name in ("joint", "dice", "classification", "recall_safe", "dual_strong", "final_polish")}
    talon_checkpoint_paths = {name: checkpoint_dir / f"best_{name}.pt" for name in talon_checkpoint_scores}
    global_epoch = 0
    previous_phase_path: Path | None = None
    resume_state: dict[str, Any] | None = None
    resume_phase_index = 1
    resume_phase_epoch = 1
    if resume_path.exists():
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_state.get("experiment") != experiment or int(resume_state.get("seed", -1)) != int(seed):
            raise RuntimeError(f"Resume checkpoint identity mismatch: {resume_path}")
        model.load_state_dict(resume_state["model_state_dict"])
        global_epoch = int(resume_state.get("global_epoch", 0))
        best_common_score = float(resume_state.get("best_common_score", -np.inf))
        talon_checkpoint_scores.update(resume_state.get("talon_checkpoint_scores", {}))
        history = list(resume_state.get("history", []))
        exposure_records = [pd.DataFrame(rows) for rows in resume_state.get("exposure_records", [])]
        phase_records = list(resume_state.get("phase_records", []))
        resume_phase_index = int(resume_state.get("phase_index", 1))
        resume_phase_epoch = int(resume_state.get("next_phase_epoch", 1))
        if bool(resume_state.get("completed", False)) and best_global_path.exists():
            history_frame = pd.json_normalize(history, sep=".")
            return model, history_frame, best_global_path
    if talon and resume_state is None:
        gradient_audit = audit_auxiliary_gradients(model, criterion, bundle.loaders["train"], device, str(phases[0]["name"]), disabled_auxiliary)
        gradient_audit.to_csv(run_dir / "auxiliary_gradient_audit.csv", index=False)
        if not bool(gradient_audit["passes"].all()):
            raise RuntimeError("At least one enabled TALON auxiliary loss has zero gradient; see auxiliary_gradient_audit.csv")
    selection_keys = {
        "teacher1_anatomy_normal": "teacher1_stability",
        "teacher2_candidate_search": "recall_safe",
        "teacher3_doctor_ball": "doctor_ball",
        "teacher3_doctor_ball_objectness": "doctor_ball",
        "teacher4_seg_cls_coupling": "dual_strong",
        "teacher5_report_ready_polish": "final_polish",
        "standard": "common",
    }
    for phase_index, phase in enumerate(phases, start=1):
        if phase_index < resume_phase_index:
            continue
        phase_name = str(phase["name"])
        learning_rate = max(float(optimizer_cfg["minimum_phase_learning_rate"]), float(optimizer_cfg["learning_rate"]) * float(phase["lr_scale"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(optimizer_cfg["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(phase["max_epochs"]), eta_min=learning_rate * float(config.section("training")["cosine_eta_fraction"]),
        )
        continuing_phase = resume_state is not None and phase_index == resume_phase_index and resume_phase_epoch > 1
        if continuing_phase:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            phase_best = float(resume_state["phase_best"])
            phase_best_epoch = int(resume_state["phase_best_epoch"])
            stale = int(resume_state["stale"])
            score_ema = resume_state.get("score_ema")
            joint_score_ema = resume_state.get("joint_score_ema")
            first_phase_epoch = resume_phase_epoch
        else:
            phase_best = -np.inf
            phase_best_epoch = 0
            stale = 0
            score_ema: float | None = None
            joint_score_ema: float | None = None
            first_phase_epoch = 1
        phase_path = checkpoint_dir / f"phase_{phase_index:02d}_{phase_name}_best.pt"
        selection_key = selection_keys[phase_name]
        phase_epoch = first_phase_epoch - 1
        for phase_epoch in range(first_phase_epoch, int(phase["max_epochs"]) + 1):
            global_epoch += 1
            sampler = getattr(bundle.loaders["train"], "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(global_epoch)
            train_summary = run_epoch(model, bundle.loaders["train"], criterion, optimizer, scaler, device, phase_name, talon, global_epoch, float(optimizer_cfg["gradient_clip_norm"]), bool(optimizer_cfg["amp"]))
            validation_summary = run_epoch(model, bundle.loaders["validation"], criterion, None, scaler, device, phase_name, talon, global_epoch, float(optimizer_cfg["gradient_clip_norm"]), bool(optimizer_cfg["amp"]))
            common_score = common_checkpoint_score(validation_summary, config.section("training")["common_checkpoint_weights"])
            phase_scores = talon_phase_scores(validation_summary) if talon else {}
            raw_score = common_score if selection_key == "common" else phase_scores[selection_key]
            dice_gap = max(0.0, train_summary.dice - validation_summary.dice)
            classification_gap = max(0.0, train_summary.balanced_accuracy - validation_summary.balanced_accuracy)
            recall_penalty = max(0.0, float(config.section("training")["recall_floor"]) - validation_summary.lesion_recall)
            raw_score -= 0.12 * dice_gap + 0.08 * classification_gap + float(config.section("training")["recall_floor_penalty"]) * recall_penalty
            decay = float(config.section("training")["ema_decay"])
            score_ema = raw_score if score_ema is None else decay * score_ema + (1 - decay) * raw_score
            score = score_ema
            if talon:
                joint_raw = phase_scores["joint_score"]
                joint_score_ema = joint_raw if joint_score_ema is None else decay * joint_score_ema + (1 - decay) * joint_raw
            validation_summary.checkpoint_score = score
            validation_summary.loss_components["common_checkpoint_score"] = common_score
            validation_summary.loss_components["phase_selection_score"] = score
            history.extend([asdict(train_summary), asdict(validation_summary)])
            sampler = getattr(bundle.loaders["train"], "batch_sampler", None)
            if hasattr(sampler, "exposure_summary"):
                exposure = sampler.exposure_summary()
                exposure.insert(1, "experiment", experiment); exposure.insert(2, "phase", phase_name)
                exposure_records.append(exposure)
            improved = score >= phase_best + float(config.section("training")["minimum_checkpoint_improvement"])
            if improved:
                phase_best, phase_best_epoch, stale = score, phase_epoch, 0
                _atomic_torch_save(_checkpoint_payload(model, optimizer, validation_summary, experiment, seed), phase_path)
            else:
                stale += 1
            if common_score > best_common_score + float(config.section("training")["minimum_checkpoint_improvement"]):
                best_common_score = common_score
                payload = _checkpoint_payload(model, optimizer, validation_summary, experiment, seed)
                _atomic_torch_save(payload, best_common_path)
                if not talon:
                    _atomic_torch_save(payload, best_global_path)
            if talon:
                checkpoint_values = {
                    "joint": float(joint_score_ema),
                    "dice": validation_summary.dice,
                    "classification": validation_summary.balanced_accuracy,
                    "recall_safe": phase_scores["recall_safe"],
                    "dual_strong": phase_scores["dual_strong"],
                    "final_polish": phase_scores["final_polish"],
                }
                for checkpoint_name, checkpoint_value in checkpoint_values.items():
                    required_gain = float(config.section("training")["minimum_checkpoint_improvement"]) if checkpoint_name == "joint" else 0.0
                    if checkpoint_value > talon_checkpoint_scores[checkpoint_name] + required_gain:
                        talon_checkpoint_scores[checkpoint_name] = checkpoint_value
                        _atomic_torch_save(_checkpoint_payload(model, optimizer, validation_summary, experiment, seed), talon_checkpoint_paths[checkpoint_name])
            scheduler.step()
            _atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "experiment": experiment,
                "seed": seed,
                "global_epoch": global_epoch,
                "phase_index": phase_index,
                "next_phase_epoch": phase_epoch + 1,
                "phase_best": phase_best,
                "phase_best_epoch": phase_best_epoch,
                "stale": stale,
                "score_ema": score_ema,
                "joint_score_ema": joint_score_ema,
                "best_common_score": best_common_score,
                "talon_checkpoint_scores": talon_checkpoint_scores,
                "history": history,
                "exposure_records": [frame.to_dict(orient="records") for frame in exposure_records],
                "phase_records": phase_records,
                "completed": False,
            }, resume_path)
            if stale >= int(phase["patience"]):
                break
            if dice_gap > float(config.section("training")["overfit_dice_gap_stop"]) and stale >= max(3, int(phase["patience"]) // 2):
                break
        if phase_path.exists():
            payload = torch.load(phase_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            previous_phase_path = phase_path
        phase_records.append({
            "phase_index": phase_index, "phase": phase_name, "selection_key": selection_key,
            "max_epochs": int(phase["max_epochs"]), "patience": int(phase["patience"]),
            "completed_epochs": phase_epoch, "best_epoch": phase_best_epoch,
            "best_phase_score": phase_best, "checkpoint": str(phase_path),
        })
        if phase_index < len(phases):
            _atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "experiment": experiment,
                "seed": seed,
                "global_epoch": global_epoch,
                "phase_index": phase_index + 1,
                "next_phase_epoch": 1,
                "best_common_score": best_common_score,
                "talon_checkpoint_scores": talon_checkpoint_scores,
                "history": history,
                "exposure_records": [frame.to_dict(orient="records") for frame in exposure_records],
                "phase_records": phase_records,
                "completed": False,
            }, resume_path)
    selected_final_path = talon_checkpoint_paths["joint"] if talon else best_common_path
    if selected_final_path.exists():
        final_payload = torch.load(selected_final_path, map_location=device, weights_only=False)
        model.load_state_dict(final_payload["model_state_dict"])
        final_payload["source_checkpoint"] = str(selected_final_path)
        _atomic_torch_save(final_payload, best_global_path)
    history_frame = pd.json_normalize(history, sep=".")
    history_dir = run_dir / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(history_dir / "training_history.csv", index=False)
    pd.DataFrame(phase_records).to_csv(history_dir / "phase_summary.csv", index=False)
    if talon:
        pd.DataFrame([{"checkpoint_type": name, "best_validation_score": score, "path": str(talon_checkpoint_paths[name])} for name, score in talon_checkpoint_scores.items()]).to_csv(history_dir / "global_checkpoint_summary.csv", index=False)
    if exposure_records:
        pd.concat(exposure_records, ignore_index=True).to_csv(history_dir / "epoch_exposure_by_subject_class_roi.csv", index=False)
    _atomic_torch_save({
        "model_state_dict": model.state_dict(),
        "experiment": experiment,
        "seed": seed,
        "global_epoch": global_epoch,
        "phase_index": len(phases) + 1,
        "next_phase_epoch": 1,
        "best_common_score": best_common_score,
        "talon_checkpoint_scores": talon_checkpoint_scores,
        "history": history,
        "exposure_records": [frame.to_dict(orient="records") for frame in exposure_records],
        "phase_records": phase_records,
        "completed": True,
    }, resume_path)
    return model, history_frame, best_global_path
