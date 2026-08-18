"""Common segmentation/classification losses and TALON auxiliary objectives."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models.common import BodyFieldEstimator, mask_descriptors


def focal_bce_with_logits(logits: Tensor, target: Tensor, positive_weight: Tensor, gamma: float) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * target + (1 - probability) * (1 - target)
    return ((1 - pt).pow(gamma) * raw).mean()


def tversky_loss(logits: Tensor, target: Tensor, alpha: float, beta: float) -> Tensor:
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    true_positive = (probability * target).sum(dims)
    false_positive = (probability * (1 - target)).sum(dims)
    false_negative = ((1 - probability) * target).sum(dims)
    score = (true_positive + 1e-6) / (true_positive + alpha * false_positive + beta * false_negative + 1e-6)
    return 1 - score.mean()


def boundary_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    sobel_x = logits.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    pred_edge = torch.sqrt(F.conv2d(probability, sobel_x, padding=1).square() + F.conv2d(probability, sobel_y, padding=1).square() + 1e-6)
    true_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1).square() + F.conv2d(target, sobel_y, padding=1).square() + 1e-6)
    return F.smooth_l1_loss(pred_edge, true_edge)


def soft_dice(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def common_segmentation_loss(logits: Tensor, target: Tensor, config: Mapping[str, float], positive_weight: Tensor) -> dict[str, Tensor]:
    focal = focal_bce_with_logits(logits, target, positive_weight, float(config["focal_gamma"]))
    tversky = tversky_loss(logits, target, float(config["tversky_alpha"]), float(config["tversky_beta"]))
    boundary = boundary_loss(logits, target)
    total = float(config["focal_bce"]) * focal + float(config["tversky"]) * tversky + float(config["boundary"]) * boundary
    return {"segmentation": total, "focal_bce": focal, "tversky": tversky, "boundary": boundary}


def phase_activation(phase_name: str) -> dict[str, float]:
    """Original TALON sequential-teacher loss multipliers."""
    phases = {
        "teacher1_anatomy_normal": {"segmentation": 0.78, "classification": 0.14, "focus": 0.22, "descriptor": 0.03, "candidate_recall": 0.12, "doctor_ball": 0.00, "segmentation_classification_coupling": 0.02, "location": 0.01, "low_prior": 0.00, "outside_body": 0.04, "centroid": 0.08},
        "teacher2_candidate_search": {"segmentation": 1.00, "classification": 0.24, "focus": 0.14, "descriptor": 0.08, "candidate_recall": 0.45, "doctor_ball": 0.08, "segmentation_classification_coupling": 0.04, "location": 0.00, "low_prior": 0.00, "outside_body": 0.02, "centroid": 0.18},
        "teacher3_doctor_ball_objectness": {"segmentation": 1.00, "classification": 0.32, "focus": 0.12, "descriptor": 0.10, "candidate_recall": 0.28, "doctor_ball": 0.55, "segmentation_classification_coupling": 0.08, "location": 0.00, "low_prior": 0.00, "outside_body": 0.025, "centroid": 0.22},
        "teacher4_seg_cls_coupling": {"segmentation": 1.00, "classification": 0.52, "focus": 0.08, "descriptor": 0.08, "candidate_recall": 0.18, "doctor_ball": 0.35, "segmentation_classification_coupling": 0.35, "location": 0.005, "low_prior": 0.01, "outside_body": 0.03, "centroid": 0.20},
        "teacher5_report_ready_polish": {"segmentation": 1.00, "classification": 0.62, "focus": 0.06, "descriptor": 0.06, "candidate_recall": 0.12, "doctor_ball": 0.22, "segmentation_classification_coupling": 0.28, "location": 0.005, "low_prior": 0.01, "outside_body": 0.03, "centroid": 0.15},
        "standard": {"segmentation": 1.00, "classification": 1.00, "location": 1, "focus": 1, "centroid": 1, "low_prior": 1, "outside_body": 1, "candidate_recall": 1, "doctor_ball": 1, "descriptor": 1, "segmentation_classification_coupling": 1},
    }
    return phases.get(phase_name, phases["teacher5_report_ready_polish"])


class MultiTaskCriterion(nn.Module):
    def __init__(self, loss_config: Mapping[str, object], class_weights: Tensor, positive_weight: Tensor, disabled_auxiliary: set[str] | None = None) -> None:
        super().__init__()
        self.config = loss_config
        self.register_buffer("class_weights", class_weights.float())
        self.register_buffer("positive_weight", positive_weight.float().reshape(1))
        self.body_estimator = BodyFieldEstimator()
        self.disabled_auxiliary = disabled_auxiliary or set()

    def forward(self, outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor], phase_name: str, talon: bool) -> dict[str, Tensor]:
        target = batch["mask"]
        seg_cfg = self.config["segmentation"]
        losses = common_segmentation_loss(outputs["lesion_logits"], target, seg_cfg, self.positive_weight)
        classification = F.cross_entropy(
            outputs["cls_logits"], batch["class_id"], weight=self.class_weights,
            label_smoothing=float(self.config["classification"]["label_smoothing"]),
        )
        losses["classification"] = classification
        total = losses["segmentation"] + float(self.config["classification"]["weight"]) * classification
        if not talon:
            losses["total"] = total
            return losses

        activation = phase_activation(phase_name)
        total = float(activation["segmentation"]) * losses["segmentation"] + float(self.config["classification"]["weight"]) * float(activation["classification"]) * classification
        probability = torch.sigmoid(outputs["lesion_logits"])
        location = F.binary_cross_entropy_with_logits(outputs["location_logits"], target)
        focus_target = F.max_pool2d(target, kernel_size=15, stride=1, padding=7)
        focus = F.binary_cross_entropy_with_logits(outputs["focus_logits"], focus_target)
        true_descriptors = mask_descriptors(target)
        descriptor = F.smooth_l1_loss(outputs["descriptors"], true_descriptors)
        centroid = F.smooth_l1_loss(outputs["descriptors"][:, 1:3], true_descriptors[:, 1:3])
        expanded_prior = outputs["dataset_prior"].expand_as(probability)
        low_prior = (probability * (1 - expanded_prior).pow(2)).mean()
        body = self.body_estimator(batch["image"])
        outside_body = (probability * (1 - body)).mean()
        candidate_recall = ((1 - probability) * target).sum() / target.sum().clamp_min(1)
        doctor_ball_target = F.max_pool2d(target, kernel_size=31, stride=1, padding=15)
        doctor_ball = (probability * (1 - doctor_ball_target)).mean()
        per_sample_dice = (2 * (probability * target).sum((1, 2, 3)) + 1e-6) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)
        correct_probability = torch.softmax(outputs["cls_logits"], dim=1).gather(1, batch["class_id"][:, None]).squeeze(1)
        coupling = F.smooth_l1_loss(correct_probability, per_sample_dice.detach())
        auxiliary = {
            "location": location, "focus": focus, "descriptor": descriptor, "centroid": centroid,
            "low_prior": low_prior, "outside_body": outside_body, "candidate_recall": candidate_recall,
            "doctor_ball": doctor_ball, "segmentation_classification_coupling": coupling,
        }
        for name, value in auxiliary.items():
            losses[name] = value
            enabled = 0.0 if name in self.disabled_auxiliary else float(activation.get(name, 0.0))
            total = total + float(self.config[name]) * enabled * value
        losses["total"] = total
        return losses


def estimate_training_weights(class_counts: Mapping[int, int], foreground_pixels: int, background_pixels: int, cap: float) -> tuple[Tensor, Tensor]:
    counts = torch.tensor([class_counts.get(0, 0), class_counts.get(1, 0)], dtype=torch.float32).clamp_min(1)
    class_weights = counts.sum() / (2 * counts)
    positive_weight = torch.tensor(min(background_pixels / max(foreground_pixels, 1), cap), dtype=torch.float32)
    return class_weights, positive_weight
