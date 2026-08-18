"""Mask-guided multi-task U-Net segmentation/classification baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import DownBlock, ResidualBlock, SegmentationDecoder, weighted_average_pool


class MaskGuidedMultiTaskUNet(nn.Module):
    """Guide classification pooling with the model's own predicted soft mask.

    Ground-truth masks supervise segmentation but are never model inputs. At
    validation and test time this architecture therefore requires only the image.
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.30) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = ResidualBlock(3, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.bottleneck = ResidualBlock(channels[3], channels[3])
        self.decoder = SegmentationDecoder(channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels[3] * 2, 192), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(192, 64), nn.SiLU(), nn.Dropout(dropout / 2), nn.Linear(64, 2),
        )

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        s1 = self.stem(image)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        bottleneck = self.bottleneck(self.down3(s3))
        lesion_logits, _ = self.decoder(bottleneck, (s1, s2, s3))
        predicted_mask = torch.sigmoid(lesion_logits).detach()
        global_features = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        lesion_features = weighted_average_pool(bottleneck, predicted_mask)
        cls_logits = self.classifier(torch.cat([global_features, lesion_features], dim=1))
        return {"lesion_logits": lesion_logits, "coarse_logits": lesion_logits, "cls_logits": cls_logits}


def build_capacity_matched_baseline(
    model_config: Mapping[str, Any], reference_model: nn.Module,
) -> tuple[MaskGuidedMultiTaskUNet, list[dict[str, float | int | bool]]]:
    """Choose the U-Net width whose trainable-parameter count is closest to TALON.

    No data or forward pass is used.  The complete candidate audit is returned so
    the capacity-matching decision can be reported instead of being implicit.
    """
    target = sum(parameter.numel() for parameter in reference_model.parameters() if parameter.requires_grad)
    candidates = [int(value) for value in model_config.get("baseline_base_channel_candidates", [16, 20, 24, 28, 32, 36, 40, 48])]
    if not candidates:
        raise ValueError("model.baseline_base_channel_candidates must contain at least one width")
    dropout = float(model_config["classifier_dropout"])
    rows: list[dict[str, float | int | bool]] = []
    built: list[tuple[MaskGuidedMultiTaskUNet, int, float]] = []
    for width in sorted(set(candidates)):
        candidate = MaskGuidedMultiTaskUNet(width, dropout)
        count = sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad)
        relative_difference = abs(count - target) / max(target, 1)
        built.append((candidate, count, relative_difference))
    selected_index = min(range(len(built)), key=lambda index: built[index][2])
    for index, (_, count, difference) in enumerate(built):
        rows.append({
            "talon_trainable_parameters": int(target),
            "baseline_base_channels": int(sorted(set(candidates))[index]),
            "baseline_trainable_parameters": int(count),
            "relative_parameter_difference": float(difference),
            "selected": bool(index == selected_index),
        })
    return built[selected_index][0], rows


def baseline_width_from_checkpoint(checkpoint: Mapping[str, Any], model_config: Mapping[str, Any]) -> int:
    """Recover the capacity-matched baseline width needed to reload its weights."""
    metadata = checkpoint.get("model_metadata", {}) if isinstance(checkpoint, Mapping) else {}
    return int(metadata.get("base_channels", model_config["base_channels"]))
