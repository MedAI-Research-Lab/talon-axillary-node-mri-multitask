"""TALON-Net and controlled ablation variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..data import read_binary_mask
from .common import (
    AnatomicalLocationPriorHead, AttentionNeck, DatasetAdaptiveSpatialPrior, DoctorViewClassifier,
    DownBlock, FocusGate, PatchNINRefiner, ResidualBlock, SegmentationDecoder, TalonRichInput,
    mask_descriptors,
)


@dataclass(frozen=True)
class TalonVariant:
    spatial_prior: bool = True
    doctor_view: bool = True
    aspp: bool = True
    vh_attention: bool = True
    patch_nin: bool = True
    sequential_teacher: bool = True


ABLATION_VARIANTS: Mapping[str, TalonVariant] = {
    "TALON_FULL": TalonVariant(),
    "TALON_NO_SPATIAL_PRIOR": TalonVariant(spatial_prior=False),
    "TALON_NO_DOCTOR_DESCRIPTOR": TalonVariant(doctor_view=False),
    "TALON_NO_ASPP": TalonVariant(aspp=False),
    "TALON_NO_VH_ATTENTION": TalonVariant(vh_attention=False),
    "TALON_NO_PATCH_NIN": TalonVariant(patch_nin=False),
    "TALON_NO_SEQUENTIAL_TEACHER": TalonVariant(sequential_teacher=False),
}


def build_training_spatial_prior(mask_paths: list[Path], threshold: int, size: int = 32) -> Tensor:
    """Build a spatial prior from training masks only."""
    if not mask_paths:
        return torch.ones(1, 1, size, size)
    accumulator = np.zeros((size, size), dtype=np.float64)
    from PIL import Image
    for path in mask_paths:
        mask = read_binary_mask(path, threshold)
        resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
        accumulator += np.asarray(resized, dtype=np.float32) / 255.0
    accumulator /= len(mask_paths)
    accumulator /= max(float(accumulator.max()), 1e-6)
    return torch.from_numpy(accumulator.astype(np.float32))[None, None]


class TALONNetDoctorView(nn.Module):
    def __init__(self, base_channels: int, dropout: float, prior: Tensor, variant: TalonVariant) -> None:
        super().__init__()
        self.variant = variant
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.rich_input = TalonRichInput()
        self.stem = ResidualBlock(13, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.neck = AttentionNeck(channels[3], use_vh=variant.vh_attention, use_aspp=variant.aspp)
        self.location = AnatomicalLocationPriorHead(channels[3])
        chosen_prior = prior if variant.spatial_prior else torch.ones_like(prior)
        self.dataset_prior = DatasetAdaptiveSpatialPrior(chosen_prior)
        self.focus = FocusGate(channels[3])
        self.decoder = SegmentationDecoder(channels)
        self.refiner = PatchNINRefiner(channels[0]) if variant.patch_nin else None
        self.classifier = DoctorViewClassifier(channels[3]) if variant.doctor_view else nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[3], 2)
        )

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        image_prior = self.dataset_prior(image.shape[-2:])
        enriched_input, image_location_logits = self.rich_input(image, image_prior)
        s1 = self.stem(enriched_input)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        bottleneck = self.neck(self.down3(s3))
        location_logits = self.location(bottleneck, image)
        dataset_prior = self.dataset_prior(bottleneck.shape[-2:])
        focus_logits = self.focus(bottleneck, location_logits, dataset_prior)
        focused = bottleneck * (1.0 + torch.sigmoid(focus_logits))
        coarse_logits, decoder_features = self.decoder(focused, (s1, s2, s3))
        lesion_logits = self.refiner(decoder_features, coarse_logits) if self.refiner is not None else coarse_logits
        lesion_probability = torch.sigmoid(lesion_logits)
        descriptors = mask_descriptors(lesion_probability)
        if self.variant.doctor_view:
            # The classifier cannot reshape the mask for an easier diagnosis, while
            # descriptors stay differentiable for the explicit auxiliary loss.
            # Doctor-view descriptor reductions and their classifier run in fp32;
            # fp16 overflow here otherwise produces NaN classification loss.
            with torch.autocast(device_type=image.device.type, enabled=False):
                cls_logits = self.classifier(
                    bottleneck.float(), lesion_probability.detach().float(), descriptors.detach().float()
                )
        else:
            cls_logits = self.classifier(bottleneck)
        return {
            "lesion_logits": lesion_logits,
            "coarse_logits": coarse_logits,
            "cls_logits": cls_logits,
            "location_logits": F.interpolate(location_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "image_location_logits": F.interpolate(image_location_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "focus_logits": F.interpolate(focus_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "dataset_prior": F.interpolate(dataset_prior, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "descriptors": descriptors,
        }


def build_talon(model_config: Mapping[str, object], prior: Tensor, experiment_name: str) -> TALONNetDoctorView:
    if experiment_name not in ABLATION_VARIANTS:
        raise KeyError(f"Unknown TALON experiment: {experiment_name}")
    return TALONNetDoctorView(
        base_channels=int(model_config["base_channels"]), dropout=float(model_config["classifier_dropout"]),
        prior=prior, variant=ABLATION_VARIANTS[experiment_name],
    )
