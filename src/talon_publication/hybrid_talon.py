"""Hybrid TALON architecture: revised TALON segmentation + legacy classifier.

The segmentation path is a literal module-level port of the executed
``TALON_FULL`` graph in ``02_Models_and_Losses.ipynb``.  The classification
path restores the archived multi-evidence ``DoctorViewClassifier`` including
global, lesion, perilesional, body/location, descriptor, and mask-quality
evidence.  This module defines models only; it contains no training entrypoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Revised TALON segmentation path (kept structurally identical)
# ---------------------------------------------------------------------------


class FixedTextureMaps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None])
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)[None, None])
        self.register_buffer("laplace", torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None])

    def forward(self, x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
        variance = (F.avg_pool2d(gray.square(), 5, stride=1, padding=2) - mean.square()).clamp_min(0)
        return torch.cat([x, grad, lap, variance], dim=1)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ConvBNAct(in_channels, out_channels), ConvBNAct(out_channels, out_channels))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.body(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VHAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 8)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.vertical = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden)
        self.horizontal = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        attention = torch.sigmoid(self.expand(self.horizontal(self.vertical(F.silu(self.reduce(x))))))
        return x * (1.0 + attention)


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: Sequence[int] = (1, 3, 6, 9)) -> None:
        super().__init__()
        branch_channels = max(channels // len(rates), 16)
        self.branches = nn.ModuleList([ConvBNAct(channels, branch_channels, 1 if rate == 1 else 3, rate) for rate in rates])
        self.project = ConvBNAct(branch_channels * len(rates), channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class AttentionNeck(nn.Module):
    def __init__(self, channels: int, use_vh: bool = True, use_aspp: bool = True) -> None:
        super().__init__()
        self.vh = VHAttention(channels) if use_vh else nn.Identity()
        self.aspp = ASPP(channels) if use_aspp else nn.Identity()
        self.fuse = ResidualBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(self.aspp(self.vh(x)))


class CoordDistanceMaps(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        yy = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square()).clamp_max(math.sqrt(2)) / math.sqrt(2)
        return torch.stack([grid_x, grid_y, radius], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class BodyFieldEstimator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        minimum = gray.amin(dim=(2, 3), keepdim=True)
        maximum = gray.amax(dim=(2, 3), keepdim=True)
        normalized = (gray - minimum) / (maximum - minimum + 1e-6)
        return torch.sigmoid((normalized - 0.12) * 12.0)


class TalonRichInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.texture = FixedTextureMaps()
        self.coords = CoordDistanceMaps()
        self.body = BodyFieldEstimator()
        self.learned_location = nn.Sequential(ConvBNAct(10, 16), nn.Conv2d(16, 1, 1))

    def forward(self, image: Tensor, train_prior: Tensor) -> tuple[Tensor, Tensor]:
        image_texture = self.texture(image)
        base = torch.cat([image_texture, self.coords(image), self.body(image)], dim=1)
        location = self.learned_location(base)
        prior = F.interpolate(train_prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[0] == 1 and image.shape[0] > 1:
            prior = prior.expand(image.shape[0], -1, -1, -1)
        enriched = torch.cat([base, torch.sigmoid(location), prior, 1.0 - prior], dim=1)
        if enriched.shape[1] != 13:
            raise RuntimeError(f"TALON enriched input must have 13 channels, observed {enriched.shape[1]}")
        return enriched, location


class AnatomicalLocationPriorHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.coords = CoordDistanceMaps()
        self.body = BodyFieldEstimator()
        self.net = nn.Sequential(ConvBNAct(in_channels + 4, 24), nn.Conv2d(24, 1, 1))

    def forward(self, features: Tensor, image: Tensor) -> Tensor:
        image_small = F.interpolate(image, size=features.shape[-2:], mode="bilinear", align_corners=False)
        fields = torch.cat([self.coords(features), self.body(image_small)], dim=1)
        return self.net(torch.cat([features, fields], dim=1))


class DatasetAdaptiveSpatialPrior(nn.Module):
    def __init__(self, prior: Tensor | None = None) -> None:
        super().__init__()
        base = torch.ones(1, 1, 32, 32) if prior is None else prior.float()
        self.register_buffer("prior", base / base.amax().clamp_min(1e-6))

    def forward(self, size: tuple[int, int]) -> Tensor:
        return F.interpolate(self.prior, size=size, mode="bilinear", align_corners=False)


class FocusGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(channels + 2, channels), nn.Conv2d(channels, 1, 1))

    def forward(self, features: Tensor, location: Tensor, dataset_prior: Tensor) -> Tensor:
        if dataset_prior.shape[0] == 1 and features.shape[0] > 1:
            dataset_prior = dataset_prior.expand(features.shape[0], -1, -1, -1)
        return self.net(torch.cat([features, torch.sigmoid(location), dataset_prior], dim=1))


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up3 = ResidualBlock(c4 + c3, c3)
        self.up2 = ResidualBlock(c3 + c2, c2)
        self.up1 = ResidualBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, bottleneck: Tensor, skips: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        s1, s2, s3 = skips
        x = F.interpolate(bottleneck, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, s3], dim=1))
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, s1], dim=1))
        return self.head(x), x


class PatchNINRefiner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, decoder_features: Tensor, coarse_logits: Tensor) -> Tensor:
        return coarse_logits + self.net(torch.cat([decoder_features, coarse_logits], dim=1))


def mask_descriptors(mask: Tensor) -> Tensor:
    """The revised TALON descriptor retained for its existing auxiliary loss."""
    if mask.ndim == 3:
        mask = mask[:, None]
    mask = mask.float()
    _, _, height, width = mask.shape
    yy = torch.linspace(-1, 1, height, device=mask.device, dtype=mask.dtype)[None, None, :, None]
    xx = torch.linspace(-1, 1, width, device=mask.device, dtype=mask.dtype)[None, None, None, :]
    mass = mask.sum(dim=(2, 3)).clamp_min(1e-6)
    cx = (mask * xx).sum(dim=(2, 3)) / mass
    cy = (mask * yy).sum(dim=(2, 3)) / mass
    var_x = (mask * (xx - cx[:, :, None, None]).square()).sum(dim=(2, 3)) / mass
    var_y = (mask * (yy - cy[:, :, None, None]).square()).sum(dim=(2, 3)) / mass
    cov = (mask * (xx - cx[:, :, None, None]) * (yy - cy[:, :, None, None])).sum(dim=(2, 3)) / mass
    area = mass / float(height * width)
    max_prob = mask.amax(dim=(2, 3))
    mean_prob = mask.mean(dim=(2, 3))
    perimeter = (mask[:, :, 1:] - mask[:, :, :-1]).abs().mean(dim=(2, 3)) + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean(dim=(2, 3))
    compactness = area / (perimeter.square() + 1e-6)
    elongation = torch.maximum(var_x, var_y) / (torch.minimum(var_x, var_y) + 1e-6)
    normalized_size = mass.sqrt() / math.sqrt(height * width)
    return torch.cat([area, cx, cy, var_x, var_y, cov, max_prob, mean_prob, perimeter, compactness, elongation, normalized_size], dim=1)


class NewTALONSegmentation(nn.Module):
    """Exact revised TALON segmentation graph, separated from its classifier."""

    def __init__(self, base_channels: int, prior: Tensor) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.channels = tuple(channels)
        self.rich_input = TalonRichInput()
        self.stem = ResidualBlock(13, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.neck = AttentionNeck(channels[3], use_vh=True, use_aspp=True)
        self.location = AnatomicalLocationPriorHead(channels[3])
        self.dataset_prior = DatasetAdaptiveSpatialPrior(prior)
        self.focus = FocusGate(channels[3])
        self.decoder = SegmentationDecoder(channels)
        self.refiner = PatchNINRefiner(channels[0])

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
        lesion_logits = self.refiner(decoder_features, coarse_logits)
        return {
            "lesion_logits": lesion_logits,
            "coarse_logits": coarse_logits,
            "bottleneck": bottleneck,
            "location_logits": F.interpolate(location_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "image_location_logits": F.interpolate(image_location_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "focus_logits": F.interpolate(focus_logits, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
            "dataset_prior": F.interpolate(dataset_prior, size=lesion_logits.shape[-2:], mode="bilinear", align_corners=False),
        }


# ---------------------------------------------------------------------------
# Archived TALON classification path (literal functional restoration)
# ---------------------------------------------------------------------------


class LegacyConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None, groups: int = 1) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.block(input_tensor)


class LegacyFixedTextureMaps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None])
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)[None, None])
        self.register_buffer("laplacian", torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None])

    def forward(self, image_tensor: Tensor) -> Tensor:
        gray = image_tensor.mean(dim=1, keepdim=True)
        gradient_x = F.conv2d(gray, self.sobel_x, padding=1)
        gradient_y = F.conv2d(gray, self.sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2) + 1e-6)
        laplacian_abs = torch.abs(F.conv2d(gray, self.laplacian, padding=1))
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_mean_squared = F.avg_pool2d(gray.pow(2), kernel_size=9, stride=1, padding=4)
        local_variance = torch.clamp(local_mean_squared - local_mean.pow(2), min=0.0)
        return torch.cat([gradient_magnitude, laplacian_abs, local_variance], dim=1)


class LegacyBodyFieldEstimator(nn.Module):
    def __init__(self, threshold: float = 0.035, temperature: float = 0.018, smooth_kernel: int = 17) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.smooth_kernel = int(smooth_kernel)

    def forward(self, input_tensor: Tensor) -> Tensor:
        gray = input_tensor.mean(dim=1, keepdim=True)
        body_field = torch.sigmoid((gray - self.threshold) / max(self.temperature, 1e-6))
        if self.smooth_kernel > 1:
            kernel_size = self.smooth_kernel + (self.smooth_kernel % 2 == 0)
            body_field = F.avg_pool2d(body_field, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return torch.clamp(body_field, 0.0, 1.0)


def soft_weighted_mean(feature_map: Tensor, weight_map: Tensor, epsilon: float = 1e-6) -> Tensor:
    return (feature_map * weight_map).sum(dim=(2, 3)) / (weight_map.sum(dim=(2, 3)) + epsilon)


def soft_weighted_std(feature_map: Tensor, weight_map: Tensor, epsilon: float = 1e-6) -> Tensor:
    weighted_mean = soft_weighted_mean(feature_map, weight_map, epsilon).view(feature_map.shape[0], feature_map.shape[1], 1, 1)
    variance = ((feature_map - weighted_mean).pow(2) * weight_map).sum(dim=(2, 3)) / (weight_map.sum(dim=(2, 3)) + epsilon)
    return torch.sqrt(torch.clamp(variance, min=0.0) + epsilon)


def legacy_lesion_descriptors(image: Tensor, lesion_prob: Tensor, texture_maps: Tensor | None = None, epsilon: float = 1e-6) -> Tensor:
    batch_size, _, image_height, image_width = lesion_prob.shape
    probability = torch.clamp(lesion_prob, 0.0, 1.0)
    gray = image.mean(dim=1, keepdim=True)
    context_ring = torch.clamp(F.max_pool2d(probability, 25, stride=1, padding=12) - probability, 0.0, 1.0)
    area_fraction = probability.mean(dim=(2, 3))
    x_coords = torch.linspace(0, 1, image_width, device=image.device, dtype=probability.dtype).view(1, 1, 1, image_width)
    y_coords = torch.linspace(0, 1, image_height, device=image.device, dtype=probability.dtype).view(1, 1, image_height, 1)
    probability_sum = probability.sum(dim=(2, 3), keepdim=True) + epsilon
    centroid_x = (probability * x_coords).sum(dim=(2, 3), keepdim=True) / probability_sum
    centroid_y = (probability * y_coords).sum(dim=(2, 3), keepdim=True) / probability_sum
    spread_x = torch.sqrt(((x_coords - centroid_x).pow(2) * probability).sum(dim=(2, 3), keepdim=True) / probability_sum + epsilon).view(batch_size, 1)
    spread_y = torch.sqrt(((y_coords - centroid_y).pow(2) * probability).sum(dim=(2, 3), keepdim=True) / probability_sum + epsilon).view(batch_size, 1)
    lesion_mean = soft_weighted_mean(gray, probability)
    lesion_std = soft_weighted_std(gray, probability)
    context_mean = soft_weighted_mean(gray, context_ring + epsilon)
    perimeter_x = torch.abs(probability[:, :, :, 1:] - probability[:, :, :, :-1]).sum(dim=(2, 3)) / (image_height * image_width)
    perimeter_y = torch.abs(probability[:, :, 1:, :] - probability[:, :, :-1, :]).sum(dim=(2, 3)) / (image_height * image_width)
    perimeter = perimeter_x + perimeter_y
    compactness = area_fraction / (area_fraction + perimeter.pow(2) + epsilon)
    texture_mean = torch.zeros_like(area_fraction) if texture_maps is None else soft_weighted_mean(texture_maps.mean(dim=1, keepdim=True), probability)
    descriptors = torch.cat([
        area_fraction,
        centroid_x.view(batch_size, 1),
        centroid_y.view(batch_size, 1),
        spread_x,
        spread_y,
        lesion_mean,
        lesion_std,
        context_mean,
        lesion_mean - context_mean,
        perimeter,
        compactness,
        texture_mean,
    ], dim=1)
    return torch.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)


def legacy_weighted_avg_pool(feature_map: Tensor, weight_map: Tensor, epsilon: float = 1e-6) -> Tensor:
    weights = torch.clamp(weight_map, 0.0, 1.0)
    return (feature_map * weights).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + epsilon)


class LegacyDoctorViewClassifier(nn.Module):
    """Archived multi-evidence classifier, parameterized by bottleneck width."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        descriptor_dim: int = 12,
        descriptor_embed_dim: int = 64,
        dropout: float = 0.50,
        descriptor_dropout: float = 0.30,
        global_scale: float = 0.08,
        mask_quality_dim: int = 5,
        mask_quality_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.global_scale = float(global_scale)
        global_feature_dim = max(in_channels // 2, 64)
        self.projection = LegacyConvBNAct(in_channels, in_channels, 1, 1, 0)
        self.global_projection = nn.Sequential(nn.Linear(in_channels, global_feature_dim), nn.LayerNorm(global_feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.lesion_projection = nn.Sequential(nn.Linear(in_channels, in_channels), nn.LayerNorm(in_channels), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.context_projection = nn.Sequential(nn.Linear(in_channels, in_channels), nn.LayerNorm(in_channels), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.body_location_projection = nn.Sequential(nn.Linear(in_channels, in_channels // 2), nn.LayerNorm(in_channels // 2), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.descriptor_normalizer = nn.LayerNorm(descriptor_dim)
        self.descriptor_dropout_layer = nn.Dropout(descriptor_dropout)
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_embed_dim), nn.LayerNorm(descriptor_embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(descriptor_embed_dim, descriptor_embed_dim), nn.GELU(),
        )
        self.mask_quality_mlp = nn.Sequential(nn.LayerNorm(mask_quality_dim), nn.Linear(mask_quality_dim, mask_quality_embed_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        fusion_dim = global_feature_dim + 2 * in_channels + in_channels // 2 + descriptor_embed_dim + mask_quality_embed_dim
        self.classifier_head = nn.Sequential(
            nn.Linear(fusion_dim, in_channels), nn.LayerNorm(in_channels), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_channels, num_classes)
        )

    def forward(
        self,
        features: Tensor,
        lesion_prob: Tensor,
        focus_map: Tensor,
        descriptors: Tensor,
        body_field: Tensor,
        location_prior: Tensor,
        train_spatial_prior: Tensor,
        low_prior_mask: Tensor,
        mask_quality: Tensor,
    ) -> Tensor:
        features = self.projection(features)
        output_size = features.shape[-2:]
        lesion_prob = F.interpolate(lesion_prob, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        _ = F.interpolate(focus_map, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        body = F.interpolate(body_field, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        location = F.interpolate(location_prior, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        prior = F.interpolate(train_spatial_prior, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        low_prior = F.interpolate(low_prior_mask, size=output_size, mode="bilinear", align_corners=False).clamp(0, 1)
        context = torch.clamp(F.max_pool2d(lesion_prob, 17, stride=1, padding=8) - lesion_prob, 0.0, 1.0)
        global_features = self.global_projection(legacy_weighted_avg_pool(features, torch.clamp(0.85 * body + 0.15 * prior, 0.0, 1.0))) * self.global_scale
        lesion_features = self.lesion_projection(legacy_weighted_avg_pool(features, lesion_prob))
        context_features = self.context_projection(legacy_weighted_avg_pool(features, context))
        body_location = self.body_location_projection(legacy_weighted_avg_pool(features, torch.clamp(0.55 * body + 0.20 * location + 0.25 * prior, 0.0, 1.0)))
        descriptor_features = self.descriptor_mlp(self.descriptor_dropout_layer(self.descriptor_normalizer(descriptors)))
        quality_features = self.mask_quality_mlp(mask_quality)
        return self.classifier_head(torch.cat([
            global_features, lesion_features, context_features, body_location, descriptor_features, quality_features
        ], dim=1))


class LegacyClassificationEvidence(nn.Module):
    def __init__(self, low_prior_gamma: float = 2.0) -> None:
        super().__init__()
        self.texture = LegacyFixedTextureMaps()
        self.body = LegacyBodyFieldEstimator()
        self.low_prior_gamma = float(low_prior_gamma)

    def forward(
        self,
        image: Tensor,
        lesion_probability: Tensor,
        train_prior: Tensor,
        segmentation_location_logits: Tensor,
    ) -> dict[str, Tensor]:
        texture = self.texture(image)
        body = self.body(image)
        # Location guidance belongs to the locked revised segmentation path.
        # The archived classifier consumes that guidance without introducing a
        # second, independently trainable location head.
        location_prior = torch.sigmoid(F.interpolate(
            segmentation_location_logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        ))
        prior = F.interpolate(train_prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[0] == 1 and image.shape[0] > 1:
            prior = prior.expand(image.shape[0], -1, -1, -1)
        low_prior = torch.clamp(body * (1.0 - prior).pow(self.low_prior_gamma), 0.0, 1.0)
        guidance = lesion_probability.detach()
        descriptors = legacy_lesion_descriptors(image, guidance, texture.detach())
        guidance_sum = guidance.sum(dim=(2, 3)) + 1e-6
        mask_quality = torch.cat([
            guidance.mean(dim=(2, 3)),
            (guidance * (1.0 - body.detach())).sum(dim=(2, 3)) / guidance_sum,
            (guidance * low_prior.detach()).sum(dim=(2, 3)) / guidance_sum,
            (guidance * body.detach()).sum(dim=(2, 3)) / guidance_sum,
            (guidance * prior.detach()).sum(dim=(2, 3)) / guidance_sum,
        ], dim=1)
        return {
            "texture_maps": texture,
            "body_field": body,
            "location_prior": location_prior,
            "train_spatial_prior": prior,
            "low_prior_mask": low_prior,
            "legacy_descriptors": descriptors,
            "mask_quality": mask_quality,
            "lesion_guidance": guidance,
        }


# ---------------------------------------------------------------------------
# Reference revised model and requested hybrid model
# ---------------------------------------------------------------------------


class NewDoctorViewClassifier(nn.Module):
    def __init__(self, feature_channels: int, descriptor_count: int = 12, classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_channels * 2 + descriptor_count, 256), nn.SiLU(), nn.Dropout(0.30),
            nn.Linear(256, 96), nn.SiLU(), nn.Dropout(0.15), nn.Linear(96, classes),
        )

    def forward(self, features: Tensor, lesion_probability: Tensor, descriptors: Tensor) -> Tensor:
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        weight = F.interpolate(lesion_probability, size=features.shape[-2:], mode="bilinear", align_corners=False)
        lesion_features = (features * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)
        return self.head(torch.cat([global_features, lesion_features, descriptors], dim=1))


class NewTALONReference(nn.Module):
    def __init__(self, base_channels: int, prior: Tensor) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.segmentation = NewTALONSegmentation(base_channels, prior)
        self.classifier = NewDoctorViewClassifier(base_channels * 8)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        outputs = self.segmentation(image)
        probability = torch.sigmoid(outputs["lesion_logits"])
        descriptors = mask_descriptors(probability)
        with torch.autocast(device_type=image.device.type, enabled=False):
            cls_logits = self.classifier(outputs["bottleneck"].float(), probability.detach().float(), descriptors.detach().float())
        return {**outputs, "cls_logits": cls_logits, "descriptors": descriptors}


class HybridTALON(nn.Module):
    """Revised segmentation with the archived multi-evidence classifier."""

    def __init__(self, base_channels: int, prior: Tensor) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.segmentation = NewTALONSegmentation(base_channels, prior)
        self.classification_evidence = LegacyClassificationEvidence()
        self.classifier = LegacyDoctorViewClassifier(base_channels * 8)

    @property
    def neck(self) -> nn.Module:
        """Expose the revised neck without duplicate module registration for Grad-CAM."""
        return self.segmentation.neck

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        outputs = self.segmentation(image)
        probability = torch.sigmoid(outputs["lesion_logits"])
        evidence = self.classification_evidence(
            image,
            probability,
            outputs["dataset_prior"],
            outputs["image_location_logits"],
        )
        focus_map = torch.sigmoid(outputs["focus_logits"])
        with torch.autocast(device_type=image.device.type, enabled=False):
            cls_logits = self.classifier(
                outputs["bottleneck"].float(),
                evidence["lesion_guidance"].float(),
                focus_map.float(),
                evidence["legacy_descriptors"].detach().float(),
                evidence["body_field"].detach().float(),
                evidence["location_prior"].detach().float(),
                evidence["train_spatial_prior"].detach().float(),
                evidence["low_prior_mask"].detach().float(),
                evidence["mask_quality"].detach().float(),
            )
        # Keep the revised descriptor output so the already locked segmentation
        # auxiliary-loss interface remains unchanged in the later training stage.
        descriptors = mask_descriptors(probability)
        return {
            **outputs,
            **evidence,
            "cls_logits": cls_logits,
            "descriptors": descriptors,
        }


def build_new_talon_reference(model_config: Mapping[str, object], prior: Tensor) -> NewTALONReference:
    return NewTALONReference(int(model_config["base_channels"]), prior)


def build_hybrid_talon(model_config: Mapping[str, object], prior: Tensor) -> HybridTALON:
    return HybridTALON(int(model_config["base_channels"]), prior)
