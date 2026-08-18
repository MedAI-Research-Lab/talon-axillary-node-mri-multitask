"""Reusable TALON-Net building blocks derived from the archived notebook."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FixedTextureMaps(nn.Module):
    """Append gradient magnitude, absolute Laplacian and local variance maps."""

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
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
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
    """Construct the preserved 13-channel TALON input representation.

    Channels: RGB/grayscale triplicate (3), texture maps (3), x/y/radius (3),
    body field (1), learned image-location prior (1), train spatial prior (1),
    and its low-prior complement (1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.texture = FixedTextureMaps()
        self.coords = CoordDistanceMaps()
        self.body = BodyFieldEstimator()
        self.learned_location = nn.Sequential(
            ConvBNAct(10, 16), nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor, train_prior: Tensor) -> tuple[Tensor, Tensor]:
        image_texture = self.texture(image)
        base = torch.cat([image_texture, self.coords(image), self.body(image)], dim=1)
        location = self.learned_location(base)
        prior = F.interpolate(train_prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[0] == 1 and image.shape[0] > 1:
            prior = prior.expand(image.shape[0], -1, -1, -1)
        low_prior = 1.0 - prior
        enriched = torch.cat([base, torch.sigmoid(location), prior, low_prior], dim=1)
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
    """Training-mask-only prior stored as a non-trainable buffer."""

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
        self.net = nn.Sequential(nn.Conv2d(channels + 1, channels, 1), nn.SiLU(), nn.Conv2d(channels, channels, 1), nn.SiLU(), nn.Conv2d(channels, 1, 1))

    def forward(self, decoder_features: Tensor, coarse_logits: Tensor) -> Tensor:
        return coarse_logits + self.net(torch.cat([decoder_features, coarse_logits], dim=1))


def mask_descriptors(mask: Tensor) -> Tensor:
    """Differentiable 12-element doctor-view descriptor vector."""
    if mask.ndim == 3:
        mask = mask[:, None]
    # Reductions over 512x512 maps can exceed fp16's maximum (65,504), and the
    # compactness/elongation descriptors can also be large for an initially
    # diffuse probability map.  Descriptor arithmetic must therefore remain in
    # fp32 even when the convolutional trunk is trained with AMP.
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


def weighted_average_pool(features: Tensor, weight: Tensor) -> Tensor:
    weight = F.interpolate(weight, size=features.shape[-2:], mode="bilinear", align_corners=False)
    return (features * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)


class DoctorViewClassifier(nn.Module):
    def __init__(self, feature_channels: int, descriptor_count: int = 12, classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_channels * 2 + descriptor_count, 256), nn.SiLU(), nn.Dropout(0.30),
            nn.Linear(256, 96), nn.SiLU(), nn.Dropout(0.15), nn.Linear(96, classes),
        )

    def forward(self, features: Tensor, lesion_probability: Tensor, descriptors: Tensor) -> Tensor:
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        lesion_features = weighted_average_pool(features, lesion_probability)
        return self.head(torch.cat([global_features, lesion_features, descriptors], dim=1))
