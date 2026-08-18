"""Grad-CAM explanations and quantitative overlap checks for classification."""

from __future__ import annotations

from contextlib import AbstractContextManager

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GradCAM(AbstractContextManager):
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output

    def _backward_hook(self, _module: nn.Module, _grad_input: tuple[Tensor | None, ...], grad_output: tuple[Tensor, ...]) -> None:
        self.gradients = grad_output[0]

    def __call__(self, image: Tensor, class_index: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(image)
        logits = outputs["cls_logits"]
        if class_index is None:
            class_index = logits.argmax(dim=1)
        logits.gather(1, class_index[:, None]).sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
        minimum = heatmap.amin(dim=(2, 3), keepdim=True)
        maximum = heatmap.amax(dim=(2, 3), keepdim=True)
        return (heatmap - minimum) / (maximum - minimum + 1e-6), logits.detach()

    def __exit__(self, *args: object) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def _weighted_centroid(array: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(array, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return np.nan, np.nan
    yy, xx = np.indices(weights.shape)
    return float((yy * weights).sum() / total), float((xx * weights).sum() / total)


def xai_overlap_metrics(
    heatmap: np.ndarray,
    lesion_mask: np.ndarray,
    prediction_mask: np.ndarray | None = None,
    threshold_quantile: float = 0.80,
) -> dict[str, float]:
    selected = heatmap >= np.quantile(heatmap, threshold_quantile)
    lesion = lesion_mask.astype(bool)
    intersection = int((selected & lesion).sum())
    union = int((selected | lesion).sum())
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    prediction = lesion if prediction_mask is None else prediction_mask.astype(bool)
    heat_y, heat_x = _weighted_centroid(heatmap)
    gt_y, gt_x = _weighted_centroid(lesion.astype(float))
    diagonal = float(np.hypot(*heatmap.shape))
    centroid_distance = np.nan if np.isnan(heat_y) or np.isnan(gt_y) else float(np.hypot(heat_y - gt_y, heat_x - gt_x) / max(diagonal, 1.0))
    total_energy = max(float(heatmap.sum()), 1e-8)
    energy_gt = float((heatmap * lesion).sum() / total_energy)
    energy_prediction = float((heatmap * prediction).sum() / total_energy)
    return {
        "xai_iou": intersection / max(union, 1),
        "xai_lesion_recall": intersection / max(int(lesion.sum()), 1),
        "xai_pointing_game": float(lesion[peak]),
        "xai_energy_inside_gt": energy_gt,
        "xai_energy_inside_prediction": energy_prediction,
        "xai_centroid_gt_distance_normalized": centroid_distance,
        "xai_off_target_attention": 1.0 - energy_gt,
    }
