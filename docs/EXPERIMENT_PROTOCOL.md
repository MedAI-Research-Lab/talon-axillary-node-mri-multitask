# Locked experimental protocol

- Models: TALON (`HYBRID_TALON`) and Mask-guided Multi-task U-Net (`UNET_MTL_MASK_GUIDED`).
- Seeds: 42, 123, 2026, 27182, and 31415.
- Split unit: subject; reporting unit: side-specific case.
- Identical split, augmentation, class weights, optimizer budget, and threshold-selection policy within each model pair.
- Source masks are binarized at intensity >127; this is not the learned prediction threshold.
- Segmentation and classification thresholds are selected on validation data and locked before test access.
- Five TALON sequential-teacher phases retain phase-specific optimizer reset, CosineAnnealingLR, early stopping, and checkpoint selection.
- Locked test evaluation is performed once per model/seed.
- Aggregate summaries treat seed-level values as the experimental units; patient observations repeated across holdouts are not pooled as independent samples.
