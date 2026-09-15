# Locked experimental protocol

- Models: TALON (`HYBRID_TALON`) and Mask-guided Multi-task U-Net (`UNET_MTL_MASK_GUIDED`).
- Seeds: 42, 123, 2026, 27182, and 31415.
- Split unit: patient (grouping key: `SubjectID`); reporting unit: side-specific case (`CaseID`); segmentation unit: two-dimensional slice.
- Identical split, augmentation, class weights, optimizer budget, and threshold-selection policy within each model pair.
- Source masks are binarized at intensity >127; this is not the learned prediction threshold.
- The case-classification decision threshold is fixed a priori at 0.50. Model checkpoints, segmentation thresholds, connected-component criteria, and Platt-calibration parameters are derived from validation data and locked before test evaluation. Segmentation thresholds are selected from 0.10 to 0.90 in 0.05 increments by maximum mean validation-slice Dice, with ties resolved in favor of the lower threshold.
- TALON is trained in five sequential-teacher phases. Within phases 1–5, checkpointing uses the exponentially smoothed phase-specific validation scores `teacher1_stability`, `recall_safe`, `doctor_ball`, `dual_strong`, and `final_polish`, respectively (EMA decay 0.88; minimum improvement 0.004). A new AdamW optimizer and CosineAnnealingLR scheduler are initialized at each phase. After all phases, `best_overall.pt` is copied from the checkpoint with the highest exponentially smoothed joint validation score. The comparator retains the checkpoint with the highest exponentially smoothed common validation score.
- Locked test evaluation is performed once per model/seed.
- Aggregate summaries treat seed-level values as the experimental units; patient observations repeated across holdouts are not pooled as independent samples.
