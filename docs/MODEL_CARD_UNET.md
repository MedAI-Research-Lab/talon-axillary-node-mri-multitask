# Model card: Mask-guided Multi-task U-Net

1. The comparator jointly predicts a clinically preselected axillary target-node mask and InvBC-versus-IGM source-cohort class. Its classification branch uses predicted-mask guidance; the reference mask is not a classifier input at inference. It shares the cleaned data, patient-grouped split (technical grouping key: `SubjectID`), seed, augmentation, class weighting, optimizer budget, validation-only model-selection framework, and evaluation pipeline with TALON. The case-classification decision threshold is fixed a priori at 0.50; segmentation thresholds and connected-component criteria are selected from validation data and locked before test evaluation, and Platt calibration is fitted on validation predictions.

2. It is a task-matched, capacity-aware multi-task comparator rather than a classification-only CNN or segmentation-only U-Net; it is not strictly parameter-matched to TALON. It is intended for research benchmarking on slices containing a clinically preselected visible target node and does not perform whole-examination lymph-node detection or screening.

