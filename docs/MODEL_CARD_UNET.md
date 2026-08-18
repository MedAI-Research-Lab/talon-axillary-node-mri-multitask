# Model card: Mask-guided Multi-task U-Net

The comparator jointly predicts a lesion mask and cohort-origin class. Its classification branch uses predicted-mask guidance rather than a ground-truth mask at inference. It shares the cleaned data, subject-grouped split, seed, augmentation, class weighting, optimizer budget, and validation threshold policy with TALON.

It is a capacity-aware multi-task comparator, not a classification-only CNN or segmentation-only U-Net. It is intended for research benchmarking and not clinical deployment.
