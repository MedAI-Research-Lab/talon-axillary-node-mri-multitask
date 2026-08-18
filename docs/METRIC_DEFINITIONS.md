# Metric definitions

## Classification

AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier score, and expected calibration error are reported at slice and case level where available. Case probability primarily uses the mean across all slices (`mean_all`); additional aggregation rules are sensitivity analyses.

## Segmentation

Dice and IoU quantify slice-level overlap. Confidence intervals use patient-clustered bootstrap sampling. Connected-component analysis reports matched components, poorly matched components, off-target false-positive regions, and missed ground-truth regions.

Relative-overlap tiers use ground-truth coverage and prediction purity: low (both at least 25% but below the medium definition), medium (both at least 50% but below strong), and strong (both at least 75%). A missed ground-truth region has no qualifying predicted overlap.

Component area filters of >=1, >=10, and >=25 pixels are sensitivity analyses that remove increasingly small predicted components; they are not requirements to make more predictions.
