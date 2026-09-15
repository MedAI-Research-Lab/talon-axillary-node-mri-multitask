# Metric definitions

## Classification and calibration

AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier score, and expected calibration error (ECE) are reported at slice and case level where available. Internal machine-readable labels use IBC = 0 and IGM = 1; reported probabilities refer to the IGM class. The case-classification decision threshold is fixed at 0.50. Case probability primarily uses the mean across all slices (`mean_all`); additional aggregation rules are sensitivity analyses. Platt calibration is fitted on validation predictions. Brier score and ECE are lower-is-better metrics; ECE uses 10 equal-width probability bins.

## Segmentation and connected components

Dice and IoU quantify two-dimensional slice-level overlap. Unless otherwise stated, 95% confidence intervals use 5,000 percentile-bootstrap resamples clustered by `SubjectID`, so all cases belonging to one patient are resampled together. Two-dimensional components are labeled independently on each slice with 8-connectivity and are reported as component occurrences rather than three-dimensional detections.

For predicted component P and reference component G, IoU = |P intersection G| / |P union G| and prediction purity = |P intersection G| / |P|. The validation-locked matching cut-offs are IoU >= 0.05 and purity >= 0.05 for every executed model and seed. A predicted component is `matched` when both cut-offs are met for an intersecting reference component, `poorly_matched` when an intersection exists but either cut-off is not met, and `off_target_fp` when no reference-component intersection exists. A reference component is `missed` when no predicted component meets both matching cut-offs. Fragmentation denotes a reference component matched by more than one predicted component.

For the relative-overlap sensitivity analysis, reference coverage = |prediction intersection reference| / |reference|, prediction purity = |prediction intersection reference| / |prediction|, and q = min(coverage, purity). Categories are `no_overlap`; `partial_below_25` for 0 < q < 0.25; `low_25_to_50` for 0.25 <= q < 0.50; `moderate_50_to_75` for 0.50 <= q < 0.75; and `strong_75_plus` for q >= 0.75. In the reference-component view, all predicted components touching the same reference component are unioned before coverage and purity are computed. Minimum predicted-component areas of >=1, >=10, and >=25 raster pixels are sensitivity filters that progressively remove small components; they do not require additional predictions.

## Grad-CAM

Grad-CAM targets `model.neck` for TALON and `model.bottleneck` for the Mask-guided Multi-task U-Net. For each slice, gradients of the predicted-class logit are globally averaged over the target-layer feature map, used to weight the activations, passed through ReLU, bilinearly upsampled to input resolution, and min-max normalized to [0,1]. The binary activation region is defined by the within-map 80th percentile (top 20%). XAI IoU is |activation intersection reference| / |activation union reference|; target-node recall is |activation intersection reference| / |reference|; pointing-game success equals 1 when the heatmap maximum lies inside the reference mask and 0 otherwise; and activation energy within the reference mask is sum(heatmap x reference) / sum(heatmap).

