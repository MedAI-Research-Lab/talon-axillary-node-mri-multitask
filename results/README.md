# Frozen results

This directory contains sanitized outputs from the completed locked-test evaluations:

- `aggregate/`: per-seed metrics and five-seed mean/SD/range.
- `paired_comparisons/`: TALON-minus-U-Net paired summaries.
- `per_seed/`: confusion matrices, paired predictions/statistics, and relative-overlap records.
- `model_runs/`: selected training history, thresholds, curves, classification/segmentation/component metrics, predictions, XAI summaries, and TALON teacher comparisons.
- `components/`: five-seed FP ROI and GT component lists.
- `reproducibility/`: output contracts and parity manifests.

Raw probability arrays and checkpoint binaries are intentionally absent.
