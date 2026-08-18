# Public data dictionary

- `SubjectID`: deidentified split-group identifier. All cases from one subject remain in one split.
- `CaseID`: deidentified side-specific reporting case.
- `ClassName`: IBC or IGM cohort-origin label.
- `ImageName`: deidentified dataset image filename.
- `RoiSizeBin`: training-derived ROI-size category.
- `seed`: repeated-holdout seed.
- `model`: internal run identifier; map `HYBRID_TALON` to manuscript TALON.
- `calibration`: raw or validation-calibrated probability.
- `classification_threshold`, `segmentation_threshold`: validation-locked thresholds.
- `off_target`: predicted component without qualifying ground-truth overlap.
- `missed`: ground-truth component without a qualifying prediction.

`<PRIVATE_DATA_ROOT>`, `<PRIVATE_PATH>`, and `<REPOSITORY_ROOT>` are deliberate redactions of local filesystem paths.
