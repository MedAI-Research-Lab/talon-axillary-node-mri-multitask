# Public data dictionary

- `SubjectID`: deidentified patient-grouping identifier. All cases from one patient remain in one split.
- `CaseID`: deidentified side-specific reporting case.
- `ClassName`: immutable machine-readable cohort-origin code (`IBC` or `IGM`). The public display label for `IBC` is `InvBC`.
- `ImageName`: deidentified dataset image filename.
- `RoiSizeBin`: training-derived ROI-size category.
- `seed`: repeated-holdout seed.
- `model`: internal run identifier; map `HYBRID_TALON` to the public display name `TALON`.
- `calibration`: raw or validation-calibrated probability.
- `classification_threshold`: prespecified case-classification decision threshold fixed at 0.50.
- `segmentation_threshold`: model- and seed-specific threshold selected from validation data and locked before test evaluation.
- `off_target`: predicted component with no reference-component intersection.
- `missed`: reference component without a prediction that meets both validation-locked component-matching criteria.

`<PRIVATE_DATA_ROOT>`, `<PRIVATE_PATH>`, and `<REPOSITORY_ROOT>` are deliberate redactions of local filesystem paths.
