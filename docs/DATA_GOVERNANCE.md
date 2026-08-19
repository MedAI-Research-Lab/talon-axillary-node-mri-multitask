# Data governance and public-release boundary

## Included

- Deidentified `SubjectID` and side-specific `CaseID` labels.
- Sanitized split assignments, metrics, curves, connected-component records, and audit outputs.
- Two frozen composite segmentation figures prepared for the manuscript.

## Excluded

- Raw MRI/DICOM/JPEG slices and raster masks.
- Patient names, masked-name strings, identity assertions, and re-identification keys.
- Local absolute paths and private clinical-document paths.
- Per-pixel prediction arrays and model checkpoints.

## Mandatory check before public upload

The files paper_outputs/figures/main/figure_3_test_segmentation_cases_en.png and paper_outputs/figures/supplementary/supplementary_figure_S9_additional_segmentation_cases_en.png contain deidentified clinical images prepared for the manuscript. Public-release review was completed before repository publication, and dissemination of these files was confirmed under the applicable institutional, ethics, consent-waiver, data-protection, and journal requirements. No direct identifiers, identity keys, or re-identification information are included.
Case identifiers are research pseudonyms. They must never be linked to an identity key in this repository.
