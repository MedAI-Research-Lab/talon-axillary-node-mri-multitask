# Model card: TALON
1. TALON is a joint two-dimensional breast DCE-MRI model for segmentation of a clinically preselected axillary target node and InvBC-versus-IGM source-cohort classification. It uses a 13-channel anatomically enriched input, a four-scale residual encoder, vertical-horizontal attention, ASPP, anatomical focusing, a skip decoder with PatchNIN refinement, and a six-branch evidence-fusion classifier. Predicted target-node probability is detached before classifier guidance. Reference masks supervise training and are not classifier inputs at inference.

2. The executed model contains 4,930,849 trainable parameters. It was trained in five sequential-teacher phases with AdamW, validation-based checkpoint and segmentation-threshold selection, and a fixed case-classification threshold of 0.50.

3. TALON is intended for research comparison on slices containing a clinically preselected visible axillary target node; it does not perform whole-examination lymph-node detection or screening. The study is single-center, retrospective, and internally validated.
