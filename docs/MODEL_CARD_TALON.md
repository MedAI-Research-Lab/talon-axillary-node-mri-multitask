# Model card: TALON

TALON is a joint 2D breast-MRI lesion segmentation and cohort-origin classification model. It uses a 13-channel anatomically enriched input, a four-scale residual encoder, vertical-horizontal attention, ASPP, anatomical focusing, a skip decoder with PatchNIN refinement, and a six-branch legacy evidence-fusion classifier. Predicted lesion probability is detached before classifier guidance. Ground-truth masks supervise training and are not classifier inputs at inference.

The executed model contains 4,930,849 trainable parameters. It was trained in five sequential-teacher phases with AdamW and validation-based checkpoint/threshold selection.

Intended use is research comparison, not autonomous diagnosis. The study is single-center, retrospective, and internally validated; external validation is absent.
