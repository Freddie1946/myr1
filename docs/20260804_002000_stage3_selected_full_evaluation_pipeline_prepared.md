# Stage3 selected-model full evaluation pipeline prepared

Recorded at 2026-08-04 00:20 CST.

GPT-4o Stage3 has **not** yet been evaluated on PathMMU test999. Its only completed post-training
inference is the frozen PathMMU validation385 selection: epoch1 234/385, epoch2 247/385, and epoch3
245/385. Epoch2 step1000 was selected. A workspace-wide reverse lookup of the selected checkpoint
in metrics artifacts found only those validation files and no 999-record prediction artifact.

`scripts/run_stage3_selected_full_evaluations.sh` now prepares the truthful next execution boundary.
It remains gated on completion of Kimi training and Kimi's own validation385 checkpoint selection.
Once both selections exist and all GPUs are idle, it will:

1. verify both selected epoch snapshots and source-data hashes;
2. run six 16-case behavior smokes for GPT-4o/Kimi across PathMMU, PathVQA and OmniMedVQA;
3. gate only on output availability, parsing and truncation, never smoke accuracy;
4. run six full jobs concurrently on GPUs 0-5;
5. verify counts, ordered source identities, model/data hashes and prediction-file hashes.

PathMMU is labeled `test999_development`, because it has already been used during Stage3 planning.
PathVQA covers all 6,719 test questions. OmniMedVQA covers all 8,518 questions and reports its four
OOD sources separately, so the original Chest CT, ISIC2020, Retinal OCT-C8 and Diabetic Retinopathy
experiments are not duplicated under a second name.

