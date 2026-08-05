# Stage3 selected-model full evaluation pipeline prepared

Recorded at 2026-08-04 00:20 CST.

GPT-4o Stage3 has **not** yet been evaluated on PathMMU test999. Its only completed post-training
inference is the frozen PathMMU validation385 selection: epoch1 234/385, epoch2 247/385, and epoch3
245/385. Epoch2 step1000 was selected. A workspace-wide reverse lookup of the selected checkpoint
in metrics artifacts found only those validation files and no 999-record prediction artifact.

`scripts/run_stage3_selected_full_evaluations.sh` now prepares the truthful next execution boundary.
As of 2026-08-05, `PATHVLM_STAGE3_EVAL_ARMS` explicitly selects `gpt4o`, `kimi26`, or both.
The GPT-4o-only path does not read or require a Kimi selection, so the completed GPT-4o arm can be
evaluated while Kimi remains incomplete. Once the three assigned GPUs are idle, it will:

1. verify every requested selected epoch snapshot and all source-data hashes;
2. run three 16-case behavior smokes per requested arm across PathMMU, PathVQA and OmniMedVQA;
3. gate only on output availability, parsing and truncation, never smoke accuracy;
4. run three full jobs per requested arm on explicitly assigned GPUs;
5. verify counts, ordered source identities, model/data hashes and prediction-file hashes.

PathMMU is labeled `test999_development`, because it has already been used during Stage3 planning.
PathVQA covers only the frozen 3,362 yes/no subset; free-form inference remains postponed and is
excluded from the combined reporting view. OmniMedVQA covers all 8,518 questions and reports its
four OOD sources separately, so the original Chest CT, ISIC2020, Retinal OCT-C8 and Diabetic
Retinopathy experiments are not duplicated under a second name.
