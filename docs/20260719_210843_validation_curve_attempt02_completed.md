# Formal SFT validation curve Attempt02 completed

Timestamp: `2026-07-19T21:08:43+08:00`

## Formal result

Attempt02 completed successfully:

- run ID: `sft_base_n2000_n3000_seed0042_20260719_184313`;
- stage: `stage1_sft_validation_curve`;
- status: `completed`;
- `formal_result: true`;
- start: `2026-07-19T18:43:13+08:00`;
- completion: `2026-07-19T19:35:52+08:00`;
- wall time: approximately 52 minutes 40 seconds;
- run-manifest SHA-256:
  `e7170be9ddd3050b8421a02c3a12a3e59d9762b6296a26f246abe445ea06ff24`;
- test accessed: false.

The run used `pathmmu_image_disjoint_v2` validation only, deterministic decoding, 192 maximum new
tokens, parser v2, and the pinned explicit chat template. Attempt01 predictions were not reused.

## Completeness audit

- all 21 planned jobs completed and passed every job gate;
- every job saved exactly 385 raw predictions;
- total raw predictions: 8,085;
- every offline re-score matched the recorded online parser result;
- base, both formal parents, and all 20 epoch snapshots passed their recorded hash/provenance gates;
- no inference/training process remained after completion;
- output size was approximately 12 MiB; disk free remained approximately 931 GiB.

External run root:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_validation_curves/sft_base_n2000_n3000_seed0042_20260719_184313`

## Accuracy and format curves

| Model | Epoch | Step | Accuracy | Format |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 0 | 0 | 25.97% | 34.81% |
| SFT n=2000 | 1 | 250 | 52.99% | 99.74% |
| SFT n=2000 | 2 | 500 | 55.06% | 100.00% |
| SFT n=2000 | 3 | 750 | 52.21% | 99.22% |
| SFT n=2000 | 4 | 1000 | 48.83% | 99.74% |
| **SFT n=2000** | **5** | **1250** | **57.92%** | **99.74%** |
| SFT n=2000 | 6 | 1500 | 53.25% | 99.74% |
| SFT n=2000 | 7 | 1750 | 51.17% | 100.00% |
| SFT n=2000 | 8 | 2000 | 52.99% | 100.00% |
| SFT n=2000 | 9 | 2250 | 51.17% | 100.00% |
| SFT n=2000 | 10 | 2500 | 51.95% | 100.00% |
| SFT n=3000 | 1 | 375 | 58.96% | 100.00% |
| SFT n=3000 | 2 | 750 | 57.14% | 98.96% |
| **SFT n=3000** | **3** | **1125** | **60.52%** | **99.74%** |
| SFT n=3000 | 4 | 1500 | 57.14% | 83.12% |
| SFT n=3000 | 5 | 1875 | 58.44% | 100.00% |
| SFT n=3000 | 6 | 2250 | 53.77% | 100.00% |
| SFT n=3000 | 7 | 2625 | 56.10% | 100.00% |
| SFT n=3000 | 8 | 3000 | 54.81% | 100.00% |
| SFT n=3000 | 9 | 3375 | 55.32% | 99.74% |
| SFT n=3000 | 10 | 3750 | 55.58% | 100.00% |

The frozen selection rule chose n=2000 epoch 5 (223/385 correct) and n=3000 epoch 3 (233/385
correct). Descriptive 95% Wilson intervals are 52.94--62.75% and 55.56--65.28%, respectively; these
intervals were not used for checkpoint selection.

## Interpretation and limitations

The result directly shows that validation accuracy is non-monotonic with training duration. Keeping
only epoch 10 would have hidden the best checkpoints: epoch 10 is 5.97 percentage points below the
n=2000 peak and 4.94 points below the n=3000 peak. This supports early stopping/checkpoint selection
on validation and confirms that excessive SFT can reduce downstream accuracy.

The 2.60-point difference between the selected n=3000 and n=2000 checkpoints is a single-validation-
split, single-seed result; it is not by itself a significance or generalization claim. The formal
SFT-size comparison remains incomplete until the n=500/n=1000 lineages receive a predeclared,
comparable validation treatment. Seeds 43/44 and final locked test evaluation also remain pending.

## Next gate

Do not start formal Outcome GRPO solely because Attempt02 succeeded. First audit the existing n=500
and n=1000 formal checkpoints and decide whether their final-only retention is sufficiently comparable
or whether they must be rerun with the two-tier epoch policy. Then freeze the Stage 2 parent and run
the one-step parser/reward-variance/gradient/tensor-delta gate before any long RL run.
