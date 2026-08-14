# Interpretability and human-rating case selection (2026-08-14)

## Outcome

Both preregistered case-count gates passed.

| Deliverable | Required | Frozen | Status |
|---|---:|---:|---|
| Strictly effective interpretability cases | >=20 | 20 | passed |
| Blinded human reward-agreement cases | >=50 | 60 | passed |

These outputs are case panels for expert review and subsequent mechanistic work. External-model regions are pseudo-reference evidence regions, not pathology-expert ground truth.

## Interpretability panel

### Target-independent region selection

1. Freeze 160 PathMMU candidate cases before inspecting target-model outputs (40 per answer choice).
2. Annotate all visible supporting/opposing pathology regions with Gemini 3.1 Pro. All 160 cases were validated with exact served-model identity.
3. Apply a target-output-blind quality filter. 123/160 passed (A/B/C/D: 29/32/33/29).
4. Send all 123 eligible cases to Claude Opus 5 for independent review. Opus did not see Gemini's annotations, target-model predictions, saliency maps, or perturbation results.
5. Strictly validate 120 Opus outputs. Three cases (panel indices 109, 112, 117) were transparently excluded after repeated malformed-coordinate responses; no coordinates were manually repaired.
6. Retain only Gemini regions independently corroborated by at least one Opus region at overlap coefficient >=0.20. Unmatched regions remain in the audit and are not added to the primary deletion mask. This produced 94 consensus cases (A/B/C/D: 21/25/24/24).

### Target-model behavioral gate

Target checkpoint:

`stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1500`

For each consensus case, the model was evaluated with:

- original image;
- Gaussian-blur replacement inside the consensus evidence mask;
- an area/shape-matched neighboring mask;
- three area/shape-matched random translated masks.

Normal reasoning generation used `max_new_tokens=1024`. A separate forced A/B/C/D score was used only after verifying agreement with normal generation on each case.

The strict effective-case rule was frozen as:

1. clean normal generation is correct;
2. clean normal-generation choice agrees with the forced-choice score;
3. deleting the reference evidence lowers the target-choice margin;
4. that margin loss exceeds the mean loss from the three matched random deletions.

Observed funnel:

| Stage | Cases |
|---|---:|
| Dual-model consensus cases evaluated | 94 |
| Clean normal-generation correct | 51 |
| Clean correct and generation/score agreement | 47 |
| Positive reference-region margin drop among primary cases | 34 |
| Strictly effective cases | 20 |

Strict effective answer distribution: A=5, B=7, C=3, D=5.

Among the 47 primary cases, mean target-margin drops were:

| Intervention/statistic | Mean | Bootstrap 95% CI |
|---|---:|---:|
| Reference evidence deletion | 0.4542 | [0.1520, 0.8079] |
| Random matched deletion | 0.3452 | [0.1550, 0.5744] |
| Neighbor matched deletion | 0.2456 | [0.0661, 0.4520] |
| Reference minus random extra drop | 0.1090 | [-0.0648, 0.2989] |

The aggregate reference-minus-random confidence interval includes zero. Therefore, the current result supports a panel of 20 strong individual causal examples, but does not yet support a population-wide claim that reference deletion is consistently stronger than random deletion. Expert ROI verification and/or a larger confirmatory panel is still required for that stronger statement.

## Human reward-agreement panel

The packet was sampled from all 12,000 final Stage3 trajectories (1,000 unique prompts), rather than only high-scoring outputs.

- 60 unique prompt/trajectory cases;
- 36 representative cases covering stage third × answer correctness × process-score band (18 cells ×2);
- 24 challenge cases: six high-process/wrong-answer, six low-process/correct-answer, four histological-error positives, four logical-contradiction positives, and four incorrect/outdated-criterion positives;
- Reviewer A: all 60;
- Reviewer B: at least 30 overlapping cases for inter-rater agreement;
- the blinded packet excludes judge identity, reward, stage, rank, and sampling-stratum information;
- representative and challenge subsets must be reported separately.

This panel tests agreement between human ratings and the GPT-4o process reward. It is not a random benchmark-quality sample, so challenge cases must not be pooled naively with representative cases when estimating natural prevalence.

## Frozen artifacts and hashes

| Artifact | SHA256 |
|---|---|
| `protocol/predefined_evidence_candidate_panel_v2_160case_20260813.json` | `36772faaa7d4b8eb4d82ce1a543645a6c8da26305355bba521c6c821732469a8` |
| `protocol/dual_model_roi_opus_review_panel_all123_20260814.json` | `69e41e40d1cb36996fcd4f842bb8746fb952667882c2f84fb829ff398ea7c9ec` |
| `protocol/dual_model_roi_consensus_panel_all_eligible_20260814.json` | `0b7990212d3e67e843f998f53822da6a70b613d6b391ca85d2750e001bebab85` |
| `protocol/effective_interpretability_panel_20case_20260814.json` | `87cd6599c0067e66b6a3609aedd073b5c870aba62d46121100c67d0ac9ea8ae0` |
| Human packet `manifest.json` | `29bdf2883f4cc426ae30b6f1f985dfbce6fbbac205111a776856c231162ab267` |
| Human packet `cases.jsonl` | `a612b61688b43af7b021e1c07ab8197ebff8602e27554215e488592e7525dbae` |
| Human packet `ratings_template.csv` | `97e44cd7c9ce6f8a369db373dde9b997eebe7bd77a8ae628eeac69eab79a3a65` |

Run artifacts:

- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/dual_model_roi_formal_20260814`
- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/reward_agreement_60case_20260814`

Private migration backup:

- dataset: `Freddie1946/PathVLM-R1-Revision-Evaluation-Results`
- verified revision: `b2e61cbec20a679c99bba640975f772f033f121a`
- prefix: `snapshots/20260814_interpretability_human_case_selection/`
- archive size: 35,180,929 bytes
- archive SHA256: `a988f81a341f4d773bf89c58839443f19481d66f80e6ab73f5670b480762246e`
- remote visibility was verified private; the archive contains the private human-selection key and must not be made public.

## Remaining human work

1. A pathologist verifies or edits the 20 reference evidence regions without seeing target-model saliency.
2. Reviewer A rates all 60 reward-agreement cases; reviewer B independently rates the prescribed overlap.
3. Only after ratings are locked should the private selection key be joined back to compute judge–human agreement, inter-rater agreement, and error-category results.
