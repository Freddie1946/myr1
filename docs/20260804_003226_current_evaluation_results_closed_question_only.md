# Current evaluation results: closed-question reporting view

Recorded at: `2026-08-04 00:32:26 CST`

## Reporting scope

This snapshot follows the user's current reporting decision:

- PathMMU reports accuracy on all 999 multiple-choice questions.
- PathVQA reports **only the 3,362 yes/no questions**. Its 3,357 free-form questions, exact-match
  scores, token-F1 scores, semantic-Judge scores and any combined PathVQA score are excluded.
- OmniMedVQA reports contract-aligned/official multiple-choice accuracy on all 8,518 questions.
- A dash means that no valid full result exists. A partial run is never converted into a score.
- PathMMU test999 has already been used as a development diagnostic and is not described as an
  untouched final test.

## Main checkpoints and directly comparable Qwen baseline

| Model | PathMMU test999 | PathVQA yes/no only | OmniMedVQA | Coverage status |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-VL-7B-Instruct | 485/999 (48.55%) | 2233/3362 (66.42%) | 5118/8518 (60.08%) | Complete |
| SFT3000, epoch 3 | 582/999 (58.26%) | 1067/3362 (31.74%) | 4009/8518 (47.07%) | Complete; OmniMedVQA had 6200 generation-cap hits |
| SFT4000 continuation control, epoch 2 | 595/999 (59.56%) | 868/3362 (25.82%) | 4030/8518 (47.31%) | Complete; OmniMedVQA had 6645 generation-cap hits |
| Stage2 Outcome-GRPO, epoch 2 | 609/999 (60.96%) | 1234/3362 (36.70%) | 4029/8518 (47.30%) | Complete; OmniMedVQA had 6146 generation-cap hits |
| Stage3 GPT-4o Judge, selected epoch 2 | Pending | Pending | Pending | Training/validation complete; validation385 was 247/385 (64.16%) |
| Stage3 Kimi 2.6 Judge | Pending | Pending | Pending | Training active; 168/1500 at this snapshot |

The large generation-cap counts for the three Qwen-derived checkpoints are retained as result-
quality caveats. They are not corrected after seeing test outputs.

## Generative local baselines

| Model | PathMMU test999 | PathVQA yes/no only | OmniMedVQA | Coverage status |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B-Instruct | 449/999 (44.94%) | 1952/3362 (58.06%) | 5172/8518 (60.72%) | Complete |
| Lingshu-7B | 583/999 (58.36%) | 2865/3362 (85.22%) | 6846/8518 (80.37%) | Complete |
| InternVL3-8B | 553/999 (55.36%) | 2202/3362 (65.50%) | 6184/8518 (72.60%) | Complete |
| HuatuoGPT-Vision-7B | 531/999 (53.15%) | 2178/3362 (64.78%) | 6075/8518 (71.32%) | Complete |
| MedGemma-4B-IT | 400/999 (40.04%) | 2090/3362 (62.17%) | 6209/8518 (72.89%) | Complete; response-format/cap caveats retained |
| MedVLM-R1 | 413/999 (41.34%) | 1892/3362 (56.28%) | 4410/8518 (51.77%) | Complete |
| Llama 3.2 Vision 90B | 571/999 (57.16%) | 2101/3362 (62.49%) | 5800/8518 (68.09%) | Complete; ModelScope mirror provenance must be disclosed |
| Llama 3.2 Vision 11B | 222/999 (22.22%) | - | 4591/8518 (53.90%) | PathVQA withheld after smoke failed the 20% cap-hit gate; PathMMU had 633 cap hits |
| DeepSeek-VL2 | - | 2001/3362 (59.52%) | 4392/8518 (51.56%) | PathMMU excluded after the pre-test response-contract smoke could not extract option letters |

The Llama 90B and DeepSeek-VL2 PathVQA values use the frozen paper-aligned leading yes/no parser.
Their raw whole-completion exact-match values are not substituted for this yes/no reporting view.

## Hosted and contemporary API baselines

| Model | PathMMU test999 | PathVQA yes/no only | OmniMedVQA | Coverage status |
| --- | ---: | ---: | ---: | --- |
| Qwen-VL-Plus | 547/999 (54.75%) | 2272/3362 (67.58%) | 6151/8518 (72.22%) | Complete |
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | 466/999 (46.65% with failed request retained as wrong) | 1355/3362 (40.30%) | 4783/8518 (56.22%) | Complete; this is the user-resolved manuscript `Claude-4.5` identity |
| Grok 4.3 contemporary model | 679/999 (67.97% with failed request retained as wrong) | Partial: 222/6719 predictions | Partial: 186/8518 predictions | PathMMU complete; other tasks stopped by user because of cost |
| Doubao Seed 1.6 Vision contemporary candidate | Partial: 303/999 predictions | - | - | Stopped and removed from the planned full run by user |
| Historical Claude 3.5 Haiku | - | - | - | Retired endpoint; no runnable exact historical endpoint |
| Historical Grok-4-Fast | - | - | - | Retired/redirected; Grok 4.3 must not be relabelled as this model |
| Historical Doubao 1.5 Vision | - | - | - | Exact historical endpoint unavailable; contemporary partial run is not an exact reproduction |

The stored hosted-run metrics calculate Haiku and Grok PathMMU accuracy over successful requests.
This table instead retains the one failed request in each 999-case denominator, matching the
failures-are-wrong policy used elsewhere.

## Additional pathology image-text baselines

| Model | PathMMU test999 | PathVQA yes/no only | OmniMedVQA | Coverage status |
| --- | ---: | ---: | ---: | --- |
| CONCH | 338/999 (33.83%) | Not applicable | 3297/8518 (38.71%) | Complete for frozen image-text option-matching tasks |
| PLIP | 334/999 (33.43%) | Not applicable | 2026/8518 (23.78%) | Complete for frozen image-text option-matching tasks |
| UNI | Not applicable | Not applicable | Not applicable | Visual encoder without a compatible text/VQA scoring interface |

CONCH, PLIP and UNI are reviewer-response additions, not replacements for manuscript generative
baselines. PathVQA has no candidate-answer list, so applying the option-matching adapter there would
not be a valid comparison.

## Coverage summary and remaining work

- Full results on all three currently reported columns exist for Qwen2.5-VL-7B, Qwen2.5-VL-3B,
  SFT3000, SFT4000, Stage2, Lingshu, InternVL, Huatuo, MedGemma, MedVLM-R1, Llama 90B,
  Qwen-VL-Plus and Claude Haiku 4.5.
- Llama 11B and DeepSeek-VL2 are not crashed or silently unfinished. Each lacks one task because a
  predeclared compatibility gate failed before the corresponding full evaluation.
- Grok and Doubao are deliberate partial runs. They must remain partial unless the user explicitly
  reauthorizes their cost and contemporary-model status.
- GPT-4o Stage3 and Kimi Stage3 still require their selected-checkpoint evaluations.
- PathVQA free-form evaluation and semantic Judge work are deliberately postponed and excluded
  from this document's statistics.
- The five-model perturbation-fidelity visualization experiment is frozen but has not yet run.

## Authoritative artifact roots

- Pre-Stage3 PathMMU checkpoints:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/pathmmu_sft4000_stage2_20260729_221640`
- Earlier local baselines:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/pathmmu_diagnostic` and
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/external_vqa_full_20260729_224503`
- Hosted baselines:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/hosted_baseline_full_20260801`
- Llama/DeepSeek full runs:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/local_gpu_baselines_20260803`

