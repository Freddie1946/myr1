# Final closed-benchmark and blind multi-judge evaluation

Updated: 2026-08-10 CST

## Reporting contract

- PathMMU: all 999 multiple-choice questions. This split has been used for development diagnosis
  and is not described as an untouched final test.
- PathVQA: only the frozen 3,362 yes/no questions. Free-form questions are excluded from this
  table. The primary score extracts the leading yes/no answer; legacy whole-completion exact match
  is retained only as an audit field.
- OmniMedVQA: all 8,518 multiple-choice questions. The primary number is the contract-aligned
  answer extraction. The official raw-completion sequence-matcher result is shown after `/` when it
  differs materially.
- Every full result below has a count/hash integrity record. Partial API runs are never converted
  into full scores.

## Our checkpoints and direct base-model control

| Model | PathMMU | PathVQA yes/no | OmniMedVQA aligned / official |
|---|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | 485/999 (48.55%) | 2233/3362 (66.42%) | 5118/8518 (60.08%) |
| SFT3000 | 582/999 (58.26%) | 1728/3362 (51.40%) | 3823/8518 (44.88%) / 4009 (47.07%) |
| SFT4000 continuation control | 595/999 (59.56%) | 1020/3362 (30.34%) | 3674/8518 (43.13%) / 4030 (47.31%) |
| Stage2 Outcome-GRPO | 609/999 (60.96%) | 1669/3362 (49.64%) | 3874/8518 (45.48%) / 4029 (47.30%) |
| Stage3 GPT-4o selected checkpoint | 604/999 (60.46%) | 1757/3362 (52.26%) | 3991/8518 (46.85%) / 4104 (48.18%) |
| Stage3 Grok 4.3 selected checkpoint | 633/999 (63.36%) | 1758/3362 (52.29%) | 3955/8518 (46.43%) / 3974 (46.65%) |

The SFT/GRPO checkpoints frequently fill the 64-token generation allowance on OmniMedVQA, so the
aligned extraction is the primary parser-robust result and the official raw-completion number is
kept as a sensitivity analysis. Stage3 GPT-4o and Stage3 Grok respectively have 6,895 and 6,652
cap hits out of 8,518 and must carry this caveat.

## Generative local baselines

| Model | PathMMU | PathVQA yes/no | OmniMedVQA aligned / official | Note |
|---|---:|---:|---:|---|
| Qwen2.5-VL-3B-Instruct | 449/999 (44.94%) | 1952/3362 (58.06%) | 5172/8518 (60.72%) | Complete |
| Lingshu-7B | 583/999 (58.36%) | 2865/3362 (85.22%) | 6846/8518 (80.37%) | Complete |
| InternVL3-8B | 553/999 (55.36%) | 2202/3362 (65.50%) | 6182/8518 (72.58%) / 6184 (72.60%) | Complete |
| HuatuoGPT-Vision-7B | 531/999 (53.15%) | 2178/3362 (64.78%) | 6075/8518 (71.32%) | Complete |
| MedGemma-4B-IT | 400/999 (40.04%) | 2090/3362 (62.17%) | 6101/8518 (71.62%) / 6209 (72.89%) | Format/cap caveat |
| MedVLM-R1 | 413/999 (41.34%) | 1892/3362 (56.28%) | 4335/8518 (50.89%) / 4410 (51.77%) | Complete |
| Llama 3.2 Vision 90B | 571/999 (57.16%) | 2101/3362 (62.49%) | 5800/8518 (68.09%) | ModelScope mirror provenance |
| Llama 3.2 Vision 11B | 222/999 (22.22%) | 1882/3362 (55.98%) | 4591/8518 (53.90%) / 4592 (53.91%) | Long-output/cap caveat |
| DeepSeek-VL2 | 429/999 (42.94%) | 2001/3362 (59.52%) | 4392/8518 (51.56%) / 4393 (51.57%) | PathMMU used disclosed letter-only prompt |
| ScaleReasoner-R1 | 648/999 (64.86%) | 1934/3362 (57.53%) | 3688/8518 (43.30%) / 3685 (43.26%) | Omni had 1,460 cap hits |

ScaleReasoner-R1 is an added contemporary pathology reasoning baseline at fixed revision
`ce7f51daa9731f106874ac2bee0e9a864f7a3636`. Its nested Transformers configuration required only
an in-memory compatibility normalization; weights and saved configuration were not changed.
Its model card says it was initialized from Patho-R1 and GRPO-trained on Scale-VQA, and itself
reports PathMMU transfer results. We therefore report this as a contemporary external comparison,
not as a guaranteed contamination-free baseline. Its model-card PathMMU number is not substituted
for our frozen image-disjoint 999-case result.
It is the strongest completed local open-weight baseline on this PathMMU split, but its much lower
external scores show that this advantage does not transfer uniformly to PathVQA/OmniMedVQA.

## Hosted and image-text baselines

| Model | PathMMU | PathVQA yes/no | OmniMedVQA | Status |
|---|---:|---:|---:|---|
| Qwen-VL-Plus | 547/999 (54.75%) | 2272/3362 (67.58%) | 6151/8518 (72.22%) | Complete |
| Claude Haiku 4.5 | 466/999 (46.65%) | 1355/3362 (40.30%) | 4783/8518 (56.22%) | Complete; failed requests count wrong |
| Grok 4.3 hosted baseline | 679/999 (67.97%) | 222/6719 partial | 186/8518 partial | Cost-stopped partial; not reported as full |
| CONCH image-text matching | 338/999 (33.83%) | Not applicable | 3297/8518 (38.71%) | Frozen option-matching adapter |
| PLIP image-text matching | 334/999 (33.43%) | Not applicable | 2026/8518 (23.78%) | Frozen option-matching adapter |

## Stage3 paired inference conclusions

| Comparison | Benchmark | Difference | Paired / clustered inference | Conclusion |
|---|---|---:|---|---|
| GPT-4o Stage3 - Stage2 | PathMMU | -0.50 points | McNemar p=.709; cluster CI [-2.64,+1.65] | No test gain |
| GPT-4o Stage3 - Stage2 | PathVQA yes/no | +2.62 points | p=2.73e-11; cluster CI [+1.86,+3.38] | Robust gain |
| GPT-4o Stage3 - Stage2 | Omni aligned | +1.37 points | p=.0100; cluster CI [+0.32,+2.39] | Small aligned-parser gain |
| GPT-4o Stage3 - Stage2 | Omni official | +0.88 points | p=.105; cluster CI [-0.17,+1.91] | Not robust to scoring contract |
| Grok Stage3 - Stage2 | PathMMU | +2.40 points | McNemar p=.0635; cluster CI [0.00,+4.82] | Positive but borderline |
| Grok Stage3 - Stage2 | PathVQA yes/no | +2.65 points | p=8.77e-11; cluster CI [+1.86,+3.46] | Robust gain |
| Grok Stage3 - Stage2 | Omni aligned | +0.95 points | p=.0790; cluster CI [-0.11,+2.00] | Inconclusive |
| Grok Stage3 - Stage2 | Omni official | -0.65 points | p=.245; cluster CI [-1.73,+0.41] | Inconclusive |

The remembered GPT-4o Stage3 improvement was real on the checkpoint-selection validation split:
236/385 (61.30%) to 247/385 (64.16%), +2.86 points. Its paired cluster CI was
[-0.78,+6.67] and the same split selected checkpoint 1000, so it is selection-biased and not
independent confirmatory evidence. On PathMMU test999, none of GPT-4o checkpoints 500/1000/1500
beat Stage2. Grok's test999 gain is a different result: it is positive but narrowly misses the
conventional 0.05 threshold because its 89 improvements are offset by 65 regressions.
These p-values are unadjusted; applying a multiple-comparison correction would not turn any
borderline result into a stronger claim.

Independent eight-case visual audits for each Stage3 arm confirmed that the paired flips represent
real visual/knowledge/option-reasoning repairs and newly introduced errors, not parser artifacts.
The GPT-4o audit cost $0.052983; the Grok audit cost $0.054522.

## Blind multi-judge reasoning-quality evaluation

Protocol: a frozen, output-independent 100-case PathMMU panel; six anonymous candidate models;
30 cases per model repeated three times; primary inference uses run 0 only. Judges are GPT-4o,
Claude Sonnet 4.6, and Gemini 3.1 Pro. Each metric is in [0,1]: reasoning accuracy, knowledge
accuracy, rigor, professionalism, clarity, and conciseness. Models are never selected post hoc by
their most favorable judge.

The paired six-metric macro results available from the completed lanes are:

| Candidate | GPT-4o macro | Claude macro | Gemini macro |
|---|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | 0.7922 | 0.5987 | 0.6335 |
| SFT3000 | 0.7717 | 0.5939 | 0.6213 |
| SFT4000 control | 0.7634 | 0.5873 | 0.6177 |
| Stage2 Outcome-GRPO | 0.7804 | 0.5952 | 0.6126 |
| Stage3 GPT-4o | 0.7937 | 0.6123 | 0.6209 |
| Stage3 Grok 4.3 | 0.7993 | 0.6268 | 0.6223 |

Paired effect sizes versus Stage2:

| Judge | GPT-4o Stage3 - Stage2 | Grok Stage3 - Stage2 |
|---|---:|---:|
| GPT-4o | +0.0132, 95% CI [-0.0262,+0.0526] | +0.0189, 95% CI [-0.0207,+0.0576] |
| Claude Sonnet 4.6 | +0.0170, 95% CI [-0.0227,+0.0555] | +0.0315, 95% CI [-0.0107,+0.0726] |
| Gemini 3.1 Pro | +0.0082, 95% CI [-0.0105,+0.0276] | +0.0096, 95% CI [-0.0120,+0.0313] |

All three judges give both Stage3 arms a positive macro point estimate relative to Stage2, but all
six judge-by-Stage3 macro confidence intervals include zero. This supports a small, directionally
consistent reasoning-quality improvement, not a statistically significant claim on this 100-case
panel. GPT-4o is not uniquely favorable to the GPT-4o-trained arm: Claude gives that arm the
largest macro delta, while all three judges rank the Grok arm's delta at least as high.

Across 600 primary model-case pairs, inter-judge correlations are strong for the substantive
metrics (reasoning accuracy, knowledge accuracy and rigor: Pearson r=0.8413–0.9441) and lower for
style metrics (r=0.3702–0.8622). Repeated-case content-metric ICCs are 0.9428–0.9971; style
reliability is weaker for some Gemini ratings, especially conciseness (ICC=0.3937). This is why
primary conclusions are reported per judge rather than choosing or pooling the most favorable one.

All 2,880 final judgments are present (960 per judge), served-model identities were checked, one
Claude JSON-format failure and one Gemini timeout succeeded under bounded retry, and no rule
fallback was used. Formal judge calls, smoke calls, and both visual bad-case audits cost
`$29.141971` in total under the approved `$40` ceiling.

## Artifact integrity and backup

All frozen raw predictions, integrity records, paired analyses, bad-case audits, corrected rescoring
artifacts and multi-judge outputs from this goal are backed up under
`snapshots/20260810_final_eval_goal/` in the private Hugging Face dataset
`Freddie1946/PathVLM-R1-Revision-Evaluation-Results`, revision
`4985de4b09687bd8f55bb55d28f24aea1b28ec50`. A fresh remote download of the final GPT-4o
comparison JSON matched the local SHA-256
`f02b6bf1e520fd7956499e43ef7c328eb06f4edc63c8a03b61e9f9edccd56f65`.
No model weights were included in this results snapshot.
