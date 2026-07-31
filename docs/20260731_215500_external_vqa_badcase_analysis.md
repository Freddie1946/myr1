# PathVQA and OmniMedVQA bad-case analysis

## Scope and metric integrity

This audit compares the SFT3000 parent, SFT4000 continuation control and
selected Stage2 Outcome-GRPO checkpoint on the already completed frozen
PathVQA and OmniMedVQA runs.  Formal scores are not changed.  Parser-aware
counts below are explicitly diagnostic and must not replace the primary
metrics.

Reproducible machine-readable output:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/external_vqa_badcases_20260731/badcase_analysis.json`

## Formal scores

| Model | PathVQA exact match | OmniMedVQA official matcher |
|---|---:|---:|
| SFT3000 | 15.91% | 47.07% |
| SFT4000 | 12.92% | 47.31% |
| Stage2 RL | 18.41% | 47.30% |

## PathVQA: dominant output-contract failure

PathVQA's frozen primary metric is normalized whole-string exact match.  The
models normally answer in full sentences, so a medically compatible answer is
still wrong unless it exactly equals the short reference.

- Stage2 obtains only `3/3357` exact matches on free-form questions, versus
  `1234/3362` on yes/no questions.
- Stage2 has 770 formally wrong completions that nevertheless contain the
  normalized reference answer.  SFT3000 has 934 and SFT4000 has 1163.
- Example: reference `positively charged`; Stage2 completion `Each histone
  subunit is positively charged.`  The frozen exact-match metric marks it
  wrong.
- Every generation-cap case scores zero.  Cap counts are 1221 (SFT3000), 2396
  (SFT4000) and 1114 (Stage2).
- SFT4000 became substantially more verbose (40.60 mean generated tokens,
  versus 26.14 for SFT3000 and 23.40 for Stage2) and hit the cap most often.
  This explains much of its lower PathVQA formal score.

This is strong evidence of an evaluation-interface mismatch.  It does not
prove that all target-containing completions are semantically correct, so the
counts remain diagnostics rather than corrected scores.

## OmniMedVQA: scorer conflict plus genuine domain shift

The frozen official rule maps the **entire completion** to whichever option
text has the highest `SequenceMatcher` similarity.  These PathMMU-trained
checkpoints emit `<think>...</think><answer>...</answer>` and often discuss
distractors.  Whole-response matching can therefore disagree with an explicit
final option.

- Stage2 contains 320 cases where an explicit `<answer>A-D` choice equals the
  target but the frozen official mapper is wrong.  SFT3000 has 312 and SFT4000
  has 372.
- Example: target `D: CT`; Stage2 ends with `<answer>D) CT</answer>`, while the
  whole-completion mapper selects `A: Electrocardiogram (ECG)`.
- Another example has target `C: CT` and `<answer>C) CT</...>`, but the mapper
  selects `B: MRI`.

The low result is not solely a scoring artifact:

- Explicit-choice accuracy where an explicit choice is available is only
  44.73% (SFT3000), 44.52% (SFT4000) and 45.72% (Stage2).
- `2488/8518` examples are formally wrong for all three checkpoints.
- Stage2 formal accuracy varies strongly by source: Chest CT 30.42%, diabetic
  retinopathy 45.64%, ISIC2020 67.15%, retinal OCT 44.00%.

The source split and large all-three-wrong set indicate substantial true
out-of-distribution visual/domain error, especially for CT and retinal
modalities.  Stage2 was trained to improve PathMMU reasoning/reward compliance,
not to add CT, fundus, OCT or dermoscopy visual knowledge; the vision modules
were frozen.  It is therefore unsurprising that Stage2 barely changes
OmniMedVQA relative to either SFT checkpoint.

## Recommended reporting and training response

1. Keep the frozen scores as the primary table for reproducibility.
2. Add separately labelled answer-parser diagnostics and scorer-conflict
   counts; never relabel them as official accuracy.
3. For Stage3 strategy, prioritize concise final-answer compliance, but do not
   expect a text-only/frozen-vision reward stage to repair the true CT/OCT/
   fundus/dermoscopy domain gap.
4. Inspect the exported all-three-wrong and parser-conflict JSON files by
   modality when selecting qualitative cases for the reviewer response.

