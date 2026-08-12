# Formal L-SFT3000 out-of-domain bad-case analysis

Date: 2026-08-11 (Asia/Shanghai)

## Scope and headline conclusion

This analysis compares the frozen Base model with the formal L-SFT3000 step-80 checkpoint. It
does not use the invalid pre-fix rule-RL checkpoints. The primary medical-OOD set is the complete
PathVQA test Yes/No subset (3,362 questions); the non-medical sanity set is MMMU dev116. PathVQA
is scored from forced Yes/No logits with 100% coverage, so its decline cannot be attributed to
free-generation parsing or truncation.

The PathVQA decline is real at the binary decision level, but the evidence does not support a
simple claim that the model broadly forgot pathology knowledge. Most of the net loss is explained
by a task-format and decision-policy shift concentrated in two short positive-question templates.
There is a smaller component of genuine fine-grained visual-semantic regression. Visual usage is
slightly weaker but has not collapsed. MMMU shows both regressions and improvements; four of its
16 Base-correct/SFT-wrong flips are explicit no-answer or long-generation failures, while the
remaining cases contain factual, arithmetic, and reasoning errors.

`Forced Yes/No` directly compares the model's scores for the two semantic answers. It does not
generate or inspect a rationale. Consequently, a wrong forced-binary choice is a wrong final
semantic decision under the dataset label, not a parser error; however, it does not reveal whether
the hidden reasoning was correct before the final token boundary moved. Claims about explanation
quality require a separate free-generation audit.

## 1. PathVQA: exact paired outcome

| Outcome | Count |
|---|---:|
| Base correct, SFT correct | 2,012 |
| Base wrong, SFT wrong | 1,040 |
| Base correct, SFT wrong | 220 |
| Base wrong, SFT correct | 90 |

Base obtains 2,232/3,362 = 66.39%; L step 80 obtains 2,102/3,362 = 62.52%, a net loss of 130
questions (-3.87 pp). The 220 regressions span 191 distinct images, so the loss is not caused by
one duplicated image cluster. Of these regressions, 189 have target `Yes` and only 31 have target
`No`.

## 2. Dominant cause: binary task/template policy drift

The exclusive heuristic template breakdown is:

| PathVQA template | N | Base | L step 80 | Delta | Regressions / improvements |
|---|---:|---:|---:|---:|---:|
| `is X present?` | 1,507 | 65.43% | 58.93% | **-6.50 pp** | 138 / 40 |
| `does this image show X?` | 605 | 57.69% | 53.55% | **-4.13 pp** | 35 / 10 |
| Stain / marker questions | 71 | 69.01% | 61.97% | **-7.04 pp** | 5 / 0 |
| Morphology / architecture | 368 | 70.11% | 71.20% | +1.09 pp | 14 / 18 |
| Disease / diagnosis (other wording) | 194 | 72.68% | 71.65% | -1.03 pp | 6 / 4 |
| Negation / absence | 31 | 45.16% | 45.16% | 0.00 pp | 2 / 2 |
| Other | 586 | 74.23% | 73.55% | -0.68 pp | 20 / 16 |

The first two templates account for 173/220 = 78.6% of all regressions and 123/130 = 94.6% of
the net accuracy loss. This is the strongest quantitative localization of the failure.

The `does this image show X?` subset is 99.34% target-Yes. The L checkpoint becomes slightly more
conservative on this positive-caption template and therefore loses 25 net questions. On the
`is X present?` subset, the predicted-Yes rate falls from 41.47% to 30.59%, while the target-Yes
rate is 54.55%.

This is mechanistically consistent with the training mismatch. All 3,000 SFT records are long
PathMMU A/B/C/D multiple-choice questions (mean 39.4 prompt words), with balanced target letters
(`A=754`, `B=747`, `C=753`, `D=746`) and `<think>...<answer>letter...</answer>` outputs. There is
no direct Yes/No target training. Language LoRA, including the LM-head adapter, can therefore
improve the A/B/C/D task while moving the relative Yes/No token boundary.

### Concrete examples

1. **Organ/system-presence policy shift:** source 4136, `is liver present?`, target Yes. Base target
   margin is +0.25; L step 80 changes it to -1.375. The image contains large polygonal hepatocyte-
   like cells with a dense inflammatory infiltrate. This is a plausible true regression rather
   than a parser failure. Related net losses are broad: hepatobiliary -11, endocrine -10, nervous
   -9, cardiovascular -8, liver -8, and respiratory -7 questions.
2. **Positive image-caption prior:** source 1623, `does this image show Hashimoto's thyroiditis?`,
   target Yes. The margin changes from +0.25 to -0.875. The low-power image contains residual
   colloid-bearing thyroid follicles amid dense lymphoid follicles, so the SFT `No` is visually
   difficult to justify. This is evidence of a real decision error induced by the shifted policy.
3. **Same image, multiple policy failures:** image
   `56f9d724...c9350.jpg` produces three Base-correct/SFT-wrong questions: mild-to-moderate acute
   rejection, hepatobiliary present, and liver present. This pattern is more consistent with an
   image-to-answer decision-policy shift than with one isolated forgotten fact.

## 3. Confidence compression, not a simple uniform `No` bias

Across all target-Yes questions, the mean Yes-minus-No margin moves from +0.217 to -0.070. Across
target-No questions it moves from -1.804 to -1.502. Thus both sides move toward the decision
boundary. Because Base's positive margins were already much smaller than its negative margins,
this compression causes many more Yes answers to cross into No than the reverse. At dataset level,
the predicted-Yes rate falls from 41.64% to 35.93%, and mean target margin falls from 0.9469 to
0.6530.

The individual flips are mostly near the boundary: among 220 regressions, only nine have an SFT
wrong-answer margin at least 1.0, and none reaches 2.0. This argues against widespread high-
confidence destruction of medical knowledge. It instead indicates many borderline decisions
being moved by narrow-task calibration.

## 4. Secondary cause: genuine fine-grained visual-semantic transfer failures

The main policy effect does not explain every case. Several images require recognizing subtle
features or mapping them to a specific statement, and SFT reverses a previously correct choice.

1. **Cartilage/bone interface:** source 3079 asks whether conventional chondrosarcoma entraps
   native lamellar bone as a confluent cartilage mass. The margin changes +0.375 to -0.75. The
   low-resolution image contains pale cartilage-like lobules surrounding bright eosinophilic
   bone, so this is a plausible fine-grained visual-semantic regression.
2. **Epithelial atypia:** source 5668 asks whether the high-power field shows failed normal
   differentiation, marked pleomorphism, and numerous mitoses toward the surface. The margin
   changes +0.50 to -0.50. The image visibly contains stratified atypical epithelium with enlarged,
   irregular nuclei; the new No decision cannot be explained by output formatting.
3. **Special stain:** source 1725, `does iron stain?`, changes +0.625 to -0.25. The image is pale and
   the positive material is sparse/low-contrast. This is the kind of stain and resolution shift
   for which a frozen vision stack plus narrow PathMMU language adaptation may be brittle.

The aggregate evidence nevertheless rejects a global visual-collapse explanation. On the frozen
PathVQA validation panel, Base normal-minus-shuffle is 15.43 pp and L step 80 is 14.06 pp, a modest
1.37-pp reduction; normal-minus-blank is 15.04 pp for Base and 15.43 pp for L step 80. Among 25
Base-correct/SFT-wrong validation cases, 18 (72%) retain the same wrong SFT answer under both blank
and shuffled images, supporting a language/decision prior in most regressions. The model still
uses images overall.

## 5. Dataset ambiguity and label/text-image noise

Some apparent regressions cannot cleanly measure model knowledge because the legacy PathVQA
question is malformed, abstract, or poorly paired with the image.

1. Source 453 asks whether an `opened abdominal cavity with massive tumor in omentum ...` is seen,
   with target No. The supplied image is a histology field containing infiltrative glands in
   desmoplastic stroma, not a gross abdominal-cavity photograph. Base gives the correct No with
   margin +1.875; SFT gives Yes with target margin -0.8125. SFT is wrong under the dataset label,
   but this is primarily a question-image contract problem rather than a clean pathology test.
2. Questions such as `is hepatobiliary present?` and `is endocrine present?` ask for a broad
   ontology label rather than a specific visible lesion. They can be answered from dataset style
   as much as from morphology, making them particularly sensitive to answer-prior drift.
3. Source 3315 asks `does exostosis show i am not sure of diagnosis?`. The grammatical and semantic
   target is intrinsically unclear. A flip here should not be interpreted as forgotten exostosis
   knowledge.

These examples mean expert adjudication should flag noisy cases separately. They do not erase the
paired decline, because the loss spans 191 images and is statistically directional, but they do
limit literal clinical interpretation of individual PathVQA labels.

## 5a. SFT also produces genuine OOD improvements

The direction is not one-way: 90 PathVQA questions change from Base-wrong to SFT-correct. Of these,
62 have target No and 28 have target Yes. Several are visually and semantically plausible gains:

1. Source 14 asks whether AIDS-associated massive *Mycobacterium avium* infection has inadequate
   T cells to mount a granulomatous response. Base answers No; SFT answers the target Yes. The
   acid-fast image contains numerous bright magenta bacilli with little organized granulomatous
   structure, so the SFT decision is a meaningful pathology improvement.
2. Source 827 asks whether the image shows right-coronary atherosclerosis with acute thrombus.
   Base answers No; SFT answers the target Yes. The gross cross-section visibly contains a dark
   occlusive focus in the coronary lumen.
3. Source 6547 asks whether the bowel lumen shows characteristic transverse ulcers and two
   strictures. Base answers No; SFT answers the target Yes. The gross panels show narrowed segments
   and transverse ulceration marked by arrows.
4. Source 4447 asks whether deposits are basophilic/granular with healed granulomas at the
   periphery. Base answers No; SFT answers the target Yes. The slide contains conspicuous dark-blue
   granular deposits in a fibrotic region.

The correct conclusion is therefore a redistribution of capability: SFT fixes some pathology
decisions, especially some specific morphology/negative-rejection cases, but creates more new
errors than it repairs on this OOD benchmark (220 versus 90).

## 6. Training exposure shows narrow-domain specialization

| Checkpoint | Approx. epoch | PathMMU test999 | PathVQA test Yes/No |
|---|---:|---:|---:|
| Base | 0 | 48.65% | 66.39% |
| L step 16 | 0.5 | 52.25% | 65.44% |
| L step 64 | 2.0 | **57.56%** | 62.70% |
| L step 80 | 2.5 | 56.66% | 62.52% |
| L step 96 | 3.0 | 56.66% | 62.46% |

The dose-response pattern is important: early SFT acquires most of the domain gain with only a
small retention cost, while later exposure adds little PathMMU performance and worsens PathVQA.
This supports overspecialization to the PathMMU task/output distribution rather than insufficient
pathology knowledge as the primary cause.

The formal A-SFT3000 control also argues against blaming projector freezing. Training the projector
does not restore retention: A step 96 obtains 62.17% on PathVQA test, slightly below L step 96 at
62.46%, while its visual-dependence controls are no better. The shared cause is therefore the
narrow PathMMU SFT objective/exposure, not uniquely the L architecture.

## 7. MMMU non-medical sanity panel

Base scores 66/116 = 56.90%; L step 80 scores 62/116 = 53.45%. Paired outcomes contain 16
regressions and 12 improvements (exact two-sided sign/McNemar binomial p approximately 0.572), so
the net four-question loss is not precise evidence of broad catastrophic forgetting.

The regressions nevertheless reveal two useful failure classes:

### A. Explicit output/generation failures: 4/16 regressions

- `dev_Electronics_3`, `dev_Electronics_5`, and `dev_Mechanical_Engineering_5` hit the 4,096-token
  ceiling after malformed/repeated HTML or image markup and produce no parsed choice.
- `dev_Architecture_and_Engineering_2` generates 2,709 tokens of repeated external image/markup
  content, ends without a valid answer, but does not reach the hard cap.

These four are formatting/generation-control failures, not demonstrated knowledge loss. Unlike
the historical 64-token runs, the 4,096-token ceiling is already generous; the required fix is a
constrained answer protocol or repetition stopping, not merely a still larger cap.

### B. Substantive reasoning or factual regressions: 12/16 regressions

- `dev_Chemistry_3`: Base uses the correct saccharin formula `C7H5NO3S`; SFT hallucinates
  `C9H7NO3S`, changes the limiting-reagent calculation, and selects A instead of B. This is a
  concrete factual/chemical-knowledge regression.
- `dev_Art_4`: Base identifies Violet Jacob's *The Wild Geese*; SFT substitutes *The Rowan* and
  selects D instead of B. This is a factual recall substitution.
- `dev_Marketing_4`: Base works through the ANOVA comparison; SFT stops at “the result cannot be
  concluded without a supplied p-value” and selects C instead of performing the requested test.
  This is premature reasoning/shortcut behavior.
- `dev_Physics_2`: Base derives the velocity from uniformly accelerated motion; SFT uses the
  average-speed shortcut `96/0.4` and selects E instead of B. This is a reasoning-method error.
- `dev_Accounting_5`: both attempt compound-interest calculations, but SFT's approximate values
  are mapped to the wrong option A rather than B. This is an arithmetic/option-mapping error.

There are also 12 Base-wrong/SFT-correct flips (including Biology, Literature, Design, Materials,
and Math), so MMMU reflects checkpoint churn and task-style drift rather than one-way loss of all
general knowledge.

## 8. Historical OmniMedVQA boundary

The available OmniMedVQA SFT result belongs to the older transferred full SFT3000 checkpoint
(`epoch03_step1125`), not the new formal L-SFT3000 checkpoint. It must not be merged into the L
analysis. In addition, that historical inference used `max_new_tokens=64` and 6,200/8,518 = 72.79%
of generations hit the cap. Although rescoring changes the reported accuracy to 44.88%, the run
cannot support a clean semantic bad-case attribution. A current-checkpoint rerun with a safe
generation contract is required before claiming an OmniMedVQA mechanism.

## Final causal assessment

The evidence ranks the causes as follows:

1. **Primary:** narrow-task/output-distribution specialization and Yes/No boundary compression
   caused by A/B/C/D CoT SFT on the language side.
2. **Secondary:** genuine fine-grained visual-semantic transfer errors on organ recognition,
   stains, and subtle morphology; visual reliance weakens slightly but does not collapse.
3. **Smaller but visible:** legacy PathVQA ambiguity and text-image mismatch.
4. **For MMMU only:** a mixture of real reasoning/factual drift and four explicit long-generation
   or missing-answer failures.

The most defensible wording is therefore not “SFT destroys pathology knowledge.” It is: **later
narrow PathMMU SFT exposure improves the in-domain multiple-choice task while shifting the binary
OOD decision policy and mildly narrowing multimodal/general transfer; a smaller subset shows true
visual-semantic and factual regressions.**

## Reward-alignment boundary

The reward-alignment defect occurred only in the subsequent old rule-RL run. The L-SFT3000
checkpoints and their step-16/64/80/96 evaluation curve were produced before GRPO and never used
the faulty reward implementation. The defect can explain why the old rule-RL did not reliably
improve in-domain performance and makes every old RL checkpoint scientifically invalid, but it
cannot explain the SFT-only PathVQA decline. The currently running corrected rule-RL is the first
valid experiment capable of testing whether properly aligned RL recovers domain performance while
preserving OOD retention.
