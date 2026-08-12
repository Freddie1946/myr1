# OOD evaluation-method audit and semantic recheck plan

Date: 2026-08-11 (Asia/Shanghai)

## Audit result

The previous OOD experiments did not all use next-token logits. The methods and required actions
are different by benchmark.

| Benchmark/result family | Actual inference method | Cap / coverage | Scientific status | Action |
|---|---|---|---|---|
| Formal L/A PathVQA validation and test Yes/No | Forced next-token `Yes` versus `No` logits | 100% binary coverage | Useful decision-boundary and visual-dependence diagnostic; insufficient as the sole semantic-quality result | Keep as diagnostic; add cross-format free-generation probe, then use a validated generated-answer protocol for final Base/L/final-RL comparison |
| Earlier Stage3 PathVQA corrective result | Actual short-answer generation, deterministic plus LLM intent recovery | 192-token corrective rerun; 98.84% resolved | Better than logits for emitted answer intent, but prompt does not request PathMMU-style reasoning and therefore does not test explanation semantics | Keep as sensitivity result; add the same cross-format semantic panel |
| Historical transferred SFT3000/Stage2 PathVQA | Actual short generation | 64-token cap; historical SFT has 1,221/6,719 cap hits | Prompt/cap contaminated and exact-match dominated | Do not use for semantic conclusions; rerun only the paper-critical final models with the new protocol |
| PathMMU validation/test | Actual generation with `<think>...<answer>letter</answer>` | 1,024 tokens; zero cap hits for formal L checkpoints | Valid generated-answer accuracy and format evidence | No method-driven rerun required |
| MMMU dev116 | Actual free generation and answer parsing | 4,096 tokens; L step80 has four cap hits and 111/116 parsed | Valid as a small retention sanity panel, with explicit failed-generation caveat | Retry/constrain the five unresolved outputs; no need to replace the whole panel with logits |
| Historical transferred SFT3000 OmniMedVQA | Actual option-answer generation | 64 tokens; 6,200/8,518 = 72.79% cap hits | Not a clean semantic or final accuracy result | Must rerun if this checkpoint remains paper-critical |
| Corrective Stage3 GPT-4o OmniMedVQA | Actual option-answer generation and strict final-option extraction | 192 tokens; zero cap hits; 8,516/8,518 strict answers | Valid option-choice accuracy, not an explanation-quality evaluation | Keep accuracy; add only a small explanation/grounding probe if reasoning quality is claimed |
| API closed-model baselines | Actual API responses followed by task parsing/judging | Model-specific retries and audits | Not next-token logits | Retain per-run parser/cap audits |

## Reinterpretation

Forced PathVQA logits should not be deleted. They provide a clean measurement of the model's
binary decision boundary and support shuffled/blank-image controls without parser confounding.
They should be downgraded from “complete OOD semantic performance” to “binary-interface retention
diagnostic.” A final paper claim about semantic pathology transfer must be supported by actual
generated reasoning and answers.

## Frozen cross-format panel

Before generating any new outputs, a deterministic 64-case target-Yes panel was frozen from the
complete PathVQA test paired results:

- 32 Base-correct/L-SFT-wrong regressions;
- 16 Base-wrong/L-SFT-correct improvements;
- 16 Base-correct/L-SFT-correct controls;
- 64 unique images;
- regression strata include all five stain/marker candidates, all eight target-Yes morphology
  candidates, seven image-diagnosis, seven short-presence, and five other cases.

Panel:
`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_cross_format_semantic_panel_v1_20260811/panel.json`

Panel SHA256: `37717628ab092464c36082291e8c7cd7fd8bf19f6c6ead5a10870507f1b657e1`.

The target-Yes restriction is deliberate: it permits the positive statement to be transformed
into a diagnosis/finding MCQ without inventing an unknown true diagnosis for a target-No record.
Target-No regressions can be audited later with generated binary reasoning and open feature
descriptions, but should not receive synthetic “correct diagnosis” distractors without a verified
reference.

## Three probes per model

Base and L-SFT step80 will each receive the same three deterministic probes, with a 2,048-token
safety limit and explicit cap/EOS/format records:

1. **Generated binary reasoning:** answer the original question using PathMMU-style
   `<think>...</think><answer>Yes/No</answer>` output.
2. **Feature-only open response:** describe visible pathological features before naming any
   diagnosis; score feature correctness, omissions, hallucinations, and image grounding.
3. **Four-choice diagnosis/finding MCQ:** use the reference-positive finding plus three plausible,
   same-organ or same-modality distractors with counterbalanced option order; generate reasoning
   and a final option letter.

This produces 64 x 2 models x 3 probes = 384 local generations. Exact scoring is used for emitted
binary/MCQ answers. Only the open feature response requires semantic judging. Outputs must be
blinded by model identity; an external multimodal judge may score the feature response, followed
by manual visual audit of disagreements and all stain/marker cases.

## Interpretation matrix

| Generated binary | Open features | MCQ | Interpretation |
|---|---|---|---|
| Wrong | Correct | Correct | Binary-interface drift; pathology semantics retained |
| Wrong | Correct | Wrong | Diagnosis/answer mapping drift despite feature recognition |
| Wrong | Wrong | Wrong | Strong candidate for genuine visual-semantic regression |
| Correct | Correct | Correct | Forced-logit result was not representative of emitted semantic behavior |
| Correct | Wrong | Correct | Possible shortcut/guess; final choice is not supported by image-grounded reasoning |

The improvement group is analyzed symmetrically. If its open/MCQ gains are genuine, SFT both
acquires and loses pathology skills. If improvements are mostly boundary-only while regressions
retain correct open/MCQ semantics, the aggregate PathVQA drop is primarily an interface artifact.

## Execution order

1. Do not interrupt the active corrected rule-RL run.
2. While GPUs are occupied, freeze the panel, manifests, prompts, distractor-generation contract,
   and blinded judge schema.
3. After GPUs are free, run Base and L step80 on the 64-case panel first.
4. Validate the semantic scoring protocol before any full rerun.
5. If the probe demonstrates that generated binary/MCQ accuracy materially disagrees with forced
   logits, rerun the full PathVQA Yes/No subset only for Base, L step80, and the finally selected
   corrected-RL checkpoint. Do not rerun every screening checkpoint.
6. Separately rerun historical 64-token OmniMedVQA/PathVQA only for models that remain in the final
   paper table.
