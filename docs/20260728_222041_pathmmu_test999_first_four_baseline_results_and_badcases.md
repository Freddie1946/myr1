# PathMMU test999 first four baseline results and bad cases

## Scientific role

These runs are pre-Stage-3 development diagnostics authorized for score inspection, bad-case
analysis and Stage-3 strategy adjustment. They accessed the corrected PathMMU v2 test999 split and
are therefore not eligible for an untouched final Stage-3 claim. PathVQA and OmniMedVQA remain
sealed for independent post-Stage-3 confirmation.

## Frozen execution

- data: `test_0999.json`, 999 records, SHA-256
  `3420c8ff2f1642801a5ecebd21271f4e7de5877d6459814473ddbcdda996b549`;
- greedy decoding, `do_sample=false`, BF16, no quantization and 1024 maximum new tokens;
- common image-grounded reasoning and answer-tag instruction;
- current conservative parser:
  `scripts/pathmmu_rewards.py`, SHA-256
  `69c1bde669ee16de35dd8680f6f4590429e3830617498106be6b5d333664a56c`.

## Completed scores

| Model | Correct | Accuracy | Choice extracted | Strict format |
|---|---:|---:|---:|---:|
| Lingshu-7B | 583/999 | 58.36% | 999/999 | 989/999 |
| Qwen2.5-VL-7B-Instruct | 485/999 | 48.55% | 979/999 | 965/999 |
| Qwen2.5-VL-3B-Instruct | 449/999 | 44.94% | 998/999 | 963/999 |
| MedVLM-R1 | 413/999 | 41.34% | 998/999 | 981/999 |

Offline rescoring of the saved completions with the current parser changed none of the four scores.
Accuracy and strict response-format compliance are retained as separate measures.

## First joint bad-case strata

Across the four completed models:

- all four correct: 145;
- all four wrong: 162;
- exactly one correct: 232;
- exactly two correct: 262;
- exactly three correct: 198;
- prediction disagreement or a parse failure: 812;
- at least one parse failure: 22;
- at least one strict-format failure: 90.

Important complementarity counts include:

- Qwen2.5-VL-7B correct while Lingshu is wrong: 121;
- MedVLM-R1 correct while Lingshu is wrong: 127;
- Qwen2.5-VL-3B correct while Lingshu is wrong: 147.

These are descriptive pre-Stage-3 diagnostics, not the final paired inference for the eventual
Stage-3 model. The 162 all-wrong cases and 232 one-model-only cases are the primary candidates for
manual error typing and Stage-3 reward-policy discussion.

## Artifacts

Report root:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathmmu_test999_badcases_available4_20260728_2224`

- `summary.json`: SHA-256
  `35bde379edc5eb561065093ac856ce6c85314d3ccad5f3d06b2a251e6ff65b3b`;
- `badcases.jsonl`: SHA-256
  `0b485c3a865dbaa64a68d2f031b7268b1b0c924fd644281f1a4b61638fc36d46`;
- `badcase_index.csv`: SHA-256
  `7f63e02a9631b076a7f64d35b86f89863929b20579236e523398936e4bc87f50`;
- `artifact_manifest.json`: SHA-256
  `a297ea6a5ffb7e7303b1971c3c0abb10e1bfe9b6b766c8f485eeafb0cdbf60ea`;
- builder: `scripts/build_pathmmu_badcase_report.py`, SHA-256
  `d0c5a0fbc29ecd2a91078603c3f99b10805583ee99e4bd9acd31bc8770365dea`.

The bad-case JSONL retains each model's raw completion. The compact CSV is suitable for sorting and
manual annotation.

## Still pending

- MedGemma, InternVL and HuatuoGPT-Vision full diagnostics are running.
- DeepSeek-VL2 BF16 inference passed the hardware gate but failed the fixed response-contract smoke:
  it returned a nonempty natural-language answer without a choice letter, so no test999 run was
  launched.
- The validation-selected Stage-2 Outcome-GRPO checkpoint is not present on this machine or in the
  available Hugging Face account. Its diagnostic remains blocked on exact checkpoint transfer.
- No Stage-3 training has started.
