# PathMMU test999 available baseline diagnostics completed

## Result scope

All currently runnable local baselines with an approved, validation-gated scoring contract have
completed on the 999-record corrected PathMMU v2 diagnostic split. These scores and bad cases may
inform Stage-3 design, so they are development diagnostics and are not eligible for an untouched
final Stage-3 claim. PathVQA and OmniMedVQA remain sealed.

No training or Stage-3 run was started.

## Generative VQA results

| Rank | Model | Correct | Accuracy | Choice extracted | Strict format | Cap hits |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Lingshu-7B | 583/999 | 58.36% | 999 | 989 | 0 |
| 2 | InternVL3-8B | 553/999 | 55.36% | 997 | 875 | 1 |
| 3 | HuatuoGPT-Vision-7B | 531/999 | 53.15% | 961 | 76 | 0 |
| 4 | Qwen2.5-VL-7B-Instruct | 485/999 | 48.55% | 979 | 965 | 0 |
| 5 | Qwen2.5-VL-3B-Instruct | 449/999 | 44.94% | 998 | 963 | 0 |
| 6 | MedVLM-R1 | 413/999 | 41.34% | 998 | 981 | 0 |
| 7 | MedGemma-4B-IT | 400/999 | 40.04% | 840 | 28 | 24 |

The answer score and strict-format score must be read separately. Huatuo and especially MedGemma
often gave natural-language responses without the full required tags. MedGemma additionally had
159 rows without an extractable choice and 24 generation-cap hits, so its measured answer score is
more affected by response-contract failure than the other models.

## Pathology image-text matching results

These are not generative VQA models. Each selected the option whose raw text had the greatest
normalized image-text similarity:

| Model | Correct | Accuracy | Choice extracted |
|---|---:|---:|---:|
| CONCH | 338/999 | 33.83% | 999 |
| PLIP | 334/999 | 33.43% | 999 |

UNI remains a visual-only representation baseline and has no approved multiple-choice VQA scoring
contract.

## Joint bad-case package

The final available-nine report is:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathmmu_test999_badcases_available9_20260728_2252`

Across the seven generative and two matching baselines:

- all nine wrong: 55;
- exactly one model correct: 88;
- all nine correct: 15;
- prediction disagreement or parse failure: 983;
- correct-count histogram from zero through nine:
  `55, 88, 138, 121, 169, 141, 113, 107, 52, 15`.

The 88 sole-correct cases are distributed rather than belonging to one model:

- PLIP 13, CONCH 13, Qwen3 12, Huatuo 11, MedVLM-R1 10;
- MedGemma 9, InternVL 7, Lingshu 7 and Qwen7 6.

This distribution argues against using only the strongest baseline as a Stage-3 teacher. The most
useful manual-review order is:

1. the 55 all-wrong cases, to find shared visual/knowledge failures or questionable targets;
2. the 88 sole-correct cases, to identify complementary visual or medical cues;
3. format/parse failures, kept separate from semantic errors;
4. Lingshu-versus-InternVL/Huatuo disagreement cases, because those are the three strongest
   generative baselines.

Artifact hashes:

- `summary.json`:
  `a5949622331aaf1e5ae80cfc1a8a9c8a1bd915e687095b7e53e05c346cf74418`;
- `badcases.jsonl`:
  `f9364af7dd32a1ce66d8da043683de99743df307dcfb5efa78625c0d272c5319`;
- `badcase_index.csv`:
  `bd5bf01820e5bac6a38ade1392c881e07cbf58ed160fc11cd5ab9f7ffa003640`;
- `artifact_manifest.json`:
  `ad95372d960cebd1a9120277f986f8b940c185d0968fabb520c47feb7fcd13b1`.

The completion ledger with every score and prediction/metrics hash is
`protocol/pathmmu_test999_available_baselines_completion_manifest_20260728_225929.json`.

## Unscored local candidates and external gates

- DeepSeek-VL2 and LLaVA-Med both ran in BF16 and generated nonempty natural-language answers on
  validation, but omitted extractable option letters. They were stopped before test999 rather than
  being misreported as zero-accuracy models or receiving a post-test model-specific semantic
  parser.
- Meta Llama 3.2 Vision 11B/90B weights remain gated and unavailable.
- Hosted API models remain deferred by the user.

## Stage-2 diagnostic remains blocked

The requested validation-selected Stage-2 Outcome-GRPO score cannot yet be produced because the
exact checkpoint is absent from this machine and the available Hugging Face account. The required
source is:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage2_outcome_grpo/priority_n1000_seed0042/stage2_priority_n1000_seed0042_20260720_035156/formal_n1000_seed0042_epoch3/epoch_snapshots/checkpoint-1000`

Its snapshot-manifest SHA-256 is
`83df2570a33bd760bebb6ef8afca175b71c33bb585e107cdb2f3889edebc04e0`.
No substitute checkpoint is permitted. Once transferred and verified, it should run through the
same frozen generative diagnostic and be appended to the joint report.
