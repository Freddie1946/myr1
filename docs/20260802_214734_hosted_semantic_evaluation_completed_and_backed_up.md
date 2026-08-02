# Hosted semantic evaluation completed and backed up

Timestamp: `2026-08-02 21:47:34 CST`

## Outcome

The Qwen-VL-Plus and Claude Haiku 4.5 full PathVQA evaluations are now finalized under the frozen
mixed scoring contract. Yes/no cases use deterministic target-blind exact scoring; free-form cases
use the frozen GPT-5-mini semantic Judge. Duplicate semantic-content keys reuse one cached verdict
but every dataset case is scored separately. Failed generations remain in the denominator and are
wrong. No answer was manually corrected or re-Judged during finalization.

| Model | PathMMU test999 | OmniMedVQA | PathVQA mixed semantic |
|---|---:|---:|---:|
| Qwen-VL-Plus | 547/999 (54.75%) | 6151/8518 (72.22%) | 2507/6719 (37.31%) |
| Claude Haiku 4.5 | 466/999 (46.69%) | 4783/8518 (56.22%) | 1544/6719 (22.98%) |

For PathVQA, Qwen's yes/no accuracy is 67.58% and free-form semantic accuracy over all free-form
cases is 7.00%; Haiku's corresponding values are 40.30% and 5.63%. Qwen has 11 failed free-form
predictions and Haiku has 3; all are retained as wrong. Two historical Haiku judgment records tied
to failed predictions are disclosed and excluded. These are full authorized contemporary-baseline
results. PathMMU test999 remains a development diagnostic under the previously disclosed policy.

The finalizer made no paid request. Its tests plus the frozen Judge tests passed 12/12, and both
final artifacts passed all source/prediction/hash/identity/coverage/accounting gates.

## Immutable private-HF backup

The final inputs and outputs were frozen under
`snapshots/20260802_pathvqa_semantic_final/` in private dataset
`Freddie1946/PathVLM-R1-Revision-Evaluation-Results` at revision
`6dd3ef5a73d3b7603991adff7b2fd17a9883e390`.

The snapshot has 14 data files totaling 53,547,579 bytes plus `snapshot_manifest.tsv`. Its
manifest SHA-256 is
`f323c6d1f8d7f2623459df4b86ef58386a8b95923b29feb17f4e93975a05929a`.
All 14 files were downloaded again from that exact revision and verified byte-size and SHA-256
against the remote manifest. No credentials or model weights are present.

GPT-4o Stage3 remained active during the small backup; at the preceding 21:44 check it was at
step 1349/1500 with USD 56.9310975 committed cost and fallback 2 total/0 consecutive.
