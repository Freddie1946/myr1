# GPT-4o parallel smoke task gate and partial continuation

Recorded at: `2026-08-05 22:15:24 CST`

## Three-task smoke outcome

The first shared-GPU GPT-4o selected-model evaluation attempt ran the fixed 16-case behavior smoke
for PathMMU, PathVQA yes/no and OmniMedVQA. It stopped before every full evaluation because the
OmniMedVQA smoke failed the predeclared aggregate adapter gate:

- PathMMU: nonempty 100%, parseable 100%, cap-hit 0%; passed.
- PathVQA yes/no: nonempty 100%, parseable 93.75%, cap-hit 18.75%; passed.
- OmniMedVQA: nonempty 100%, cap-hit 93.75% (15/16); failed the maximum-20% gate.

The failure is preserved under
`pathvlm_revision_eval_a100/runs/stage3_selected_gpt4o_full_20260805_parallel_grok43`.
No full PathMMU, PathVQA or OmniMedVQA inference was started in that attempt. The 64-token contract
is not changed after observing these test-task outputs, and OmniMedVQA remains missing with an
explicit response-contract caveat.

## Partial continuation rule

The established local-baseline policy gates compatibility independently by model and task. It
previously allowed eligible Llama and DeepSeek tasks to continue while retaining incompatible
tasks as missing. The Stage3 selected-model launcher is now aligned with that policy:

- `PATHVLM_STAGE3_EVAL_TASKS` explicitly selects a unique subset of `pathmmu`, `pathvqa`, and
  `omnimedvqa`;
- the default remains all three tasks;
- GPU admission, smoke specs, full specs, the run contract and `completed.json` include only the
  requested tasks;
- the planned continuation uses a new directory and only `pathmmu,pathvqa`;
- it reruns their fixed smokes rather than copying gates from the failed aggregate attempt.

This is not dataset selection by accuracy. Both retained tasks passed the predeclared output-
behavior gate; OmniMedVQA failed before full inference and is not converted into a score.

## Concurrent visual arm and owner health

The independent GPT-4o Stage3 24-case perturbation-fidelity arm started on shared GPU 3. Its model
load raised GPU-3 memory from about 20.7 GiB to about 41.8 GiB, leaving roughly 39.3 GiB free.
Grok Stage3 retained its eight workers and reached step 23 without OOM or Judge failure. The visual
arm is not claimed complete by this record.

## Verification and hashes

- Shell syntax passed.
- Five shared-admission/task-selection static tests passed.
- Thirteen external-VQA scope and artifact-integrity tests passed.
- Failed-run contract SHA-256:
  `99bea22454e2ae05e8dca73791b8393dd16d4d79bf06e36fdba227fadc46c10c`.
- Omni smoke metrics SHA-256:
  `52ef8b21d98068641282e658997aa78593b829da7c1a6bff491a585dbbcba794`.

No open-answer PathVQA work, hosted baseline resumption or score correction is authorized here.
