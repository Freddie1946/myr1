# All manuscript baselines are mandatory

Recorded: `2026-07-25T01:15:28+08:00`

This is a scope correction and preparation record with `formal_result: false`. No model weight was
downloaded, no environment was changed, no inference or training was run, no GPU was allocated, and
no validation/test result was inspected.

## User decision

Every baseline named in the submitted manuscript is mandatory for the revision rerun. Cost, model
size, environment complexity, or the absence of trustworthy historical raw outputs is not a reason
to omit a row.

This supersedes only the recommendation in
`20260725_010533_baseline_access_and_environment_status.md` to triage the manuscript models into a
smaller mandatory subset. It does not rewrite that historical audit or change its factual
environment-status findings.

## Frozen manuscript baseline set

Table II contributes fourteen baselines:

1. Doubao-1.5-vision (correcting the manuscript typo `vison`);
2. Qwen-VL-Plus;
3. DeepSeek-VL2;
4. Claude-3.5-Haiku;
5. Grok-4-Fast;
6. LLaMA3.2-90B;
7. Claude-4.5;
8. LLaMA3.2-11B;
9. MedVLM-R1;
10. MedGemma-4B;
11. Qwen2.5-VL-3B;
12. Qwen2.5-VL-7B;
13. Lingshu-7B;
14. InternVL3-8B.

Table IV adds:

15. HuatuoGPT-Vision-7B.

PLIP, CONCH, UNI and any separately approved pathology/medical VLM remain an additional
reviewer-response comparison group. They do not replace any of the fifteen manuscript baselines.

## Reproducibility status

Only the pinned Qwen2.5-VL-7B baseline is currently runnable end to end on the formal machine.
The other nine local/open-weight manuscript baselines require exact snapshot selection, isolated
environment preparation, local weight hashes, model-specific image/message adapters, CPU import
checks and GPU smoke checks. The five hosted/API baselines require fresh credentials outside Git and
an exact provider/model-version contract.

The historical evaluation scripts do not identify exact provider versions for the five API rows and
do not contain complete trustworthy raw-output provenance. In particular, the manuscript label
`Claude-4.5` does not identify a Claude family member or dated API model. This must be resolved from
author records or explicitly corrected; it must not be guessed.

Two hosted rows also have temporal-identity hazards as of this audit:

- xAI documents that the old Grok-4-Fast slugs are retired and redirected to a newer model. A new
  request using an old slug therefore is not automatically an exact historical rerun.
- Volcengine documents service changes affecting Doubao-1.5-vision. Current availability and exact
  endpoint/version must be captured before evaluation.

If the historical exact endpoint is unavailable, the row will still be run with the closest
officially documented current successor after an explicit protocol decision, but will be labeled
as a contemporary rerun rather than an exact recovery.

## Evaluation contract still to freeze

All fifteen rows will use the same question/options presentation, image input policy, deterministic
decoding where supported, generation cap, answer parser and raw-prediction schema. Provider-side
limitations will be recorded rather than silently normalized away.

The manuscript reports 500 examples per Table II row, but the exact historical 500 sample IDs were
not recovered. Those old numbers therefore cannot be claimed as exactly reproduced. The revision
rerun will use a newly frozen common evaluation set. Validation remains the only place for prompt,
parser and configuration selection; the formal PathMMU test remains sealed until every baseline
contract is frozen and smoke-tested.

For each run, retain at minimum:

- exact model repository/API ID and revision or dated endpoint;
- license/access state and execution provider;
- environment lock and code hashes;
- prompt, image preprocessing, decoding parameters and parser version;
- ordered sample IDs, seed where applicable, timestamps and retry/error events;
- every raw response and parsed answer;
- aggregate metrics only after the immutable raw output is saved.

## Environment families

Separate environments are required where dependency contracts differ:

- Qwen-family environment: Qwen2.5-VL-3B/7B, subject to compatibility checks for MedVLM-R1 and
  Lingshu;
- Meta Llama 3.2 Vision environment: 11B and 90B;
- isolated DeepSeek-VL2 environment;
- isolated MedGemma environment;
- isolated InternVL3 environment because it uses model-specific code;
- isolated HuatuoGPT-Vision/LLaVA-family environment;
- hosted-model client environment for Doubao, DashScope/Qwen, Anthropic and xAI, with credentials
  loaded only from ignored environment files;
- existing pathology-encoder environment for PLIP/CONCH/UNI.

No model will be forced into an incompatible shared environment merely to reduce setup work.

## Storage gate

The exact snapshot inventory must be obtained before weight downloads. The 90B BF16 model alone is
on the order of 180 GB, and DeepSeek-VL2 is a 27B MoE model; together with the remaining snapshots,
caches and raw outputs, the complete local set can consume several hundred GB. The preparation audit
found roughly 506 GB free under `/home`, below the existing 550-GiB formal-training reserve.

Therefore this decision authorizes the exhaustive preparation plan, but not bulk weight download.
Before downloading, freeze exact byte totals and choose a storage solution that does not endanger
formal checkpoints. Quantized or substituted weights may not be used silently; any such change
requires an explicit comparability decision and disclosure.

## Next actions

1. Recover or resolve the exact identity of every manuscript baseline, especially the five API rows.
2. Record authoritative model IDs, revisions, access state and exact snapshot byte totals.
3. Build the isolated environments and adapters without opening formal test data.
4. Obtain fresh provider credentials outside Git and gated-model approvals.
5. Download/hash weights only after the storage gate is approved.
6. Run CPU imports and small GPU smoke jobs after checking shared-server allocation.
7. Freeze the common validation/test protocol, validate configurations only on validation, and run
   each mandatory baseline on test once.

