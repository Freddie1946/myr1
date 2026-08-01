# Hosted PathMMU full evaluations started

Recorded at: `2026-08-01T19:15:24+08:00`

After the fixed behavioral smoke and scoring correction passed, three non-GPU AIGCBest PathMMU test999 evaluations were started concurrently with formal GPT-4o Stage 3:

- exact `qwen-vl-plus`;
- exact `claude-haiku-4-5-20251001`;
- exact `grok-4.3`, labelled as a contemporary replacement/additional baseline and not as the historical Grok-4-Fast row.

Each evaluation uses two API workers, deterministic temperature zero, a 1024-token cap, strict requested/served identity, raw-response retention, bounded explicit-transient retries and no retry after ambiguous transport failure. The frozen passed smoke summary is a hard launch prerequisite. The runner is resumable by immutable case ID and writes each result with an fsync before advancing.

Early launch observation showed 22 total successful records across the three models, exact served identities, no retry, no failure and a parseable PathMMU choice for every response. This early observation is a transport/adapter health check, not an accuracy result.

Evidence root:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/hosted_baseline_full_20260801`

Doubao, historical Grok-4-Fast and Llama 3.2 Vision were not launched because their identity/endpoint smokes failed. PathVQA and OmniMedVQA full jobs have not started yet; they remain next after the PathMMU hosted jobs demonstrate stable progress.
