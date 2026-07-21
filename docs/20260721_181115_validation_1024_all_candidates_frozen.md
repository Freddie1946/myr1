# Corrected 1024-token all-candidate validation frozen

Timestamp: `2026-07-21T18:11:15+08:00`

## Authorization and scope

The user authorized a complete rerun rather than selective reuse after the 192-token Base
truncation was diagnosed. This run performs deterministic validation inference only. It does not
train or modify a model, start Stage 3, access test, use `picked.json`, or select a favorable seed.

Exactly 26 frozen candidates will produce 10,010 raw validation records:

- pinned Base;
- SFT n=500 and n=1000 epoch-10 final models;
- all ten SFT n=2000 epoch snapshots;
- all ten SFT n=3000 epoch snapshots;
- all three formal Outcome-GRPO n=1000 epoch snapshots.

## Corrected evaluation contract

- data: exact `pathmmu_image_disjoint_v2` validation 385, SHA-256
  `6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca`;
- deterministic greedy decoding;
- `max_new_tokens=1024`;
- pinned prompt, image preprocessing, chat template and parser v2;
- exact generated-token count and EOS/cap status saved for every completion;
- any completion reaching 1024 invalidates the entire run;
- every saved reward and choice must exactly match offline parser-v2 rescoring;
- selection is maximum accuracy, then strict format, then earliest epoch;
- no automatic retry after a failed job.

The earlier 768-token three-model pilot remains diagnostic evidence (`formal_result: false`). This
all-candidate run is the formal correction required before selecting checkpoints or calculating
paper-facing Base-to-SFT-to-RL improvements.

