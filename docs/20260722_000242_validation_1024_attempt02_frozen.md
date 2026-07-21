# Corrected 1024-token validation Attempt02 frozen

Timestamp: `2026-07-22T00:02:42+08:00`

The user confirmed that 1,024 tokens is a sufficient evaluation timeout and authorized a complete
Attempt02 rerun. Any completion reaching the limit is hard-truncated, preserved verbatim, scored by
the unchanged parser v2, counted in that model's generation-cap statistics, and treated as model
behavior rather than an infrastructure failure. Source, parser, empty-output, checkpoint and job
failures remain fail-closed. Attempt01 outputs are preserved but not reused.

Attempt02 again evaluates exactly 26 candidates and saves 10,010 validation predictions. It does
not train a model, start Stage 3, access test, or use `picked.json`.

The user also manually confirmed that physical GPUs 0--7 are available and explicitly requested
skipping the automated hardware-state confirmation stage. The launcher therefore requires an
explicit `--user-confirmed-hardware` flag and records the confirmation source; it does not execute
the automated GPU inventory/process query. Code/data/checkpoint/parser preflight gates remain.

