# Formal data-adapter idempotency correction

Timestamp: `2026-07-14 00:10:41 Asia/Shanghai`

The repeated formal bootstrap stopped at data-adapter preparation with `FileExistsError` because the
first successful bootstrap had already populated the formal adapter directory. The frozen PathMMU
image download and exact base-model snapshot had both completed successfully; this was not evidence
of missing or corrupt data.

The adapter generator now supports `--reuse-if-valid`. In this mode it deterministically reconstructs
every expected rewritten-record, LLaMA-Factory, GRPO, smoke, dataset-info, and formal-data-manifest
file in memory, compares the UTF-8 content of all 26 files against the existing output, and rejects
missing, unexpected, or differing files. It also repeats source-count, image-existence, subset
nestedness, and pairwise image-disjointness checks. It performs no writes to a valid existing adapter
directory.

The bootstrap now selects `--reuse-if-valid` whenever `OVERWRITE_DATA=0`; destructive regeneration
remains possible only through the explicit `OVERWRITE_DATA=1` operator choice. The real formal
adapter directory passed strict reuse validation with 26 files, all six top-level pairwise image
overlaps zero, and `picked_json_used: false`. A newly generated temporary adapter tree was then
modified by one YAML line and correctly rejected by the reuse validator. No formal data, model
weights, configuration, checkpoint, or training state changed, and no training ran.
