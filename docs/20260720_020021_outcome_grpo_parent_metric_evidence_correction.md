# Outcome-GRPO parent metric evidence correction

Timestamp: 2026-07-20 02:00:21 Asia/Shanghai

The first `--preflight-only` execution passed every blocking check at commit `b428e8d`, including the
clean worktree, frozen protocol/code hashes, all 17 selected parent files, SFT/validation lineage,
v2 RL-smoke membership, parser regressions, disk reserve and system memory. It did not inspect or
occupy GPUs and did not start training.

The printed parent evidence showed `validation_accuracy: null` and `validation_format: null`. The
validation-curve YAML stores planned entries under `jobs` and completed entries under `results`.
The runner correctly bound the parent path and snapshot hash through `jobs`, but incorrectly attempted
to display completed metrics from that plan-only entry.

Before any GPU action, the runner was corrected to:

1. continue binding the planned job to the exact epoch-3/step-1125 snapshot;
2. independently require the matching completed `results` entry;
3. require its `status: completed`, `formal_result: true`, `test_accessed: false`, exact checkpoint
   path, and every result gate true; and
4. record validation accuracy/format from that completed result.

This was a provenance-display and strengthening correction, not a parent-selection change or a
training attempt. The Outcome-GRPO code manifest was updated before execution. No validation/test
generation was performed, no model file changed, and no GPU was used.
