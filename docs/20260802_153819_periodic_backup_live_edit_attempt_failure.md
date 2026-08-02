# Periodic backup live-edit attempt failure

Timestamp: `2026-08-02 15:38:19 CST`

## Outcome

The 15:30 periodic invocation did not upload. While that invocation was sleeping between bounded
authentication retries, its script file was updated with the wall-clock timeout correction. The
already-running Bash process subsequently resumed from the old byte position in the changed file
and attempted to execute an informational string as a command. It exited without writing a new
`upload_history.tsv` row or advancing `last_uploaded_aggregate_sha256`.

The local 15:30 audit remains valid. The last verified remote rolling backup remains revision
`06a0de6b8a940fddeeba0ee011212a2cfc0e8f6e`; no false success marker was created. Training,
semantic-Judge workers and their artifacts were unaffected.

After the old invocation exited and released its lock, the complete current script passed a
no-network dry run at `20260802_153819` with aggregate manifest SHA-256
`c74355e67884390b371b8a84782b2338aeabc91d6fc8f4522ff6989e11ee04a4`. No further manual live
backup was started. The next minute-0/minute-30 cron invocation will use the fully parsed corrected
script. Operational rule: do not edit this script while an invocation holds `periodic.lock`.

This is a failed operational attempt (`formal_result: false`), not a model or evaluation failure.
