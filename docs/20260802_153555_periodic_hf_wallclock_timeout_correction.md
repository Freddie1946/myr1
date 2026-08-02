# Periodic HF wall-clock timeout correction

Timestamp: `2026-08-02 15:35:55 CST`

## Observation

The 15:30 periodic audit was created successfully, but the first private-HF authentication attempt
was reset by the peer and the second `hf auth whoami` process remained blocked for several minutes.
The existing three-attempt loop bounded attempt count but did not bound the duration of each CLI
call, so a hung authentication process could retain the audit lock and suppress later cron runs.

The exact blocked authentication child (PID 4103880) was terminated with `SIGTERM`; no training,
semantic-Judge or supervisor process was signalled. The already-running periodic wrapper continued
to its existing 45-second delay and final bounded retry.

## Correction

Future invocations now wrap each HF authentication call in GNU `timeout` with a default 60-second
wall-clock limit and each upload call with a default 600-second limit. Both limits are positive
integer environment overrides. The existing three attempts, 15/45-second retry delays, fail-closed
exit behavior and content-hash state advancement remain unchanged. A timed-out attempt counts as a
failed physical transport attempt and cannot mark a backup complete.

## Verification

- Bash syntax validation passed.
- Invalid zero-second configuration exits with status 2 before audit or transport work.
- GNU `timeout` was verified to terminate a synthetic five-second command at one second with exit
  status 124.
- GPT-4o Stage3 remained healthy and advanced from step 910 to step 915 during this correction;
  fallback remained 2 total and 0 consecutive.

This is an operational disaster-recovery correction (`formal_result: false`). The 15:30 remote
upload outcome remains pending and must be read from `upload_history.tsv`; no success is inferred
from the existence of the local audit.
