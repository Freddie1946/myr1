# Periodic Hugging Face transport retry correction

Timestamp: `2026-08-02 15:01:32 CST`

## Failure and correction

The first scheduled invocation ran at 15:00 CST, proving that the minute-0/minute-30 cron is
active.  Its Hugging Face `auth whoami` request then failed with a transient
`httpx.ConnectError: [Errno 104] Connection reset by peer`; no remote commit or local state-hash
advance occurred.  Training, local audits, the previous HF revisions and active PathVQA outputs
were unaffected.

The periodic backup now gives both HF authentication and upload at most three physical attempts,
with delays of 15 and 45 seconds after the first attempt.  Exhaustion remains terminal for that
invocation and is recorded in `cron.log`; the next half-hour schedule remains available.  No API
token is printed or copied into the repository.

Bash syntax and a no-network dry run passed after the correction.  A corrected real upload and
remote nested-path verification remain required.

This is operational recovery hardening (`formal_result: false`), not an efficacy result.

