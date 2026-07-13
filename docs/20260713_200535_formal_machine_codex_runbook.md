# Formal-machine Codex runbook added

Timestamp: `2026-07-13 20:05:35 Asia/Shanghai`

`FORMAL_MACHINE_CODEX_GUIDE.md` is now the concise human/Codex entry point for cloning or updating
the handoff branch, reconstructing context from Git, inventorying shared compute, authenticating
without exposing tokens, running bootstrap/preflight, and continuing experiments through explicit
scientific gates.

The runbook intentionally stops automatic progress before formal long SFT. The repository currently
generates machine-resolved SFT YAML files, but the formal launcher, run-manifest automation, and
save/reload/resume smoke gate still need to be completed and validated on the actual formal hardware.
The new Codex is instructed to implement and run that gate first, report to the user, and only then
start scale or three-seed experiments.
