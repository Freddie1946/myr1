# Git-based Codex handoff plan

Timestamp: 2026-07-12 19:18:13 Asia/Shanghai

## Requirement

The formal machine is reachable only through a remote-desktop jump host. A Codex session launched there does not inherit the local Codex conversation. Therefore the Git repository itself must carry the complete operational memory needed to configure the machine and continue experiments.

## Repository memory layers

1. `AGENTS.md`: automatically applicable rules for any Codex working in the repository.
2. `CODEX_START_HERE.md`: explicit recovery summary, current valid/invalid experiment status, and formal execution order.
3. `docs/LATEST.md`: timestamped chronological state pointer.
4. `protocol/`: fixed data/model/code/training manifests.
5. `configs/`, `scripts/`, and `formal_machine/`: executable implementation rather than prose-only guidance.
6. Run manifests and small audit JSON files: evidence for individual attempts.

## Git inclusion policy

Include:

- Source and wrapper scripts.
- Environment version locks.
- Frozen split JSON files and manifests.
- Configuration templates.
- Reward tests.
- Timestamped plans/results/corrections.
- Small predictions and audit outputs when useful.

Exclude:

- Model/checkpoint weights.
- Images.
- Conda/venv directories.
- Hugging Face caches.
- Generated output directories.
- Secrets and machine-local configuration.

## New Codex recovery contract

The new Codex must read `AGENTS.md`, `CODEX_START_HERE.md`, and `docs/LATEST.md` before acting. It must first report formal-machine facts and mismatches, then run bootstrap/preflight, then perform only a one-step formal-hardware smoke. Full experiments begin only after all gates pass.

## Human workflow

1. Push this repository to a Git remote accessible from the jump/formal machine.
2. Clone it on the formal machine.
3. Copy non-Git assets through the remote-desktop shared drive or make them accessible on mounted storage.
4. Edit `formal_machine.env`.
5. Ask formal-machine Codex to read the handoff files and execute setup.
6. Commit all new timestamped state back to Git so future sessions inherit it.
