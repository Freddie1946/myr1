# GRPO declared-dependency gate correction

Timestamp: `2026-07-13 22:23:06 Asia/Shanghai`

After the successful bootstrap, an explicit `pip check` found that the vendored Open-R1 package
declares `einops` and `hf-transfer`, but the no-dependencies editable installation had not installed
them. The trainer import and reward regression tests already passed, so this was not an observed
runtime failure; it was nevertheless an inconsistent environment and is not acceptable for the
formal gate.

The bootstrap now installs both declared packages explicitly in the isolated GRPO environment. The
preflight now executes and requires successful `pip check` results for both SFT and GRPO environments.
No model, split, adapter, reward, or training configuration changed, and no training ran.
