# Formal-machine Git/Codex handoff implemented

Timestamp: 2026-07-12 19:28:46 Asia/Shanghai

## Delivered repository behavior

The repository now acts as the cross-machine memory source. A fresh Codex session is instructed by `AGENTS.md` to read `CODEX_START_HERE.md`, `docs/LATEST.md`, protocol manifests, and timestamped state before acting.

The current valid/invalid experiment history is explicit, including the invalid Attempt03 accuracy-parser update and the pending parser-v2 Attempt04.

## Formal setup entry point

```bash
cp formal_machine/formal_machine.env.example formal_machine.env
# Edit paths and online/offline settings.
bash setup_formal_machine.sh formal_machine.env
```

The bootstrap:

1. Checks NVIDIA GPU, disk, conda, data, and source prerequisites.
2. Creates isolated SFT and GRPO conda-prefix environments.
3. Pins known working Transformers/Tokenizers/TRL/DeepSpeed versions.
4. Uses vendored, audited Open-R1 source.
5. Clones/checks out the audited LLaMA-Factory commit when online, or consumes a copied source tree offline.
6. Downloads or links the exact 7B snapshot and writes a source/revision manifest.
7. Rewrites image paths and generates LLaMA-Factory plus GRPO adapters.
8. Verifies counts, nested subsets, all image paths, `picked.json` exclusion, and pairwise image-disjoint splits.
9. Renders machine-resolved SFT smoke/scale/seed configs and Outcome-GRPO smoke/scale/seed scripts without launching long jobs.
10. Runs import, reward-parser, model-identity, hardware, and data preflight gates.

## Static and data tests completed locally

- Python compilation passed for setup/data/preflight/config-render/reward scripts.
- Bash syntax checks passed for all entry and packaging scripts.
- Reward parser v2 regression tests passed.
- Formal data preparation ran against all frozen split JSONs using 3,809 placeholder image files.
- Expected QA counts passed for all SFT/RL/validation/test sets.
- Expected unique image counts passed: 2121/708/272/708.
- All six cross-split image overlap counts were zero.
- Generated SFT and GRPO config manifests contained the expected scale and seed variants.
- Generated GRPO smoke script passed Bash syntax validation.

## Not yet claimed

The full environment installation has not been executed on the unknown formal machine because its GPU model, driver, CUDA compatibility, network policy, conda path, and mount layout are not available locally. The new Codex must inspect those facts, edit `formal_machine.env`, run bootstrap, and preserve the resulting preflight report before training.

## Git content policy

Included: source, vendored Open-R1, frozen split JSONs, configs, manifests, tests, and timestamped history.

Excluded: images, model/checkpoint weights, environments, caches, outputs, credentials, and machine-local config.

## Remaining human action

Create or select a Git remote, add it as `origin`, and push the handoff branch. The repository currently has no remote URL because none was provided.

