# LLaMA-Factory pin bootstrap failure and correction

Timestamp: `2026-07-13 20:59:56 Asia/Shanghai`

## Attempt result

The first formal-machine bootstrap attempt stopped before environment creation, package installation,
PathMMU download, base-model download, or training. Frozen split verification passed. The failure was:

```text
fatal: reference is not a tree: ef5f1c1def3da62ee2d5e6ba933f9d7d6aab4340
```

At the hardware snapshot, GPU 0 still contained the collaborator's Python process. No CUDA workload
was started and no process was terminated or disturbed.

## Diagnosis

The failed hash came from the historical W&B record described in `protocol/old_sft_audit.md`. A full
clone of the official `hiyouga/LLaMA-Factory` repository did not contain that object. A second existing
clone on the formal machine also did not contain it. Therefore it was incorrect for the online
bootstrap to treat that historical hash as an upstream-recoverable commit.

The older code hash manifest records the audited launcher SHA-256 as
`8f16bb782a6da2122b5accd50ce1a01fd99284dffa20ff340850ec3b06927b77`. That exact launcher hash is
present at the official LLaMA-Factory `v0.9.2` tag. The tag resolves to upstream commit
`e2299e261be852304bb1d370515078193ab12bd8`, consistent with the documented framework version 0.9.2.
This does not claim recovery of the complete historical private/dirty worktree; it defines a new,
publicly recoverable source pin for the formal revision experiments.

## Correction and new gates

- Pin LLaMA-Factory to official commit `e2299e261be852304bb1d370515078193ab12bd8`.
- Require the launcher SHA-256 above after checkout.
- Write `reports/llamafactory_source_manifest.json` with URL, requested revision, resolved HEAD,
  release, and launcher hash.
- Require exact revision and launcher-hash agreement in `preflight_report.json`.
- Record the same explicit values in the environment template and the gitignored machine-local config.
- Make the SFT smoke snapshot the new repository code manifest.

The corrected bootstrap may prepare environments, data, model, adapters, and preflight while GPU 0 is
occupied, but the formal SFT smoke and all training remain forbidden until a fresh GPU occupancy check
confirms the allocated devices are available.
