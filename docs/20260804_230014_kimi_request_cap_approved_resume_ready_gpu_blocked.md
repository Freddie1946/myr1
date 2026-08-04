# Kimi recovery request cap approved; resume ready but GPU-blocked

Recorded at: `2026-08-04 23:00:14 CST`

## Outcome

The user approved restarting Kimi Stage3 after the HTTP 520 correction. This approval is recorded
as authorization to raise the shared physical-request cap from 12,361 to the previously calculated
minimum 12,976. The separately retained USD ledger ceiling remains `$26.71782984`.

The run-specific launcher and supervisor now both freeze `MAX_UNIQUE_REQUESTS=12976`. The launcher
also validates that exact value before any GPU or paid work. The original paid-smoke marker remains
immutable at 12,361; it is historical evidence of the initial contract, not rewritten as if the
larger recovery allowance had existed at launch.

The increase is exactly 615 physical requests consumed by failed/discarded trajectory portions or
interrupted calls. It retains 353 retry slots after accounting for the seven retries already used
by the checkpoint-200 trajectory. It does not add optimizer steps, change the final 1,500-step
trajectory, or alter the Judge/scoring contract.

## Verification

- shell syntax: passed for launcher and supervisor;
- Kimi recovery, OpenRouter Judge, checkpoint recovery and shared AIGCBest suites: 56/56 passed;
- latest complete checkpoint discovery: checkpoint 200;
- checkpoint contents: four model shards, eight optimizer states, eight model states and eight RNG
  states;
- worktree contained only the approved launcher/supervisor/test changes before this record.

## Launch gate

Training was not started because all eight A100 GPUs are occupied by independent processes under
`/home/dataset-assist-0/czy/zw/AFI-VLA`. PIDs 618758 through 618765 each use approximately 36.9
GiB. These processes were not stopped, signaled or modified. The formal launcher requires every
GPU to use at most 10 MiB before launching, so starting the supervisor now would be a guaranteed
preflight failure and would incorrectly consume a bounded recovery index.

Once all eight GPUs are idle, the approved start is segment02 with start launch index 2, maximum
recovery index 3, checkpoint cadence 100, and automatic discovery/resume from checkpoint 200.

## Code hashes

- `scripts/launch_stage3_kimi26_formal.sh`:
  `38e1744b20a63c94feaee422618356e962931a7e5f86c9a31c7c8c8c55b58b37`
- `scripts/supervise_stage3_kimi26_formal.sh`:
  `893c96c1c061b9c5b1aaac12371b648ab2dda6e211c06b55801bb43f76fc965f`
- `scripts/test_stage3_kimi26_recovery_contract.py`:
  `0ae308deedfc13ecc8bde3506eba42d6d0ee9c5fec9cedb4c42fe619b7d6b56b`
