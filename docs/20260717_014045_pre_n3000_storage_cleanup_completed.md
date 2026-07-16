# Pre-n3000 storage cleanup completed

Timestamp: `2026-07-17T01:40:45+08:00`

## Scope and authorization

The user explicitly authorized the first three proposed cleanup classes before the fresh n=3000,
seed-42 formal SFT launch. Only the six listed `output/` directories were deleted. Every parent run
directory, run manifest, resolved configuration, training/validation log, resource trace, and
environment snapshot that existed outside `output/` was retained.

## Deleted payloads

- Failed 2026-07-14 SFT smoke output: 125,943,255,309 bytes.
- Superseded completed 2026-07-14 SFT smoke output: 235,286,131,206 bytes.
- Three non-formal speed-smoke model outputs: 16,600,373,477, 16,600,373,477, and
  16,600,373,482 bytes.
- Failed n=2000 formal-attempt output containing checkpoint-700 and checkpoint-800:
  218,690,006,693 bytes.

The sum of measured pre-deletion directory sizes was 629,720,513,644 bytes (586.47 GiB). Filesystem
free space increased by 629,724,426,240 bytes, from 1,191,457,816,576 to 1,821,182,242,816 bytes.
`/home` utilization fell from 84% to 75%.

Each associated external run manifest now contains an appended `storage_pruning` event with the
deletion timestamp, exact path, measured bytes, reason, and retained evidence. Original run status
and `formal_result` values were not changed.

## Preserved scientific outputs

The completed formal n=500 and n=1000 runs, their final models, full checkpoints, validation raw
predictions, and manifests were not touched. The current successful checkpoint-retention smoke was
also retained. No dataset, base model, environment, cache, credential, or test artifact was touched.

## Resulting launch margin

The formal runner requires 1,157,493,686,272 free bytes at startup. Post-cleanup free space exceeds
that gate by 663,688,556,544 bytes (618.11 GiB), in addition to the 550-GiB hard reserve already
included inside the runner's requirement.

## Next action

Commit this cleanup record together with the correctly labelled GC speed-smoke record. Then repeat
the physical GPU occupancy check and launch only the fresh n=3000, seed-42 formal SFT through the
audited runner. The runner must repeat all fail-closed disk, GPU, model, data, preflight,
configuration, and master-port gates before creating the run.
