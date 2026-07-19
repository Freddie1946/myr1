# Stage-2 priority pilot completed; formal launch stopped by storage gate

Timestamp: `2026-07-20T05:01:44+08:00`

## Disposition

The user-approved 50-step Outcome-GRPO pilot completed and passed every frozen pilot gate. The
pipeline then stopped before creating or launching the formal n=1000 run because its conservative
post-pilot storage projection failed. This is a successful engineering pilot followed by a
fail-closed orchestration stop, not a failed training result. The pilot remains
`formal_result: false`; no formal long RL result is claimed.

Run root:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage2_outcome_grpo/priority_n1000_seed0042/stage2_priority_n1000_seed0042_20260720_035156`

## Save/resume and update evidence

- Segment A completed steps 1--25; all 25 gradients were finite and nonzero, and 18 steps had
  positive trainer reward standard deviation.
- A complete 109,344,811,029-byte `checkpoint-25` was saved.
- A fresh eight-worker process loaded that checkpoint and continued steps 26--50. All 25 resumed
  gradients were finite and nonzero, and 16 resumed steps had positive reward standard deviation.
- Full language parameters remained trainable. All vision and multimodal-projector parameters
  remained frozen.
- The representative language tensor changed; the representative visual tensor was exactly equal.
- The final 16.6-GB model save passed offline Qwen2.5-VL model/processor reload checks.

## Online reward and source audit

The two segments preserved 800 raw reward events for 400 completions. Every event was bound to an
exact frozen RL record index, resolved image path and SHA-256, question, solution, structured prompt
and prompt-contract version. Online and offline parser results matched exactly.

| Segment | Completions | Accuracy | Format | Positive-variance groups / 50 |
| --- | ---: | ---: | ---: | ---: |
| Steps 1--25 | 200 | 60.0% | 100.0% | 22 |
| Steps 26--50 | 200 | 56.0% | 99.5% | 22 |

The combined format rate was 99.75%, above the frozen 98% pilot threshold.

## Full validation result

Deterministic validation generated and preserved all 385 raw predictions using the pinned chat
template and parser v2. Exact source provenance and offline re-scoring both passed.

- Parent SFT n=3000 epoch 3: 233/385 = 60.5195%.
- Pilot step 50: 232/385 = 60.2597%.
- Difference: -1/385 = -0.2597 percentage point.
- Strict format: 385/385 = 100%.
- Extracted A--D choice: 385/385 = 100%.
- Empty completions: 0.
- Paired outcomes: both correct 224, parent-only 9, pilot-only 8, both incorrect 144.
- Exact two-sided McNemar p-value: 1.0; no significant degradation.

The pilot therefore passed the minimum 222/385 accuracy, 98% format/extraction, no-empty and paired
non-degradation gates. Test was not accessed.

## Storage stop

After the pilot, free space was 839,668,518,912 bytes. Measured payloads were:

- one full pilot resume checkpoint: 109,344,802,837 bytes at the projection instant;
- gathered model payload: 16,584,414,544 bytes.

The frozen projection budgets two simultaneous full checkpoints during save/rotation, four model
payloads for three epoch snapshots plus final save, and 20 GiB overhead. Together with the 550-GiB
hard reserve, it required 897,060,103,530 free bytes. The deficit was 57,391,584,618 bytes, so the
runner stopped before formal-run directory creation. All GPUs returned to the 18-MiB idle baseline.

The most direct storage-safe option is to record and remove only the now-consumed pilot
`checkpoint-25` resume payload. The final pilot model, train-state audits, reward events, logs,
tensor comparison, full validation predictions and manifests would remain. This deletion is not
performed without explicit user approval. After any approved cleanup, a new formal-only launcher
must recheck the frozen pilot manifest, exact code/parent/data hashes, idle GPUs, port and storage;
the 50-step pilot should not be rerun.

## Reviewer mapping

This supplies engineering evidence for R2-4 and the outcome-only prerequisite for R3-3. It preserves
raw paired validation evidence for R3-6 and prepares the prioritized n=1000 point for R3-5. It does
not complete the formal RL arm, RL scale curve, multi-seed study, Stage 3 or test evaluation.
