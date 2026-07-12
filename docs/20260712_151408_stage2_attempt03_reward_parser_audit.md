# Stage 2 Attempt 03 reward-parser audit

Timestamp: 2026-07-12 15:14:08 Asia/Shanghai

## What worked

- Four 192-token-cap completions were generated online.
- Eight reward events were saved.
- Logged reward variance was nonzero.
- Gradient norm was 6.4459543.
- The representative language tensor changed in 47,611 of 4,194,304 elements.
- The representative frozen visual tensor changed in 0 of 1,638,400 elements.
- Checkpoint save and exit status were successful.

## Parser defect discovered during case analysis

One completion reasoned toward fibroblasts but placed a JSON object inside `<answer>` that enumerated `Option A`, `Option B`, `Option C`, and `Option D`. The old parser selected the first `Option A` occurrence and incorrectly awarded accuracy 1.0.

The old parser also failed to recognize explicit malformed-but-clear JSON fields such as `{"option": "B") Melanocytes"}`. Therefore the policy update used inconsistent multiple-choice parsing and the checkpoint is invalid for scientific use.

## Corrected offline audit of the same four completions

- Accuracy rewards: `[0, 0, 0, 0]`.
- Format rewards: `[1, 1, 0, 0]`.
- Total rewards: `[1, 1, 0, 0]`.
- Mean accuracy: 0.0.
- Mean format: 0.5.
- Sample total-reward standard deviation: 0.5773503.

Thus the same completions still contain legitimate format-reward variance, but the saved parameter update cannot be accepted because online advantages were computed with the flawed accuracy credit.

## Fix

Parser v2:

1. Prioritizes explicit `answer`, `option`, or `choice` value fields.
2. Requires a separator for phrases such as `Answer: B` or `Option = B`.
3. Accepts a lone `B)` style choice only when it is unambiguous.
4. Rejects answer blocks that enumerate multiple choices without selecting one.

Regression tests cover explicit JSON choices, malformed JSON-like choices, ordinary `A)` answers, and option-enumeration false positives. Local and remote tests pass.

## Next action

Prune only Attempt03's invalid model shards after approval, retain all audit artifacts, and run Attempt04 from the original SFT parent using parser v2.

