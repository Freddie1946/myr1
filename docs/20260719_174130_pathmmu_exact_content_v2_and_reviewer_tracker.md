# PathMMU exact-content v2 correction and reviewer-response tracker

Timestamp: `2026-07-19T17:41:30+08:00`

## Why v1 required correction

`pathmmu_image_disjoint_v1` grouped all QAs sharing one image basename and passed all six pairwise
basename-overlap gates. A later audit hashed the exact bytes of every referenced physical image and
found one cross-split duplicate that filenames alone could not detect:

- one RL image basename and one test image basename differed;
- their exact image-file SHA-256 was identical;
- the duplicate is present in the pinned official PathMMU image archive, so it was not created by
  the adapter or training pipeline;
- one additional exact-content duplicate group is internal to SFT and does not cross a split.

This exposed a narrow v1 gate deficiency: it proved basename disjointness, not exact-content
disjointness. No formal Outcome GRPO or test evaluation had run when the issue was discovered.

## Blinded correction decision

The approved minimal correction is frozen as `pathmmu_image_disjoint_v2`:

- retain v1 unchanged as historical provenance;
- preserve SFT, RL, validation, and every nested SFT/RL subset byte-for-byte;
- remove exactly one opaque QA from test, selected solely because its image bytes duplicate an RL
  image;
- do not inspect or use the removed question, answer, any model output, or any performance value;
- keep test evaluation-only.

The v2 primary counts are:

| Split | QA | Image basenames |
| --- | ---: | ---: |
| SFT | 3000 | 2121 |
| RL | 1000 | 708 |
| Validation | 385 | 272 |
| Test | 999 | 707 |

There are 3,808 referenced image basenames and 3,807 unique exact-file contents. All six pairwise
split overlaps are zero by both basename and exact-content SHA-256. The difference between basename
and content counts is the duplicate group wholly inside SFT.

## Implementation and gates

Repository artifacts:

- `formal_machine/build_pathmmu_image_disjoint_v2.py`: deterministic v1-to-v2 derivation;
- `formal_machine/verify_frozen_splits_v2.py`: recorded-file hashes, with/without-CoT membership,
  exact counts, nested/image-complete subsets, parent preservation, 3,808 physical image hashes,
  basename overlaps, exact-content overlaps, blind-decision flags, and `picked.json` absence;
- `data/pathmmu_image_disjoint_v2/manifest.json` SHA-256
  `aba5a93c1ace542034e3244e4379b3dc362e854f70eb3bc4ca96c543da7fdede`;
- `data/pathmmu_image_disjoint_v2/validation_report.json` SHA-256
  `13c00207b91314fb29d9b72b002c124eddbf9a8884d1cfbad5980ab1e194a5c0`;
- `data/pathmmu_image_disjoint_v2/image_content_sha256.json` SHA-256
  `e200edf3659e5af860d824a539bb0c5a29ee0c790de8b7a4ec528989c45b51ed`.

The builder was rerun into `/tmp`; its 26-file tree was byte-identical to the repository v2 tree.
The full verifier passed all 14 gates against the real PathMMU image directory. Formal framework
adapters were then created at:

`/home/wjy/pathvlm_r1_v1_formal/data/pathmmu_image_disjoint_v2`

The active machine preflight was regenerated after preserving the v1 report. It passed every model,
LLaMA-Factory, environment, import, dependency, count, basename-isolation, exact-content-isolation,
and reward-parser gate. At the snapshot time all eight RTX 4090 GPUs had no compute process; no
training or inference was launched.

The machine-local `formal_machine.env` and `FORMAL_PATHS.env` data pointers were switched to v2 after
backing up the v1 path file. The prior generated-config directory was also preserved, then all seven
SFT configs and six Outcome-GRPO scripts were re-rendered; every active data path now points to v2.
The GPU check briefly failed inside the restricted filesystem sandbox because `/dev/nvidia*` was not
exposed there; the required out-of-sandbox read-only NVML check immediately confirmed all eight GPUs
at 18 MiB and 0% utilization. This was not a host driver failure.

## Scientific bridge for completed SFT

The completed n=2000/n=3000 SFT run manifests remain truthfully labeled v1 and must never be
rewritten. They remain eligible as parents because v2 SFT and validation source files are
byte-identical to v1. The machine-rewritten v2 validation adapter has the same SHA-256 as v1:

`6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca`

Future validation manifests explicitly record a v1-parent/v2-evaluation bridge. Future SFT and RL
runs must use v2 and a v2 preflight. Test remains unopened.

## Correction to the chat-template failure description

The earlier Attempt01 record correctly identified the missing archived `chat_template.json` files
and the explicit-template remedy, but its statement that LLaMA-Factory wrote the template
“inconsistently” was too strong. A deeper timing audit demonstrates an archival race:

- in the original n=3000 epoch-9 checkpoint, `preprocessor_config.json` was timestamped
  `12:43:20.216+08:00` and `chat_template.json` `12:43:20.815+08:00`;
- the archiver manifest captured the preprocessor file but not the template, meaning its directory
  scan occurred during that approximately 0.599-second checkpoint-write interval;
- the original output checkpoint now contains the template, while the archived model snapshot does
  not; analogous missing-template snapshots follow the same failure mode.

Therefore the root cause is that the asynchronous archiver accepted a checkpoint before all
processor/template files were quiescent. The scientifically safe evaluation fix remains to inject
one pinned base-model chat template into every job and verify its hash. Future retention code must
also require the template or a checkpoint-complete/quiescence marker before archiving.

## Reviewer mapping and ongoing record

`manuscript/revision/reviewer_response_tracker.md` is now the living point-by-point response ledger
for the Associate Editor and all comments from Reviewers 1--3. The v2 action directly addresses
Reviewer 2 comment 2 (split unit and leakage), and supports the reproducibility/data-scale requests
under Reviewer 2 comment 1 and Reviewer 3 comments 5--6. It deliberately does not claim patient-level
deduplication because PathMMU records expose no verified patient identifier, and exact-byte hashing
does not rule out perceptually near-duplicate images.

Every material future experiment, failure, correction, figure/table edit, and rebuttal decision must
update a timestamped `docs/` record, the relevant manifest, `docs/LATEST.md`, and the affected entries
in the reviewer tracker. Historical failed or v1 manifests remain immutable.

## Next gate

Freeze and verify the new code manifests, commit the v2 correction, then run only the validation-curve
preflight. Attempt02 may launch later as a fresh 21-job validation-only run; no Attempt01 output may be
reused, and test remains untouched.
