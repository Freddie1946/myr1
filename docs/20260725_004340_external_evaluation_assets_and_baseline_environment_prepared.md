# External evaluation assets and baseline environment prepared

Completed: `2026-07-25T00:43:40+08:00`

Preparation protocol:
`protocol/evaluation_assets_preparation_manifest_20260725_000657.json`

Completion manifest:
`protocol/evaluation_assets_preparation_completion_manifest_20260725_004340.json`

External root: `/home/wjy/pathvlm_revision_eval`

This is preparation-only engineering evidence with `formal_result: false`. No model inference,
training, Stage 3 work, GPU allocation, formal PathMMU test-label access, prompt selection or storage
cleanup occurred.

## Prepared data

- Fixed-revision PathMMU metadata: `data.json`, `instructions.md`, `socialpath_mapping.json`,
  `README.md` and `construct_pathcls.py`; no second image archive was downloaded.
- Fixed-revision OmniMedVQA archive and four requested full source subsets:
  - Chest CT Scan: 871 QA, 382 referenced paths, 349 unique exact contents.
  - ISIC2020: 1,580 QA, 1,499 referenced paths/contents.
  - Retinal OCT-C8: 4,016 QA, 3,224 referenced paths and 3,222 unique exact contents.
  - Diabetic Retinopathy: 2,051 QA, 1,966 referenced paths/contents.
- Fixed-revision normalized copy of the author-provided PathVQA test split: three parquet shards,
  6,719 QA and 858 unique exact image contents.

The actual Diabetic Retinopathy count is 1,966 unique referenced images, correcting the earlier
1,996 estimate. Every OmniMedVQA image reference exists.

The integrity audit found internal exact duplicates in the upstream external data: Chest CT has 33
extra paths across 17 duplicate-content groups, and OCT-C8 has two extra paths in two groups. These
must be grouped by exact content for image-level confidence intervals or sampling. They are not
silently removed in the preserved upstream package.

No exact image-content overlap was found between any prepared external source and the 3,807 unique
formal PathMMU v2 contents. No exact overlap was found between PathVQA test and the four OmniMedVQA
sources. The audit report passed all gates:

`/home/wjy/pathvlm_revision_eval/reports/external_eval_asset_audit_20260725.json`

Report SHA-256:
`d5a4d8e21e207c26e6c0853026566b0b3e09793a99a73677fe6617e9526b727d`

## Prepared baseline environment

Created `/home/wjy/pathvlm_revision_eval/envs/pathology_clip` without modifying formal SFT/GRPO
environments. It uses Python 3.10, Torch 2.5.1+cu121, torchvision 0.20.1+cu121, transformers 4.49.0,
tokenizers 0.21.0, timm 0.9.8 and datasets 3.3.2. `pip check` and CPU-only imports pass; CUDA was
hidden during imports.

Exact source commits:

- CONCH: `141cc09c7d4ff33d8eda562bd75169b457f71a62`
- UNI: `42715efc11722a496e0a67f3369505a8f277206c`
- PLIP: `f010f3d0bef20f4e8cc64cc26c301cbd26305fa1`
- LLaVA-Med: `30697ca50b5c29a8e955c99330b259776aef27b9`

CONCH and UNI installed editable from the fixed commits. PLIP's upstream `setup.py` is invalid
because it declares a nonexistent package directory; the three-repository Attempt01 failed at PLIP
metadata generation. The corrected environment does not modify upstream code: PLIP passes a
CPU-only source import through `PYTHONPATH`, while the future adapter will use the official
Transformers `CLIPModel`/`CLIPProcessor` interface.

Full environment records:

- pip freeze:
  `/home/wjy/pathvlm_revision_eval/reports/pathology_clip_pip_freeze_20260725.txt`
- Conda explicit list:
  `/home/wjy/pathvlm_revision_eval/reports/pathology_clip_conda_explicit_20260725.txt`

Their SHA-256 values are recorded in the completion manifest.

## Remaining access blocks

PLIP public configuration/tokenizer metadata was downloaded at model revision
`67ade53ddd32195868f422585f72698ef5d15094`; its 605-MB weight was not downloaded under the current
data/environment-only scope.

The logged-in Hugging Face account can read CONCH and UNI model cards, but authenticated HEAD checks
for both weight files returned 403. The user must accept/request access at
`MahmoodLab/CONCH` and `MahmoodLab/UNI`; shell/filesystem permission cannot bypass their licenses.
The fixed model revisions and reported weight sizes are in the completion manifest.

LLaVA-Med's separate environment and 15-GB weight snapshot remain deferred. No score is available
for PLIP, CONCH, UNI, LLaVA-Med, OmniMedVQA or PathVQA.

## Storage

The new external root occupies approximately 18 GB. `/home` had 506 GB free after preparation,
above the 450-GiB preparation stop floor but below the formal-training 550-GiB reserve. Formal
training must therefore remain stopped unless storage is separately audited and authorized.

## Migration record

The complete repeatable data, source, environment, validation and credential steps were added to
Section 5 of `FORMAL_MACHINE_CODEX_GUIDE.md`. The historical timestamped migration record remains
unchanged.
