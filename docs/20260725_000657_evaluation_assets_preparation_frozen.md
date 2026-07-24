# External evaluation data and baseline-environment preparation frozen

Frozen: `2026-07-25T00:06:57+08:00`

Repository commit at freeze: `8fc2e095bae99f65ff9e6cfd468c83c07e89750c`

Manifest: `protocol/evaluation_assets_preparation_manifest_20260725_000657.json`

Baseline source manifest:
`protocol/pathology_baseline_source_manifest_20260725_000657.json`

Direct dependency pins: `protocol/pathology_clip_requirements_20260725.txt`

## Authorization and scope

The user authorized preparation of the newly proposed evaluation data and baseline environments.
This authorization does **not** authorize Stage 3 training or validation, any other training,
inference/evaluation, opening the formal PathMMU test split, GPU allocation, or storage cleanup.

This is a preparation record with `formal_result: false`. Downloads, extraction, schema inspection,
hashing, dependency installation and CPU-only import checks are allowed. No resulting model score is
a paper result.

The old API credentials found in historical scripts were reported by the user as expired and
invalidated. They must not be used. New credentials must remain outside Git and be supplied through
environment variables or the relevant credential store.

## Isolation and storage policy

- External root: `/home/wjy/pathvlm_revision_eval`
- The formal SFT and GRPO environments under
  `/home/wjy/pathvlm_r1_v1_formal/envs/` are immutable for this work.
- New environments, repositories, model caches, raw archives and extracted data remain outside Git.
- Repository additions are limited to source/configuration files, small manifests, hashes and
  documentation.
- No archive, model weight, image, environment, cache, token or API key may be committed.
- No cleanup is authorized in this preparation phase.
- Starting free space was 524 GB on `/home`; stop before any action projected to leave less than
  450 GiB free and report the blocked item instead of deleting anything.

## Frozen preparation order

1. Create the isolated directory structure.
2. Resolve and record immutable dataset repository revisions before downloading large assets.
3. Acquire small PathMMU metadata first and audit its schema/source fields without reading the
   sealed formal test predictions or producing any inference.
4. Acquire the four requested OmniMedVQA source subsets: Chest CT, ISIC2020, Retinal OCT-C8 and
   Diabetic Retinopathy. If the distribution is a monolithic archive, retain it initially and
   extract only the required files when the layout permits.
5. Prepare a normalized copy of the author-provided PathVQA test split as an independent external
   benchmark. Its labels may be used only for final scoring under a separately frozen evaluation
   protocol; this preparation step performs no model selection or inference.
6. Create an isolated pathology-encoder environment for PLIP, CONCH and UNI. Pin exact source
   revisions and package versions. Run CPU-only imports and model-construction checks where they do
   not require weights.
7. Download weights only after license/access checks. PLIP and CONCH are intended for
   multiple-choice image-text matching; UNI is a vision-only positioning baseline and must not be
   presented as a generative VQA model.
8. Keep LLaVA-Med in a separate optional environment because it is a direct generative VQA baseline
   with a materially larger dependency and weight footprint.
9. Write a completion or failure record with exact revisions, hashes, byte counts, paths and access
   limitations before any evaluation protocol is proposed.

## Data-contamination gates

- The formal `pathmmu_image_disjoint_v2` split and its manifests remain unchanged.
- `picked.json` remains forbidden.
- No external item may be added to training data under this authorization.
- Before any external evaluation, audit exact-byte image overlap against SFT, Outcome-GRPO,
  validation and the sealed PathMMU test split.
- Perform perceptual-near-duplicate auditing as a separate, documented analysis if implemented.
- Preserve benchmark source identifiers so results can be grouped by source, organ system,
  pathology subspecialty, question type and difficulty where supported.
- Historical exact 500-item cross-modal sample IDs have not been recovered. New full-subset results
  must not be described as an exact reproduction of those historical samples.

## Hardware and permission state at freeze

- All eight RTX 4090-class 48-GB GPUs showed active third-party memory use and utilization.
- No GPU process will be terminated or disturbed and no GPU will be selected in this phase.
- Hugging Face authentication was verified as user `Freddie1946`.
- Network access and writes under `/home/wjy/pathvlm_revision_eval` require explicit scoped
  escalation approval. Broad unrestricted shell access is neither necessary nor requested.

## Reviewer mapping

- R1-3 and R3-1: prepare complete Chest CT, ISIC, OCT and DR reruns for a transparent
  modality-by-modality transfer analysis, including negative transfer.
- R2-3: prepare compatible PLIP/CONCH comparisons and the separately qualified UNI positioning
  analysis; retain existing medical VLMs from the manuscript as the main generative baselines.
- R2-5: prepare independent/full held-out pathology evaluation with raw outputs and later confidence
  intervals.
- R3-4: record task/capability and comparability limits for UNI, CONCH and PLIP.

## Explicitly not started

Stage 3 prompt validation, process-reward training, any sensitivity run, all model inference, formal
test scoring, new training seeds and SFT/RL scale experiments remain unstarted by this decision.

## Migration handoff

The reproducible commands, immutable source/data revisions, extraction scope, environment pins,
credential rules and post-migration checks are maintained in Section 5 of
`FORMAL_MACHINE_CODEX_GUIDE.md`. The historical timestamped migration record was not rewritten.
