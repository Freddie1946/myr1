# Bbox output formats inside the trained think/answer envelope

Date: 2026-08-10 01:20 Asia/Shanghai

## Motivation

The direct-JSON smoke showed that the Stage3 checkpoints strongly prefer their trained
`<think>...</think><answer>...</answer>` response. This follow-up tested whether bbox localization
could be added without abandoning that familiar envelope.

## Formats

GPT-4o Stage3 step 1000 received the same three frozen PathMMU validation cases under four output
contracts:

1. XML `<evidence>` and `<bbox .../>` elements inside `<think>`;
2. Qwen-style `<ref>finding</ref><box>(x1,y1),(x2,y2)</box>` inside `<think>`;
3. a plain `VISUAL_EVIDENCE: type; boxes=[[...]]` line associated with the think/answer output;
4. strict JSON inside an `<evidence_json>` element within `<think>`.

The cases included one non-localizable reference and two focal/multifocal references. Ordinary
model answers had been recorded before the format probe, allowing answer preservation to be
measured separately from answer correctness.

## Results

| Format | Answer extracted | Answer preserved | Bbox payload valid | Spatial applicability correct |
|---|---:|---:|---:|---:|
| XML bbox in think | 3/3 | 1/3 | 0/3 | 0/3 |
| Qwen ref/box in think | 3/3 | 2/3 | 0/3 | 0/3 |
| Plain coordinates | 3/3 | 2/3 | 1/3 | 0/3 |
| Evidence JSON in think | 3/3 | 2/3 | 0/3 | 0/3 |

The XML and Qwen-native instructions were omitted entirely from all outputs. The only parseable
plain-coordinate payload declared a genuinely multifocal duct case `not_localizable` and returned
no box. Evidence-JSON sometimes produced coordinate-like strings after the normal answer, but the
payloads had malformed JSON, missing quotes, an unexpected array wrapper, or non-schema evidence
types. None yielded a valid spatial box.

## Interpretation

Keeping the outer think/answer format does not by itself unlock reliable bbox output. The model
learned not only the XML envelope but also a strong tendency to terminate after the answer and to
omit untrained structured fields. The earlier isolated meaningful bbox therefore remains evidence
of latent technical capability, not a stable interface.

The next technically justified attempt is not another prompt spelling. It is a two-pass decoder:

1. freeze the normal model answer;
2. start a separate localization turn for that fixed option;
3. enforce a bbox grammar during decoding rather than merely requesting one;
4. causally test the returned region with deletion/retention against matched random regions.

If grammar-constrained decoding still gives semantically poor boxes, a small bbox-format SFT is
needed. Its pseudo-labeled training cases must be disjoint from held-out localization evaluation
and later expert review.

## Artifacts

- Runner: `scripts/run_bbox_output_format_probe.py`
- Unit tests: `scripts/test_run_bbox_output_format_probe.py`
- Output:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/bbox_output_format_probe_gpt4o_stage3_20260810`
