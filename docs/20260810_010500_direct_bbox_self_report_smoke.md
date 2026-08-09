# Direct bbox self-report smoke

Date: 2026-08-10 01:05 Asia/Shanghai

## Question and frozen cases

This diagnostic asks whether the PathVLM model can directly return image-space boxes for the
visual evidence it says supports its answer. It uses eight pre-existing PathMMU validation cases:
five independently classified by Claude Sonnet 4.6 as focal/multifocal and three classified as
diffuse/not-localizable. Claude annotations are external-model pseudo references, not pathology
expert ground truth. No test split was accessed.

The requested output is a strict JSON object with an answer, evidence type, zero to four boxes in
normalized 0--1000 image coordinates, and a short reason. The model is explicitly permitted to
abstain from box output for diffuse or non-localizable evidence.

## Prompt behavior

With a normal assistant start, GPT-4o Stage3 ignored the bbox request in all eight cases and emitted
its trained `<think>...<answer>...` format. A structured system message plus an assistant-side JSON
prefix improved extraction but did not fully override that format.

| Model | Strict JSON | Recoverable bbox payload | Answer correct together with payload | Spatial cases with a payload box | Non-spatial cases with a valid abstention |
|---|---:|---:|---:|---:|---:|
| GPT-4o Stage3 step 1000 | 2/8 | 4/8 | 1/8 | 4/5 | 0/3 |
| Grok Stage3 step 1500 | 2/8 | 4/8 | 1/8 | 4/5 | 0/3 |

Among four spatially comparable payloads, GPT-4o had mean union IoU `0.1082`, mean predicted-area
precision `0.3837`, mean reference-area recall `0.1679`, and center-hit rate `2/4`. Grok had mean
IoU `0.0136`, precision `0.0750`, recall `0.0155`, and center-hit `0/4`.

One GPT-4o result is genuinely promising: for the correctly answered bilayered-epithelium case,
the model returned a small box fully inside one of three external duct-region boxes. Its union IoU
was only `0.0454` because the external reference covers three broad regions, while predicted-area
precision and center hit were both `1.0`. This illustrates why IoU alone is inadequate. Another
GPT-4o box overlapped a focal central reference with IoU `0.381`, but the associated answer was
wrong, so it cannot support a correct-decision explanation.

Freezing the model's original answer before asking only for localization improved strict formatting
to 2/3 in a tiny follow-up, but did not solve semantic reliability. One malformed response contained
a plausible duct box, one focal case incorrectly abstained, and no new confirmation result was
established.

## Conclusion

The present checkpoints can sometimes emit meaningful bbox coordinates, so the interface is
technically feasible. They cannot yet do so reliably enough for a reviewer-facing localization
claim. Joint answer-plus-bbox prompting also changes or corrupts answers and therefore is not a
fair replacement for the frozen answer evaluation.

A defensible future version should use a two-stage contract: freeze the ordinary answer first,
then request a bbox for that fixed option with grammar-constrained JSON decoding. Returned boxes
must be evaluated on a held-out set against independent pseudo boxes and later expert review, and
must pass deletion/retention tests against matched random regions. A small bbox-format SFT may be
needed, but its training annotations must be disjoint from the localization evaluation set.

## Artifacts

- Implementation: `scripts/run_direct_bbox_self_report.py`
- Unit tests: `scripts/test_run_direct_bbox_self_report.py`
- GPT-4o eight-case output:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/direct_bbox_self_report_confirmation8_gpt4o_stage3_20260810`
- Grok eight-case output:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/direct_bbox_self_report_confirmation8_grok43_stage3_20260810`
- Frozen-answer follow-up:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/direct_bbox_fixed_answer_smoke_gpt4o_stage3_20260810`
