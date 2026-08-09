# Extended bbox output-format probe

Date: 2026-08-10 02:00 Asia/Shanghai

## Question

Can the GPT-4o Stage3 checkpoint report an image-space decision box by embedding coordinates in
its learned `<think>...</think><answer>...</answer>` tags, or by adding a separate localization
turn after its answer is frozen?

This is an exploratory validation-only experiment. It does not access PathMMU test data and does
not treat the external Claude annotations as expert ground truth.

## Formats searched

The first three frozen panel cases were tested under eight deterministic output contracts:

1. prefilled plain coordinates directly after `<think>`;
2. prefilled XML `<evidence><bbox .../></evidence>` inside `<think>`;
3. prefilled Qwen `<ref>...</ref><box>...</box>` inside `<think>`;
4. bbox nested inside `<answer>`;
5. bbox encoded as attributes of `<answer>`;
6. bbox encoded as attributes of `<think>`;
7. a second-turn `<bbox evidence_type="...">[...]</bbox>` response after freezing the ordinary
   answer;
8. a second-turn `VISUAL_EVIDENCE: ...; boxes=[[...]]` response after freezing the answer.

The two second-turn formats were then confirmed on the same eight-case panel already frozen for
the direct-bbox confirmation: panel indices 0, 1, 2, 4, 9, 10, 20 and 22.

Three separate properties are reported:

- **coordinate-valid**: the response contains a geometrically valid box list or a valid empty-box
  abstention;
- **protocol-compliant**: it also uses one of the requested evidence-type values rather than a
  placeholder or free-form label;
- **spatial agreement**: box overlap with the external pseudo-reference on cases that both sides
  classify as spatial.

## Three-case format search

Prompt-only instructions without an assistant prefix were generally ignored. Prefilling
`<think>VISUAL_EVIDENCE`, XML, or Qwen tags increased syntactic continuation but did not provide a
reliable combined answer-and-localization interface:

- plain-in-think produced no coordinate-valid payload;
- XML-in-think produced one empty diffuse payload but no spatial box;
- Qwen prefill produced three parseable *empty* `not_localizable` payloads, not actual boxes, and
  changed two of the three baseline answers;
- bbox nested in `<answer>`, `<answer>` attributes, and `<think>` attributes produced no valid
  payloads.

Only a separate second turn produced actual image coordinates while leaving the preceding answer
unchanged.

## Eight-case second-turn confirmation

| Second-turn encoding | Answer preserved | Coordinate-valid | Protocol-compliant | Spatial applicability | Comparable spatial cases | Mean union IoU | Center hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<bbox evidence_type="...">[...]</bbox>` | 8/8 | 7/8 | 2/8 | 5/8 | 4 | 0.0883 | 1/4 |
| `VISUAL_EVIDENCE: ...; boxes=[[...]]` | 8/8 | 4/8 | 0/8 | 3/8 | 3 | 0.2647 | 3/3 |

The 50% answer accuracy in this panel is inherited from the frozen first-turn answers; the
localization follow-up neither improves nor changes those answers.

Visual inspection supports a narrow interpretation of the plain-coordinate result. In one case,
the returned box covered most of the central lesion (IoU 0.6883). In another, two small returned
boxes fell inside externally marked bilayered ducts: high precision but low recall, so IoU remained
small. A third returned box also had its center inside the reference. Therefore the second-turn
coordinates are not uniformly random, but output compliance, abstention and coverage remain too
unstable for reviewer-facing localization evidence.

The tag variant was more likely to emit numbers, but often wrote free-form values such as
`LOCATION` or a diagnosis into the `evidence_type` attribute. Its spatial agreement was weaker.
The plain variant often copied placeholders or malformed the evidence type even when its box was
spatially plausible.

## Decision

Prompt-only bbox self-report is not selected as the primary interpretability method. If this route
is revisited, use the two-stage design: freeze the ordinary answer, then constrain a separate
localization decoder to a small grammar. Before any claim, validate those boxes on a larger
pre-frozen panel and test their causal relevance through matched deletion/retention controls.

A small bbox-format SFT is the fallback if constrained decoding fixes syntax but not semantics.
Its pseudo-labeled training set must be disjoint from localization evaluation and later expert
review.

## Artifacts

- Runner: `scripts/run_bbox_output_format_probe_extended.py`
- Tests: `scripts/test_run_bbox_output_format_probe_extended.py`
- Eight-format smoke:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/bbox_output_format_probe_extended_gpt4o_stage3_20260810`
- Corrected eight-case confirmation:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/bbox_output_second_turn_confirmation8_v2_gpt4o_stage3_20260810`

