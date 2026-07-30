# Stage3 HTTP 403 cause attribution corrected

Timestamp: `2026-07-30T00:45:32+08:00`

Status: scientific/audit correction. Stage3 remains unstarted and OpenRouter committed spend
remains USD 0.

The prior report correctly recorded the exact HTTP 403 response, lack of provider selection and
zero billing. It was too definite in attributing the rejection specifically to distillation
terms.

## What the response proves

- OpenRouter returned HTTP 403 with the generic message:
  `The request is prohibited due to a violation of provider Terms Of Service.`
- `provider_name` was null.
- the only remaining Azure endpoint had `selected=false`.
- no successful generation ID was returned.
- key usage and account total usage did not change.
- router metadata contained no detailed `pipeline` or named guardrail stage.

Therefore the request was blocked before an upstream provider served it, but the returned evidence
does not identify the precise matched rule.

## What OpenRouter could infer from the request

The request did not contain a literal statement that its output would update model weights.
However, it did contain multiple signals consistent with a model-evaluation/reward workflow:

- application title: `PathVLM-R1 Stage3 Pilot`;
- repository referer identifying the PathVLM project;
- system role: pathology reasoning-process auditor;
- user section: `CANDIDATE COMPLETION TO AUDIT`;
- a reference answer plus six structured process-quality events.

An automated terms/content classifier could infer Judge or training-related use from this
combination. Alternatively, it could have classified the malignant-lesion pathology question or
image as a restricted medical use, or applied another account/provider permission rule.

## Correct conclusion

Distillation is a material compliance concern because the output is intended to become online
GRPO reward, and future training requests should use an explicitly distillable model unless
compatible permission is established. But the observed 403 alone does not prove that distillation
was the rule that fired.

A controlled differential diagnosis is required to attribute the 403:

1. benign non-medical structured-output control under the same model/provider/privacy settings;
2. de-identified educational pathology request without candidate-completion auditing;
3. non-medical candidate-completion audit with the same six-event-shaped evaluator structure;
4. the exact research request against an explicitly distillable vision model.

These controls must accurately describe their immediate diagnostic purpose. They must not conceal
the intended training use, weaken privacy requirements, or be used to evade a provider restriction.
