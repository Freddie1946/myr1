# Claude Sonnet 4.6 non-training feasibility check

Timestamp: `2026-07-30T00:59:55+08:00`

Status: conditional interface feasibility established, but the current OpenRouter account/key
cannot route a Claude Sonnet 4.6 request. Stage3 remains unstarted and no Claude output has been
used for training, sample selection, reward computation or model updates.

## Scope

The user requested a feasibility check before obtaining supplemental written permission for using
Claude Sonnet 4.6 as a Stage3 Judge. The check was deliberately limited to:

- read-only model and endpoint inspection;
- one synthetic 64x64 blue/red image;
- a non-medical visual statement;
- an explicit statement that the response would not be used for training;
- strict JSON Schema output;
- no connection to the GRPO reward path.

No PathMMU image, candidate completion, checkpoint or training process was submitted.

## Read-only capability result

OpenRouter's live endpoint catalog reported:

- requested model: `anthropic/claude-sonnet-4.6`;
- modalities: text, image and file input to text output;
- context length: 1,000,000 tokens;
- structured output and `response_format` support;
- Anthropic, Azure, Google and Amazon Bedrock endpoints;
- list price: USD 3/M prompt tokens and USD 15/M completion tokens.

This establishes that the model's advertised interface can represent the Stage3 Judge request.
It does not by itself establish that this account may route the request.

## Runtime controls

Three router-level checks occurred:

1. Anthropic-only with `zdr=true` and `data_collection=deny`: HTTP 404 because no Claude Sonnet
   4.6 endpoint matched the Zero Data Retention requirement.
2. Anthropic-only without ZDR, retaining `data_collection=deny`: HTTP 403 with the generic
   provider Terms-of-Service message; no endpoint was selected.
3. Azure-only without ZDR on the same synthetic non-training request: HTTP 403 with the same
   generic message; no endpoint was selected.

The second and third controls rule out the submitted pathology image/content and an explicit
training-use statement as necessary causes of the earlier generic 403. The response still does
not reveal the precise rule. Plausible unresolved causes include account eligibility, region,
provider qualification or another router/provider restriction.

## Billing evidence

Across all three controls:

- key usage remained exactly USD `0`;
- account total usage remained exactly USD `0.358321845`;
- no provider endpoint was selected;
- no successful generation ID was returned;
- no paid model call occurred.

## Operational conclusion

Claude Sonnet 4.6 is interface-compatible in the catalog, but it is not runnable through the
current OpenRouter key. Written training-use permission is necessary but may not be sufficient:
before any Stage3 use, the project also needs a route that successfully passes account/provider
eligibility, plus an explicit data-retention decision for the evaluation images.

Acceptable future routes are:

1. OpenRouter confirms and enables this account/key for Claude under the supplemental permission;
2. a direct Anthropic, Azure or Bedrock credential covered by the written permission and the
   required data-processing terms.

Claude Code pointed at the current OpenRouter key would use the same routing path and therefore
would not resolve this blocker.

The production Judge constants remain unchanged. `provider.enforce_distillable_text=true` remains
mandatory until written permission and a compatible route are both documented.

## Credential incident

During local secret-file metadata inspection, a redaction expression failed to cover the
`export NAME=value` form and printed the current key into tool output. The key must be revoked and
replaced before any further paid or production API activity.
