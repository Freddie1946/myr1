# Stage3 OpenRouter smoke blocked by distillation terms

Timestamp: `2026-07-30T00:39:15+08:00`

Status: Stage3 training remains unstarted. Two router-level smoke attempts selected no provider
endpoint and incurred no charge. The GPT-4o Judge contract is blocked pending an explicitly
distillable replacement or documented provider permission.

## Attempt 1: parameter capability rejection

The first smoke used the approved model/provider/privacy/schema contract but sent the Judge output
limit as `max_tokens`. OpenRouter returned HTTP 404:

```text
No endpoints found that can handle the requested parameters.
```

Router metadata showed both Azure and OpenAI endpoints with `selected=false`. The live endpoint
catalog explains the mismatch: the Azure endpoint supports `max_completion_tokens`, whereas the
OpenAI endpoint supports `max_tokens`. Since Azure-only routing and
`require_parameters=true` were frozen, the router correctly rejected the request.

The client was corrected to use `max_completion_tokens=320`. This is the Judge JSON-output limit;
it does not alter the frozen 192-token local-model completion limit.

## Attempt 2: provider Terms-of-Service rejection

With the capability name corrected, OpenRouter returned HTTP 403:

```text
The request is prohibited due to a violation of provider Terms Of Service.
```

The error reported `provider_name=null`; router metadata again showed the Azure endpoint with
`selected=false`. This occurred before a provider was selected. The intended use of Judge outputs
as online training reward is a distillation/training use. OpenRouter provides a specific
`provider.enforce_distillable_text=true` control for such workflows and states that it restricts
routing to models whose authors explicitly permit output reuse for training/distillation.

The implementation now enables that control. It must not be removed merely to make GPT-4o route.
Doing so would bypass the provider's terms filter. GPT-4o 2024-08-06 therefore cannot be used for
this Stage3 training through the current OpenRouter contract unless explicit compatible permission
is established.

## Cost and ledger reconciliation

Before the attempts, after attempt 1, and after attempt 2:

- key usage: exactly USD `0`
- account total usage: exactly USD `0.358321845`
- selected endpoint: none
- successful generation ID: none

Both conservative USD 0.05 reservations were therefore released with the above evidence through
the ledger's audited `release_proven_unbilled` operation. Current pilot ledger:

- committed spend: USD `0`
- completed unique requests: `0`
- unresolved reservations: `0`
- unbilled releases: `2`
- budget breached: no

## Compliant live alternatives found read-only

The current OpenRouter models API supports simultaneous filters for image input, ZDR and
distillation permission. A read-only query found 23 models that additionally advertise structured
outputs. Scientifically relevant candidates include:

| Candidate | Independence | Prompt/output USD per M tokens | Note |
|---|---|---:|---|
| `qwen/qwen3-vl-235b-a22b-instruct` | same Qwen family | 0.21 / 1.90 | strongest directly relevant large VLM candidate; disclose same-family teacher |
| `moonshotai/kimi-k3` | independent | 3.00 / 15.00 | stronger independence; cost envelope comparable to or above GPT-4o |
| `moonshotai/kimi-k2.5` | independent | 0.57 / 2.85 | cheaper independent multimodal pilot |
| `mistralai/mistral-small-2603` | independent | 0.15 / 0.60 | cheap independent contemporary pilot |
| `qwen/qwen2.5-vl-72b-instruct` | same model family | 0.80 / 1.00 | historically closer, but creates stronger same-family circularity |

Any replacement changes the Stage3 reconstruction claim. It must be frozen as a contemporary
distillable-Judge pilot and cannot be reported as the manuscript's GPT-4o reconstruction.

## Implementation and tests

- `scripts/stage3_openrouter_judge.py`
  SHA-256 `fdfde18cb40cb06e352d2323a1a1096d46ed6350245b7da9171dc0d70d56939e`
- `scripts/test_stage3_openrouter_judge.py`
  SHA-256 `13f21cc014853fce9ff79b10d80c214469c7a2bc30b65e77c1bdd099466665c0`
- `scripts/grpo_pathmmu.py`
  SHA-256 `3c20993b29420ff061246935422a4a44d3959fcb3bd34b8f62d79bc4702741ee`

Fifteen CUDA-free Judge/schema/cache/ledger/mock-transport tests pass. The original PathMMU reward
tests also pass. The first budget test run contained one incorrect test expectation; the budget
code correctly refused a USD 0.11 projection under a USD 0.10 test cap. The expectation was
corrected and all tests then passed.

GPU 0 remains occupied by the previously authorized InternVL OmniMedVQA evaluation. GPUs 1--7 are
idle. Even if the Judge gate were resolved, the frozen eight-GPU Stage3 preflight would still wait
for GPU 0.
