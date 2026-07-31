# AIGCBest closed-baseline provider qualification gate

The public pricing catalog at `https://api2.aigcbest.top/api/pricing` advertises
vision-capable entries relevant to the manuscript, including GPT-4o
2024-11-20, Claude Haiku 4.5, Claude Sonnet 4.6, Grok 4 Fast, Llama 3.2 Vision
11B/90B, Qwen VL Plus and Kimi K2.6.  The unauthenticated `/v1/models` endpoint
returns HTTP 401, and this workspace contains no credential for this provider.

More importantly, the provider's public notice at
`https://api2.aigcbest.top/api/notice` explicitly states that the service is not
provided to users in mainland China and instructs restricted users to stop
using it.  This machine/user context is mainland China.  No paid request and no
PathVQA/OmniMedVQA image was sent to this provider.

This is a provider-eligibility gate, not merely a missing-key gate.  It must not
be bypassed.  Closed-model baselines should instead use official or otherwise
compliant providers, or be executed by a lawfully eligible collaborator under
written authorization and a separately frozen data/price/privacy contract.

