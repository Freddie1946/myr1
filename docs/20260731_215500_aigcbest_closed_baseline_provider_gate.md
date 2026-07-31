# AIGCBest closed-baseline provider qualification gate

The public pricing catalog at `https://api2.aigcbest.top/api/pricing` advertises
vision-capable entries relevant to the manuscript, including GPT-4o
2024-11-20, Claude Haiku 4.5, Claude Sonnet 4.6, Grok 4 Fast, Llama 3.2 Vision
11B/90B, Qwen VL Plus and Kimi K2.6.  The unauthenticated `/v1/models` endpoint
returns HTTP 401, and this workspace contains no credential for this provider.

The provider's public notice at `https://api2.aigcbest.top/api/notice` excludes
users in mainland China.  The user subsequently clarified that they are located
in Taiwan, not mainland China.  The prior conclusion that this notice excluded
the user was therefore incorrect and is withdrawn.  No paid request and no
PathVQA/OmniMedVQA image had been sent before this correction.

Provider eligibility is no longer blocked by that particular notice.  Execution
still requires a provider-specific credential, an explicit paid-evaluation
budget, and confirmation of the account terms/data-handling contract.  The
workspace currently contains only an OpenRouter credential, not an AIGCBest
credential.
