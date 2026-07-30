# Claude-4.5 manuscript identity corrected to Haiku 4.5

Timestamp: `2026-07-29T23:24:21+08:00`

The user supplied the missing identity provenance: the manuscript Table II row labelled
`Claude-4.5` refers to **Claude Haiku 4.5**, not Claude Sonnet 4.5.

The frozen OpenRouter candidate for the contemporary full rerun is therefore:

- OpenRouter model ID: `anthropic/claude-haiku-4.5`
- provider model ID exposed by current endpoints:
  `anthropic/claude-4.5-haiku-20251001`
- input modalities: text, image and file
- required capabilities: temperature and strict structured outputs
- audit-time list price: USD 1/M input tokens and USD 5/M output tokens

This addendum supersedes only the unresolved identity choice in
`20260729_231550_openrouter_hosted_baseline_and_stage3_judge_audit.md`. It does not rewrite that
timestamped audit. Claude Sonnet 4.5 remains only a possible independent Stage3 secondary judge,
not the manuscript baseline.

No paid request, adapter smoke, hosted baseline evaluation or Stage3 call was performed by this
identity correction. Before evaluation, the Haiku 4.5 provider, routing/privacy policy, prompt,
parser, retry policy and maximum budget still require a frozen launch manifest and external
credential.
