# Hugging Face storage inventory — 2026-08-12

The authenticated account is `Freddie1946`. The Hub API available to this
machine does not expose a numeric remaining-quota field, so this is a measured
inventory of repository file sizes, not a claim about remaining quota.

| Repository | Visibility | Measured files | Size |
|---|---:|---:|---:|
| `cot-3000-7b` | public | 16,600,384,229 B | 15.46 GiB |
| `PathVLM-R1-SFT-n3000-seed42-epoch3` | private | 16,600,556,023 B | 15.46 GiB |
| `PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2` | private | 16,600,756,321 B | 15.46 GiB |
| `PathVLM-R1-SFT-n4000-control-seed42-epoch2` | private | 16,600,408,983 B | 15.46 GiB |
| `PathVLM-R1-Process-GRPO-GPT4o-n1000-seed42-epoch2` | private | 16,600,834,950 B | 15.46 GiB |
| `PathVLM-R1-Process-GRPO-Grok43-n1000-seed42-epoch3` | private | 16,600,367,486 B | 15.46 GiB |
| `PathVLM-R1-Stage2-Continued-RuleRL1000-seed42` | public + manual gate | 16,600,487,751 B | 15.46 GiB |
| `PathVLM-R1-Base-RuleRL4000-seed42` | public + manual gate | 1,519 B | negligible (upload incomplete) |

The four old public 3B repositories listed above were deleted with explicit
owner approval. The remaining measured model-repository total is approximately
**108.22 GiB**. The separate private evaluation datasets occupy approximately
**0.65 GiB**. The planned model-only backups for n=4 and n=8 are approximately
15.46 GiB each, giving a projected post-backup total of approximately
**139.79 GiB**.

The remaining likely space-saving candidate is the older public
`cot-3000-7b` snapshot (approximately 15.46 GiB); it has not been deleted.
No other repository has been deleted.
