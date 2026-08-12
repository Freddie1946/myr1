# Hugging Face storage inventory — 2026-08-12

The authenticated account is `Freddie1946`. The Hub API available to this
machine does not expose a numeric remaining-quota field, so this is a measured
inventory of repository file sizes, not a claim about remaining quota.

| Repository | Visibility | Measured files | Size |
|---|---:|---:|---:|
| `qwen2.5vl-3b-sft` | public | 7,525,389,803 B | 7.01 GiB |
| `qwen2.5vl-3b-20000sft` | public | 7,525,406,754 B | 7.01 GiB |
| `cot-3000-3b-750` | public | 7,525,398,378 B | 7.01 GiB |
| `cot-3000-3b-4500` | public | 7,525,461,726 B | 7.01 GiB |
| `cot-3000-7b` | public | 16,600,384,229 B | 15.46 GiB |
| `PathVLM-R1-SFT-n3000-seed42-epoch3` | private | 16,600,556,023 B | 15.46 GiB |
| `PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2` | private | 16,600,756,321 B | 15.46 GiB |
| `PathVLM-R1-SFT-n4000-control-seed42-epoch2` | private | 16,600,408,983 B | 15.46 GiB |
| `PathVLM-R1-Process-GRPO-GPT4o-n1000-seed42-epoch2` | private | 16,600,834,950 B | 15.46 GiB |
| `PathVLM-R1-Process-GRPO-Grok43-n1000-seed42-epoch3` | private | 16,600,367,486 B | 15.46 GiB |
| `PathVLM-R1-Stage2-Continued-RuleRL1000-seed42` | public + manual gate | 16,600,487,751 B | 15.46 GiB |
| `PathVLM-R1-Base-RuleRL4000-seed42` | public + manual gate | 1,519 B | negligible (upload incomplete) |

Measured total: approximately **120.8 GiB**. The planned model-only backups for
the completed n=4 and future n=8 step-1000 checkpoints are approximately
15.46 GiB each, before any duplicate files or metadata. Thus the projected
total is approximately **151.7 GiB plus small evaluation archives**.

No repository has been deleted. The likely space-saving candidates are the old
public 3B snapshots (`qwen2.5vl-3b-sft`, `qwen2.5vl-3b-20000sft`,
`cot-3000-3b-750`, `cot-3000-3b-4500`) and any duplicate private checkpoint
whose paper role is no longer needed. Deletion requires the owner's explicit
selection; this inventory is the confirmation point.

