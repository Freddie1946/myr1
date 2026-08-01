# GPT-4o penalty pilots and matched validation385 complete

Timestamp: 2026-08-01 16:35:20 +0800

## Outcome

All three matched 100-step coefficient pilots are now complete. The penalty-0.5 arm was rerun
fresh from the same fixed Stage2 parent after the original attempt-cap stop; it used a new Judge
root and did not reuse the failed arm's cache. Its complete distributed checkpoint-100 and
train-state audit passed the supervisor. No training run was restarted automatically.

The frozen deterministic PathMMU validation385 protocol then ran once for each completed model.
All 385 answers in every arm had an extracted choice and valid output format.

| Penalty | Steps | Process rewards | HTTP attempts | Valid Judge caches | Fallbacks | Judge cost | Validation correct | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 100 | 800 | 800 | 800 | 0 | $4.122075 | 234/385 | 60.7792% |
| 0.4 | 100 | 800 | 800 | 799 | 0 | $4.130780 | 237/385 | 61.5584% |
| 0.5 fresh01 | 100 | 800 | 803 | 800 | 0 | $4.1277625 | 240/385 | 62.3377% |

The 0.4 cache count is 799 because one exact within-arm duplicate was served from cache; one
billed response also failed strict validation and was retried. The recovered 0.5 arm required
three retry attempts. All ledgers ended with no unresolved reservations and no budget breach.

## Paired interpretation

Because every model evaluated the same 385 records, comparisons use paired correctness rather
than treating the scores as independent. Exact two-sided McNemar tests and exploratory paired
percentile bootstrap intervals (20,000 resamples, seed 42) give:

| Comparison | Improved | Regressed | Net correct | Accuracy change | Exact p | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 to 0.4 | 11 | 8 | +3 | +0.7792 pp | 0.6476 | [-1.2987, 3.1169] pp |
| 0.4 to 0.5 | 14 | 11 | +3 | +0.7792 pp | 0.6900 | [-1.8182, 3.3766] pp |
| 0.3 to 0.5 | 15 | 9 | +6 | +1.5584 pp | 0.3075 | [-0.7792, 4.1558] pp |

The point estimates increase monotonically from 0.3 to 0.5, but none of the paired differences is
statistically significant and every interval includes zero. These results support reporting the
coefficient sensitivity as stable over 0.3--0.5; they do not support selecting 0.5 as conclusively
better or changing the formal-run coefficient solely because it has the largest validation point
estimate. The formal coefficient decision remains separate and must be frozen before a long run.

## Budget accounting

The original stopped 0.5 arm's six unresolved reservations were conservatively settled at their
full $0.02 reserve, making that immutable failed-arm ledger $4.242265. Including that sunk cost,
all paid pilot expenditure is:

- completed 0.3 arm: $4.122075;
- completed 0.4 arm: $4.130780;
- original stopped 0.5 arm after conservative settlement: $4.242265;
- fresh recovered 0.5 arm: $4.1277625;
- aggregate: $16.6228825;
- remaining under the approved $18 ceiling: $1.3771175.

No further paid action was performed.

## Audit anchors

| Artifact | SHA-256 |
| --- | --- |
| 0.3 validation metrics | `b52a65ffe568b28682409330e80e8811339aeba5d6400262b90ee95154d6ea9f` |
| 0.3 validation predictions | `b54fc1581c5b93f3552e2c075f04bc677326bb6f1271d34aa50a0fea9b2adfd9` |
| 0.4 validation metrics | `da19ed08b5f019704b0a0a13997655ad14f29e27e993f98e0836eef2beb100a0` |
| 0.4 validation predictions | `73270c48e1b38e761b676bf380ad363a3ea22c544260ef89c31d5abe19db273e` |
| recovered 0.5 validation metrics | `e596c65ece3adf883d96858c412559e09c87b37cd0c4f2d6380f7fbe546e1f51` |
| recovered 0.5 validation predictions | `1f6e32c3972ce95565ad1d8c1904dd3e42456719d22b8cca29f3b8c72a1a77fc` |
| recovered 0.5 train-state audit | `7801a10398c184cd638a2cef7594bd2e35dc4bc9dc426ec5edac4ef12b4f43dd` |
| recovered 0.5 checkpoint trainer state | `2b637ffd0d71d00793de2e04aa7271fce9658fa0dc4fd53fc239b297be113d4e` |
| recovered 0.5 supervision validation | `b1ca269e99a16128d75b884d4d637d3c49f41783414a8483736462a7c110bc66` |
| completed supervisor status | `17cf470fe1772407edc45ffa02ec6e0fbaa01c5035404b9a02886ff35d731ac2` |
| settled original 0.5 ledger | `36196250acb7d899b7cae33b87ed9a5d381b02fae81e4f652d0d73a01599756f` |

This was a bounded development sensitivity study, not the formal long Stage3 run. PathMMU
test999 was not accessed by this recovery or by any of the three validation runs.
