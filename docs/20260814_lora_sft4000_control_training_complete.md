# L-r16 SFT3000 + RL1000-as-SFT 对照训练完成

- Parent：`formal_selected_sft3000_20260811/output/checkpoint-80`
- Parent adapter SHA256：`60148876d9dd35339c15bcce845259ee0ed0bdedbd51c681261f1a029a6e4abb`
- 数据：与正式 rule-RL 完全相同的1000条记录
- 数据 SHA256：`5a818b3b4f3c9e5e1a17579111ff3e21008cf8124c4a05cb74eb2dcd326bbd73`
- Vision encoder：frozen
- Projector：frozen
- Language：继续训练同一个 LoRA r16 / alpha32 adapter
- GPU：8 x A100 80GB
- micro-batch/GPU：12
- global batch：96
- cutoff：1024
- additional SFT epochs：2
- prompt exposure：2000
- optimizer steps：22
- runtime：130.9227 s
- train loss：0.9811336
- final adapter SHA256：`360ac115ac1c24ffeb76baf8d099ff66804c7897cdaa66b8ed5dfe9a0c7686e5`
- final adapter config SHA256：`dc7300c0da05659de845419465b89cd1dcaf14e6f54ccf796fda21ae2439add7`
- checkpoint：epoch1/step11、epoch2/step22、final adapter
- PathMMU test was not used for training or checkpoint selection.

两步八卡冒烟先确认：

- 原 adapter 被以 trainable 状态继续加载，没有创建第二套 LoRA；
- trainable parameters = 40,370,176；
- loss/gradient有限；
- 与并行单卡评测共置时无 OOM。

## 已完成的核心评测

| 指标 | 结果 |
|---|---:|
| PathMMU Val385 | 204/385 = 52.9870% |
| PathMMU Test999 | 565/999 = 56.5566% |
| PathVQA Val512，自由 Yes/No | 309/512 = 60.3516% |
| PathVQA Val512，固定 A=Yes/B=No | 330/512 = 64.4531% |
| PathVQA Test3362，固定 A=Yes/B=No | 2122/3362 = 63.1172% |
| MMMU non-medical 116 | 65/116 = 56.0345% |
| PathVQA Normal（forced-logit视觉诊断） | 326/512 = 63.6719% |
| PathVQA cyclic image shuffle | 256/512 = 50.0000% |
| PathVQA global-mean blank | 254/512 = 49.6094% |

视觉依赖差值：

- Normal - Shuffle = +13.6719 pp
- Normal - Blank = +14.0625 pp

生成式 PathMMU Val/Test 均无 generation-cap hit；PathVQA 两种生成合同均100%可解析、无截断。固定 A/B 接口比自由 Yes/No 高4.10 pp，说明统一输出接口仍然重要。

## PathMMU Test999 配对结论

| 对比 | 左→右提升 | 退化 | 准确率差 | 95% image-cluster bootstrap CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| LoRA-SFT4000 → L-r16-SFT3000 parent | 56 | 55 | +0.10 pp | [-1.95, +2.15] pp | 1.0000 |
| LoRA-SFT4000 → full rule-RL n4 step1000 | 115 | 69 | +4.60 pp | [+2.06, +7.20] pp | 0.000860 |
| LoRA-SFT4000 → full rule-RL n8 step1000 | 136 | 67 | +6.91 pp | [+4.15, +9.65] pp | 1.46e-6 |

因此，在相同额外1000条数据、相同2000 prompt exposure的对照下，继续 LoRA-SFT 与其 SFT3000 parent 在 Test999 上几乎完全持平，而 full rule-RL n4/n8 获得显著净提升。该结果支持“提升来自RL目标，而非仅继续看相同数据”的解释。PathVQA Test3362为63.12%，100%可解析、100%严格格式且无截断；OmniMedVQA8518仍在运行，最终泛化结论待其完成后更新。
