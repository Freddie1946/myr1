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

初步 PathMMU Val385：204/385 = 52.9870%，格式385/385，choice extracted 384/385，无截断。其余核心评测正在并行运行，不能在完成前下最终结论。
