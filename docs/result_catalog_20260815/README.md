# 评测结果追溯目录（2026-08-15）

本目录解决两个问题：一个 JSON 是哪个模型产生的，以及它使用了哪套评测协议。

## 判定规则

不要只依赖目录名。每个正式结果以三件套为准：

1. `run_config.json`：模型/adapter 路径及 hash、数据路径及 hash、prompt/generation contract、解码参数；
2. `predictions.jsonl`：逐题原始生成和目标盲解析结果；
3. `metrics.json`：聚合指标与 predictions SHA-256。

`result_lineage.json` 对两个 runs 根目录中的每个结果目录建立了机器可读记录，并给出 `traceability_level`：

- `complete`：模型、模型 hash、数据、数据 hash、prompt/合同、解码参数、逐题预测和汇总均存在；
- `strong_partial`：模型/数据/合同明确，但缺少部分 hash 或逐题文件；
- `legacy_partial`：模型和数据可定位，但旧结果缺少明确 prompt/generation contract；
- `ambiguous_nonstandard`：非标准诊断/汇总，不能仅凭名字写入论文主表。

当前共索引 891 个结果目录：{'complete': 340, 'legacy_partial': 112, 'ambiguous_nonstandard': 286, 'strong_partial': 153}。

## 关键协议边界

- OmniMedVQA 正式修订口径：`omnimed_domain_think_answer_v4_1024`；旧 64-token 结果不可混入。
- PathVQA 主报告为原始自由 Yes/No；固定 `A=Yes/B=No` 仅作接口敏感性诊断。
- PathMMU Test999 已用于工程诊断，不能再声称是未触碰测试集。
- `paper_tables.md` 只从显式 JSON 路径取值；运行中的格子写 `RUNNING`，不借用不兼容历史分数。
