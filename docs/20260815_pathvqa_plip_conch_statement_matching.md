# PLIP / CONCH PathVQA 问题-答案陈述匹配

## 冻结合同

- 数据：PathVQA test 的全部 3,362 道 `yes_no` 问题（1,816 Yes；1,546 No）。
- 每题构造两条候选文本：
  - `Answer: Yes. Question: {question}`
  - `Answer: No. Question: {question}`
- 以图像 embedding 与两条文本 embedding 的余弦相似度较大者作为预测。
- 答案词置于开头，避免 CLIP-family 文本上下文截断把唯一候选差异截掉。
- 这是 `exploratory_image_question_answer_statement_matching_v1`，属于 retrieval-style 图文匹配诊断，不是自由生成 VQA，也不评价推理质量。

## 全量结果

| Model | Correct / 3362 | Accuracy | Wilson 95% CI | Balanced accuracy | Yes recall | No recall | Predicted Yes rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLIP | 1624 | 48.30% | [46.62%, 49.99%] | 48.94% | 41.02% | 56.86% | 42.00% |
| CONCH | 1811 | 53.87% | [52.18%, 55.55%] | 52.53% | 69.16% | 35.90% | 66.84% |

CONCH 的点估计高于 PLIP，但两者都存在明显类别倾向：PLIP 偏向 No，CONCH 偏向 Yes。因此正文应同时报告 balanced accuracy/分类召回，不能只给 accuracy。该结果只能作为病理图文 encoder 的补充对照，不能与能够遵循指令并生成 `<think>/<answer>` 的 VLM 作无条件排名。

## 可审计产物

- PLIP：`pathvlm_revision_eval_a100/runs/pathvqa_statement_matching_20260815/plip_full_yesno3362/`
- CONCH：`pathvlm_revision_eval_a100/runs/pathvqa_statement_matching_20260815/conch_full_yesno3362/`
- 每个目录均含 `run_config.json`、`metrics.json`、`predictions.jsonl`。
- 实现：`scripts/run_pathvqa_statement_matching.py`
