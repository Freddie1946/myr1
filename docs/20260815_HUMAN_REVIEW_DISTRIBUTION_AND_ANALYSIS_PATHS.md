# 人工复核分发、回收与统计单一入口

## 1. 哪个包发给专家

默认分发中英双语、带外部参考、本地保存与 ZIP 导出的便携包：

```text
/home/dataset-assist-0/czy/wjy/backup_archives/human_review_distribution_20260815_v3/pathvlm_human_review_bilingual_assisted_20260815.tar.gz
```

不要把负责人包发给专家。带外部参考的结果必须标为 `reference-assisted expert review`；若论文需要独立盲评主结果，应改发不显示参考的 v2 reviewer-blinded 包。

HF 备份位置：

```text
repo: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: ef660a6a82a931df3f0c07b7aa05c4c4041bd607
path: increments/20260815_bilingual_expert_review_v1/human_review/pathvlm_human_review_bilingual_assisted_20260815.tar.gz
```

下载命令：

```bash
hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset \
  --revision ef660a6a82a931df3f0c07b7aa05c4c4041bd607 \
  --include 'increments/20260815_bilingual_expert_review_v1/human_review/pathvlm_human_review_bilingual_assisted_20260815.tar.gz' \
  --local-dir /path/to/reviewer_distribution
```

## 2. 哪个包由负责人保管

负责人统计、解盲与核对模型身份时使用：

```text
/home/dataset-assist-0/czy/wjy/backup_archives/human_review_distribution_20260815_v2/pathvlm_human_review_owner_complete_20260815.tar.gz
```

HF 备份位置：

```text
revision: b182e8de003412c0d68ea9d31fa53a81d99bc8ba
path: increments/20260815_results_material_index_v2/human_review/pathvlm_human_review_owner_complete_20260815.tar.gz
```

该包包含解盲映射、自动评分及内部 key，不能与专家分发包混放。

## 3. 专家结果回收到哪里

专家返回的原始 ZIP 先原样存入：

```text
/home/dataset-assist-0/czy/wjy/human_review_returns/reward/
```

校验并导入：

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
python3 scripts/import_expert_reward_exports.py \
  /home/dataset-assist-0/czy/wjy/human_review_returns/reward/*.zip \
  --output-dir /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815
```

导入器默认拒绝 partial 包、manifest/SHA 不一致、Reviewer ID 不合法、缺题、非六布尔字段及同名异内容覆盖。

`expert_submissions_20260815/path1/` 是网页功能测试记录，不是正式专家结果，正式统计必须排除。正式编号使用预先冻结的匿名 ID，例如 `pathologist_A`、`pathologist_B`。

## 4. 奖励复核统计命令

主要比较对象是训练期 GPT-4o；Claude Sonnet 4.6 仅作第二参考。六事件均用训练时相同的 0.4 公式换算为过程分，主结果报告人工−目标分数的有符号均值、MAE、RMSE、完全同分率、差值不超过 0.2 的比例及病例 bootstrap 95% CI。事件 agreement 只用于定位分歧来源。

在删除 `path1` 等测试目录或将正式评分单独放入一个干净目录后执行：

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
python3 scripts/analyze_human_reward_score_differences.py \
  --ratings-dir /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815 \
  --gpt-key /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/reward_agreement_60case_20260814/internal_selection_key.jsonl \
  --claude-references /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/references.jsonl \
  --output-json /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815/final_score_differences.json \
  --output-md /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815/final_score_differences.md
```

更安全的做法是把正式导入输出指定到一个新的 `formal_expert_submissions_20260815/`，从源头排除测试记录。

## 5. ROI 与生成质量怎样统计

- ROI：专家首先判断区域病理相关性、覆盖度、遗漏和修框需求；冻结专家确认区域后，重算 reference deletion 相对 area-matched random/neighbor deletion 的 target-margin loss、flip rate 和配对置信区间。不能把 Gemini/Opus 框直接称为真值，也不以框 IoU 作为唯一主终点。
- 生成质量：在 A/B 身份仍盲化时汇总 `A better / B better / Tie` 和分项 1–5 分；冻结统计后才用 owner key 解盲为 Stage2/Stage3。报告净偏好、置信区间和专家间一致性；Claude/Gemini 只作 concordance 分析。
- 两位专家的原始文件、清洗副本、分歧、第三专家仲裁记录和最终输出均需分别保存，不能覆盖原始评分。

ROI 与生成质量在尚未收到人工结果时不生成伪造的空统计。字段说明、完整统计协议和论文表述边界见 `docs/20260815_human_review_packet_instructions.md`。

## 6. 相关脚本和说明的远端备份

下列文件已经在同一个 manual-gated HF revision 中：

```text
increments/20260815_bilingual_expert_review_v1/tools/import_expert_reward_exports.py
increments/20260815_bilingual_expert_review_v1/tools/analyze_human_reward_score_differences.py
increments/20260815_bilingual_expert_review_v1/documents/human_review_packet_instructions.md
```

本单一入口文档另在 `increments/20260815_final_migration_guide_v1/` 保存，并由 `protocol/final_migration_guide_hf_backup_20260815.json` 记录远端 revision 与 SHA。
