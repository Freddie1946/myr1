# 2026-08-15 最终迁移与人工复核闭环

## 当前实验状态

- 论文冻结范围内的非人工实验与评测已经完成。
- `protocol/final_revision_assets_audit_20260815.json` 共检查 85 项，失败 0 项，状态为 `complete_except_human_ratings`。
- 核心模型、PathMMU、PathVQA、OmniMedVQA、MMMU 结果、数据配比消融、重复推理统计、三模型 Judge、可解释性产物和人工复核输入均在该审计范围内。
- 当前没有训练、推理或评测进程；GPU 0–7 均无计算进程。唯一常驻任务是本地专家复核网页服务。
- 后续可选的更多种子、更大人工 panel 或其他探索不是当前审稿回复的必需缺口，除非论文范围再次改变。

因此，当前唯一未完成的论文结果单元是病理专家人工复核及其统计汇总。

## 人工材料收回后与什么比较

1. 奖励六事件：人工按同一 0.4 公式得到的过程分，主要与训练期 GPT-4o 过程分逐病例比较；Claude Sonnet 4.6 是第二参考。主终点是分数差，不是简单事件总体一致率。
2. ROI：外部模型区域先由专家确认或修订，再用确认后的区域重算 reference deletion 相对 area-matched random deletion 的 margin loss 和 flip。外部框本身不能直接称为真值。
3. 生成质量：专家在 A/B 盲化状态下比较冻结的 Stage2/Stage3 回答；完成统计后才解盲。Claude/Gemini 只用于外部一致性分析。

完整导入命令、比较表、解释边界和汇总命令见 `docs/20260815_human_review_packet_instructions.md`。

## 迁移资产层级

1. GitHub `codex/a100-stage3-eval`：代码、测试口径、实验协议、论文修订文本、完整表格、结果血缘和恢复指南。
2. private HF Codex snapshots：secret-free sessions/history/SQLite；不含 token、API key、`auth.json` 和缓存。
3. manual-gated HF migration archive：评测 JSON/JSONL、metrics、日志、人工材料与解释性产物。
4. 独立模型仓库：Stage3 500/1000/1500、SFT/RL 起点和关键对照 checkpoint。
5. 可从固定上游 revision 重建的基础模型与数据集：不重复归档无权再分发的第三方内容。

迁移不等于逐字节复制整个工作区：优化器分片、可下载缓存、重复权重、失败冒烟和非关键机制 checkpoint 按既定政策排除。所有科学结论所需的原始预测、结果 JSON、测试口径与关键模型均已纳入远端验证清单。
