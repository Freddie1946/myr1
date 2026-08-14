# 最终非人工结果与迁移交接（2026-08-14）

## 目标边界

本次闭环包含：数据规模/配比消融、最终模型重复推理置信区间、统一域内/域外报表、外部基线与异构 Judge 结果、可解释性计算结果、HF/Git 双备份。唯一明确排除的缺口是后续由用户组织的病理专家 ROI 与生成质量人工评分。

当前个人 Codex 使用的本地 sessions/history/SQLite 状态可以通过
`scripts/create_codex_online_snapshot.py` 做一致性在线快照，而不必退出当前会话。该快照
不等同于客户端官方导出，但可用于同一 Codex 环境的迁移恢复；认证文件必须排除并在新机
重新登录。本仓库中的计划、协议、审计和结果表仍是长期可读的决策记忆。

## 主结果入口

- 统一自动结果表：`docs/20260814_final_unified_results_tables.md`
- 机器可读总表：`protocol/final_unified_results_tables_20260814.json`
- 核心最小配对统计：`protocol/final_core_pathmmu_statistics_20260814/cluster_bootstrap_results.json`
- 旧合同外部基线和 2,880 项异构 Judge：`docs/20260810_final_closed_benchmark_and_multijudge_results.md`
- 可解释性/因果视觉证据：Stage3 step1500 final-eval 下的 `reference_evidence_*` metrics；外部模型 ROI 不是病理专家真值。
- 最终 fail-closed 完备性审计：`protocol/final_nonhuman_data_completion_audit_20260814.json`

## 评测合同边界

- PathVQA 主报告保留原始 Yes/No 答案接口；历史 short-answer 和当前结构化
  reasoning+Yes/No 子合同必须显式标注。
- 固定 `A=Yes/B=No` 是同题配对的补充接口敏感性诊断，不覆盖或池化到 Yes/No 主分数。
- PathMMU Test999 已长期用于开发诊断，不能描述为独立未触碰最终测试。
- 五次随机推理 t 区间衡量解码 seed 波动；case/image-cluster bootstrap 衡量测试样本不确定性，两者不能互相替代。
- 不扩展全模型 paired flip/badcase；只保留最终核心模型的最小 paired image-cluster bootstrap/McNemar 证据。

## 恢复/复现

1. 克隆 `codex/a100-stage3-eval` 并核对最终 Git commit。
2. 恢复工作区环境和私有 HF 认证，不复制任何 token/key 文件。
3. 从 HF 下载归档清单/校验记录并核对 SHA-256。完整最终 tar 因私有 HF 配额满而保留在本机；此前的 pre-completion 完整快照仍在 HF。若迁移前仍未扩容，需用受控文件传输复制本地最终 tar，并验证 `36f10566c7087f4603ba3abee95ec84a47b400ae4aabd1b27a1778c3a8f435cd`。
4. 运行 `scripts/audit_final_nonhuman_data_completion.py --require-final-backup`；只有状态为 `complete` 才表示除人工专家项外全部闭环。
5. 若继续专家验证，必须盲于模型 attention/RISE 结果，单独记录 ROI 诊断相关性与生成质量评分。

## 未授权/未执行内容

- 未删除训练权重或其他模型；任何后续存储清理仍需用户逐项确认。
- 未上传凭据、原始图片、优化器 shard 或超大可解释性中间张量。
- 未把人工专家评分伪装成外部模型评分，也未把外部模型 ROI 称为 pathology ground truth。

## HF 最终备份状态

- 私有 dataset：`Freddie1946/PathVLM-R1-Evaluation-Archive-20260814`
- 最终 manifest/verification 以及 27 个 essential 结果文件（总表、审计、核心统计、新消融 metrics、重复推理 summaries）已上传；4 项 fresh-download SHA spot-check 全部一致，revision `1d613f216204e465c574664c5fc80f3ecfaf7d13`。
- 131 MiB 完整最终 tar 上传被 HF 明确以“private repository storage limit reached”拒绝；这不是 token 或网络错误。
- 未经授权没有把数据改为公开。完整 tar 本地路径和 SHA 见 `protocol/final_hf_results_backup_20260814.json`。
