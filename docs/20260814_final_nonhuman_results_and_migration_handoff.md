# 最终非人工结果与迁移交接（2026-08-14）

## 目标边界

本次闭环包含：数据规模/配比消融、最终模型重复推理置信区间、统一域内/域外报表、外部基线与异构 Judge 结果、可解释性计算结果、HF/Git 双备份。唯一明确排除的缺口是后续由用户组织的病理专家 ROI 与生成质量人工评分。

原始 Codex 平台对话不是工作区文件，无法由脚本直接导出。本仓库通过计划、协议、审计、结果表和本交接文档保存可迁移的决策记忆；平台原始对话需使用客户端自身的导出能力另行保存。

## 主结果入口

- 统一自动结果表：`docs/20260814_final_unified_results_tables.md`
- 机器可读总表：`protocol/final_unified_results_tables_20260814.json`
- 核心最小配对统计：`protocol/final_core_pathmmu_statistics_20260814/cluster_bootstrap_results.json`
- 旧合同外部基线和 2,880 项异构 Judge：`docs/20260810_final_closed_benchmark_and_multijudge_results.md`
- 可解释性/因果视觉证据：Stage3 step1500 final-eval 下的 `reference_evidence_*` metrics；外部模型 ROI 不是病理专家真值。
- 最终 fail-closed 完备性审计：`protocol/final_nonhuman_data_completion_audit_20260814.json`

## 评测合同边界

- 当前核心 PathVQA 主表使用固定 `A=Yes/B=No`、结构化 think/answer、较高 token 上限的生成合同。
- 外部基线多数使用历史 short-answer/自由 Yes-No 合同，已经全量完成但必须另表报告，不能伪装为同 prompt 结果。
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
- 最终 manifest 与 verification 已上传并通过 fresh-download SHA 校验，revision `b8a0b0fd95269113b7d4c6e717523781bc10ffc0`。
- 131 MiB 完整最终 tar 上传被 HF 明确以“private repository storage limit reached”拒绝；这不是 token 或网络错误。
- 未经授权没有把数据改为公开。完整 tar 本地路径和 SHA 见 `protocol/final_hf_results_backup_20260814.json`。
