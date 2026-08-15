# 最终实验记录备份审计与新机器恢复入口

> 状态：本地内容级校验与 HF 固定 revision 远端校验均已通过。本文描述的是实验记录层，
> 不重复承载已经独立备份的模型权重。

## 备份结论

本次最终记录快照覆盖 49,245 个文件，归档前总大小 10,226,535,291 字节，压缩归档大小
3,737,087,357 字节。归档 SHA-256：

```text
0c00a19b827ec8d3ec0c539ca6741507355397569d4e851e5eb60c66ae6cb625
```

独立校验逐个读取归档成员并复算 SHA，结果为：

```text
verified_payload_files: 49245
verified_payload_bytes: 10226535291
missing: 0
unexpected: 0
```

## 具体保存了什么

| 类别 | 文件数 | 归档前字节数 | 主要内容 |
|---|---:|---:|---|
| 正式评测运行 | 3,869 | 6,334,193,353 | predictions/metrics/run config/log、Judge 与可解释性正式产物 |
| 评测报告 | 157 | 151,080,278 | bad-case、费用、统计和阶段报告 |
| 人工复核材料 | 1,079 | 91,653,788 | 奖励复核、生成质量盲评、ROI/可解释性复核材料与工具 |
| 主线/消融训练记录 | 43,965 | 3,565,534,648 | rollout、reward event、Judge 响应、费用、trainer config/state JSON 与日志 |
| 训练报告 | 58 | 23,020,442 | 配置审计、训练结论和运行报告 |
| 训练数据合同 | 69 | 38,828,754 | split、数据配比、冻结训练输入与 manifest |
| 评测数据合同 | 36 | 22,204,038 | 冻结 panel、评测输入合同和 manifest |
| 生成训练配置 | 12 | 19,990 | 实际生成的训练配置 |

归档中出现文件只表示它被保存用于追溯，不自动表示它可进入论文。论文有效口径仍由
`protocol/final_revision_assets_audit_20260815.json` 与
`docs/result_catalog_20260815/result_lineage.json` 决定。

## 明确没有保存什么

- smoke、吞吐门禁和临时冒烟产物；
- 模型权重与 adapter（由独立 HF 模型仓库承担）；
- optimizer、scheduler、RNG 和 DeepSpeed/ZeRO 状态；
- 可由上游固定 revision 重建的第三方数据、基础模型和下载缓存；
- API key、HF token、Codex auth 和 `.secrets/`；
- 浏览器功能测试生成的伪专家提交。

两条历史 Grok Judge 缓存中包含 Cloudflare `set-cookie` 的 token-shaped 字符串。源文件未改动，
归档副本仅对这些字符串脱敏；请求哈希、模型响应、reward events、评分、usage 和费用保留。

## HF 固定位置

```text
repo: Freddie1946/PathVLM-R1-Migration-Archive-20260815
repo type: dataset
visibility: public + manual gated
revision: 5d2485312266ec0670a14494b1fe999d10fe94e1
prefix: increments/20260815_final_experiment_records_v1
```

该前缀包括：归档本体、逐文件 manifest、备份审计、外层 verification 和独立逐成员 verification。

## 新机器一键拉取与校验

新机器先克隆 Git，然后执行：

```bash
git clone --branch codex/a100-stage3-eval --single-branch \
  https://github.com/Freddie1946/myr1.git "$WJY_WORK_ROOT/myr1"
cd "$WJY_WORK_ROOT/myr1"
hf auth login
bash scripts/restore_final_experiment_records_increment_20260815.sh \
  "$WJY_WORK_ROOT/restored_results/final_experiment_records_20260815" --extract
```

脚本固定 HF revision 和归档 SHA，支持 Hugging Face 下载续传；校验失败时不会解压。若只需下载并校验，
省略 `--extract`。解压始终落到独立目录，脚本拒绝覆盖非空目标。

机器可读的远端上传审计位于：

```text
protocol/final_experiment_records_hf_backup_20260815.json
```
