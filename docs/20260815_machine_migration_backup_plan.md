# 机器迁移备份计划（2026-08-15）

## 目标

在不停止当前 Codex 会话、不删除或移动本机文件、不上传凭据的前提下，建立可验证、
可增量更新的迁移备份。恢复目标是新机器能够重建代码、数据合同、环境、评测原始输出、
训练记录和关键模型，并能够继续当前研究计划。

## 当前容量与边界

- 工作区文件系统：9.8 TiB，总使用 6.3 TiB，可用约 3.6 TiB。
- `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100`：约 2.0 TiB。
- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100`：约 230 GiB。
- `/home/dataset-assist-0/czy/wjy/cache`：约 6.2 GiB。
- Git 仓库：约 102 MiB。
- 私有 HF 存储此前拒绝新增 131 MiB 完整评测归档；认证和网络本身正常。

## 分层备份

### A. 立即备份：代码、协议与决策记录

- 将 `myr1` 当前分支提交并推送 GitHub。
- 包括 docs、protocol、runner、测试、结果索引和迁移指南。
- 不包括密钥、认证文件或大模型权重。

### B. 立即备份：Codex 在线状态

- 使用 SQLite online backup API 复制当前 Codex 状态，不要求退出 Codex。
- 包括 sessions、history、state/goals/logs/memories SQLite、rules、skills、shell snapshots。
- 排除 `auth.json`、缓存、临时文件、安装包和插件缓存。
- 当前基线快照：
  `/home/dataset-assist-0/czy/wjy/backup_archives/codex_online_snapshot_20260815T000000Z`
  （目录名按本地日期命名，精确 UTC 时间以内部 `manifest.json` 为准）。
- 该快照只允许进入 private HF 仓库或受控点对点传输，绝不进入 public/gated 归档。

### C. 立即上传：稳定评测与实验记录

- 新建 public + manual-gated dataset 仓库。
- 先上传已经闭合且本地 SHA-256 验证的完整评测归档、manifest 和 verification。
- 包括原始 predictions、rollout JSON/JSONL、metrics、日志、协议和结果表；不包括凭据。
- 当前仍在追加的 LLaVA-Med PathMMU/PathVQA/OmniMedVQA 不进入第一版归档；完成并
  验证后作为 immutable 增量前缀上传。

### D. 模型快照：model-only 两级优先队列

- public + manual-gated，每个关键模型独立仓库或使用不可变版本前缀，上传前验证 gate 状态。
- **不备份任何 optimizer/DeepSpeed ZeRO 分片、scheduler 或 RNG 训练状态。** 后续训练从可加载
  model-only 快照重新建立 optimizer；因此不能逐比特恢复旧 Adam 动量，但不影响推理、评测、
  继续 SFT/RL 或接入新的 process reward。
- 第一优先级为关键可加载模型快照：最终/候选 Stage3、n=4/n=8 rule-RL、选定 SFT/RL
  parent、正式数据消融关键端点，以及论文主表实际使用的模型。
- 第二优先级为其余全部可加载模型快照，包括中间 checkpoint、小型消融、冒烟和失败尝试中
  实际保存下来的模型版本。
- “全部模型快照”按实验节点和权重内容定义，而不是按重复路径定义。同一个模型若同时存在于
  `output/checkpoint-*`、`model_snapshots/checkpoint-*` 和 output 根目录，只上传一份权重，
  manifest 保留所有原始路径到内容哈希的映射。这样是无损去重，不删除任何模型版本。
- 每套快照必须包含加载所需的 config、processor/tokenizer、权重索引及 adapter 配置；上传后
  进行 fresh-download、SHA-256 和最小 load smoke 验证。

### E. 数据集、基础模型、环境与缓存

- 数据集本体或可验证的固定 revision/下载脚本进入 gated dataset 归档。
- 对第三方基础模型逐一检查许可证；允许再分发的可上传 gated，不允许的仅保存 repo、
  revision、文件 SHA-256 和下载命令。
- 环境采用双层备份：一层保存完整虚拟环境归档用于同类机器快速恢复，另一层保存
  lock/requirements、CUDA/PyTorch/驱动审计、安装脚本和 smoke tests，用于路径或系统版本
  变化时可靠重建。完整环境中排除密钥和认证材料。
- HF/ModelScope cache 默认保存索引与重建方法；只有无法稳定重下载且允许分发的 blob 才归档。

## 一致性与增量策略

1. 运行中的 JSONL 在每次归档前记录字节数和 SHA-256；只有完成门禁通过后标记 final。
2. 每个远端前缀不可变；后续变化写入新的时间戳增量前缀。
3. 每个归档都包含文件清单、总字节数、逐文件哈希和 fresh-download 抽检。
4. 不通过删除本地副本来证明远端成功；必须先完成远端可见性/gate/哈希验证。
5. 任何本地清理、私有仓库迁移或旧 HF 权重删除均需再次取得用户逐项确认。

## 恢复顺序

1. 克隆 GitHub 固定 commit。
2. 恢复 private Codex 快照到新的隔离 `CODEX_HOME`，重新人工认证，不复制旧 token。
3. 运行环境安装脚本和硬件 preflight。
4. 下载 gated 数据/模型并核对 snapshot manifest。
5. 恢复数据路径映射，运行小型 smoke。
6. 读取 `docs/LATEST.md`、统一结果表和最新迁移状态，再决定续训或继续评测。
