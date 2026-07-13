# PathVLM-R1 大修执行日志

更新时间：2026-07-05

用途：记录大修过程中已经执行的检查、得到的结果、下一步行动和待确认事项。建议后续所有实验、下载、脚本改动和表格数字都在此登记，避免本地/远程、代码/数据版本再次混乱。

## 建议的结果管理方式

### 目录建议

建议在 `/home/wjy/Majorrevision` 下维护轻量文档，在 `/home/wjy/revision_runs` 下维护实验产物：

```text
/home/wjy/Majorrevision/
  code_version_audit.md
  revision_todo_priorities.md
  revision_execution_log.md
  response_draft.md

/home/wjy/revision_runs/
  00_manifests/
    final_experiment_manifest.md
    data_split_manifest.md
    model_checkpoint_manifest.md
  01_data_checks/
    split_leakage_check_YYYYMMDD.tsv
    duplicate_image_examples.json
  02_statistics/
    confidence_intervals.csv
    mcnemar_tests.csv
  03_gpt4o_expert_validation/
    sampled_cases.json
    expert_scores_template.xlsx
    agreement_results.csv
  04_baselines/
    model_name/
      run_config.yaml
      outputs.json
      metrics.csv
  05_figures_tables/
    revised_table_II.csv
    revised_table_IV.csv
    revised_fig4/
    revised_fig7/
```

### 每个实验必须登记的信息

每次运行都建议记录：

- 任务目的
- 本机或远程服务器
- hostname、工作目录
- git commit 或文件 SHA256
- 数据文件路径与 SHA256
- 模型 checkpoint 路径
- 运行命令
- 输出文件路径
- 样本数
- 关键指标
- 是否可复现
- 是否进入论文表格

当前已有的关键文档：

- `/home/wjy/Majorrevision/code_version_audit.md`
- `/home/wjy/Majorrevision/revision_todo_priorities.md`

---

## 1. 已执行：数据划分泄漏初查

执行时间：2026-07-05

执行位置：本机 `lab`，目录 `/home/wjy`。

### 检查对象

候选数据文件：

- `PathMMU/sft_with_cot_data_sample3000.json`
- `PathMMU/with_cot_data_sample3000.json`
- `processed_data/picked.json`
- `PathMMU/without_cot_2000_val.json`
- `PathMMU/without_cot_data.json`

### 字段结构发现

`PathMMU/sft_with_cot_data_sample3000.json` 是 LLaMA-Factory 消息格式：

- 字段：`messages`, `images`
- 图片路径在 `images[0]`
- 问题文本在 `messages` 中，需要解析 `<image>` 后的内容

`PathMMU/with_cot_data_sample3000.json` 是原始 PathMMU 格式：

- 字段：`image`, `problem`, `solution`

`PathMMU/without_cot_2000_val.json` 是验证/测试候选格式：

- 字段：`image`, `problem`, `solution`

`processed_data/picked.json` 是另一组 1000 条数据：

- 字段：`image`, `problem`, `solution`
- 当前看起来更像另一个 VQA/processed 数据源，不属于 PathMMU 5385 子集同一图片集合

### 增强版泄漏检查结果

| 数据文件别名 | 路径 | 样本数 | unique images | 重复 image 行数 | unique questions | 重复 question 行数 | SHA256 前缀 |
|---|---|---:|---:|---:|---:|---:|---|
| sft_messages_3000 | `PathMMU/sft_with_cot_data_sample3000.json` | 3000 | 2473 | 527 | 3000 | 0 | b6a8acc35187 |
| sft_raw_with_cot_3000 | `PathMMU/with_cot_data_sample3000.json` | 3000 | 2473 | 527 | 3000 | 0 | 38c9e3652840 |
| rl_picked_1000 | `processed_data/picked.json` | 1000 | 1000 | 0 | 752 | 248 | 5541de7c8370 |
| test_without_cot_1385 | `PathMMU/without_cot_2000_val.json` | 1385 | 1268 | 117 | 1385 | 0 | b229f4c0530b |
| all_without_cot_5385 | `PathMMU/without_cot_data.json` | 5385 | 3809 | 1576 | 5385 | 0 | 00e808836f4e |

### 跨文件重叠结果

| A | B | image overlap | question overlap | QA overlap | 解释 |
|---|---|---:|---:|---:|---|
| sft_messages_3000 | sft_raw_with_cot_3000 | 2473 | 3000 | 3000 | 同一批 3000 的两种格式 |
| sft_messages_3000 | rl_picked_1000 | 0 | 0 | 0 | 未发现重叠 |
| sft_messages_3000 | test_without_cot_1385 | 437 | 0 | 0 | **存在图像级重叠，但问题不同** |
| sft_messages_3000 | all_without_cot_5385 | 2473 | 3000 | 0 | 3000 是 5385 中的子集/改写格式 |
| sft_raw_with_cot_3000 | test_without_cot_1385 | 437 | 0 | 0 | **存在图像级重叠，但问题不同** |
| rl_picked_1000 | test_without_cot_1385 | 0 | 0 | 0 | 未发现重叠 |
| test_without_cot_1385 | all_without_cot_5385 | 1268 | 1385 | 0 | test 是 5385 中的子集/改写格式 |

### 关键结论

1. 如果论文中的 3000 SFT 数据确实对应 `PathMMU/with_cot_data_sample3000.json` 或 `sft_with_cot_data_sample3000.json`，并且 1385 测试集对应 `PathMMU/without_cot_2000_val.json`，则当前划分存在 **437 张图像级重叠**。
2. 这些重叠不是同一 question 重复，而是同一图像对应不同问题。审稿人 2 明确指出这种情况可能导致泛化高估。
3. 因此必须在修稿中处理：要么证明论文实际使用的 split 不是这两个文件；要么承认当前是 Q&A-level split，并补充 image-level disjoint split 评测；要么重新构造 image-level split。

### 抽样重叠案例

同一图像 `e54bc2b270e3007708aad881114522b273e140ec90d267dbfac8bc8990d547d7.jpg`：

- SFT 问题：What is the state of the collagen in the dermis as observed in this image?
- Test 问题：What histological feature is evident in the epidermis of the skin section?

同一图像 `17bd32a6219faa09f1739cb76a5348e7ba2608170cf2d341949411d9fadb8d1c.jpg`：

- SFT 问题：Based on the histological features observed, what type of cells predominantly compose the tissue in the image?
- Test 问题：What does the presence of cells with darker-staining nuclei interspersed within the tissue likely represent?

### 下一步

- 确认论文实际使用的 SFT/RL/Test canonical 文件。
- 若上述文件确为论文使用版本，需要生成 image-level disjoint split。
- 建议将泄漏检查脚本保存为正式脚本，并把结果输出到 `/home/wjy/revision_runs/01_data_checks/`。

状态：**已发现高优先级问题，需继续处理。**

---

## 2. 已执行：术语、拼写和强表述初查

执行时间：2026-07-05

检查来源：论文 PDF 转换文本 `/tmp/pathvlm_revision_review/paper.txt`。

### 已定位的问题类型

1. **模型命名混用**
   - 出现 `PathVLM-α`、`PathVLM-β`、`PathVLM-γ`、`PathVLM-δ`、`PathVLM-ϵ`
   - 同时出现 `PathVLM-Alpha`、`PathVLM-Gamma`、`PathVLM-Epsilon`
   - 建议统一为希腊字母形式，首次出现时括号说明英文名。

2. **Process Reward 术语混用**
   - 出现 `cross-modal process reward`
   - 出现 `Process Reward`
   - 出现 `cross-modal procedural rewards`
   - Fig.3 使用 `cross-modal procedural loss`
   - Fig.5 说明中有 `cross-modal procedural losses`
   - 建议统一为 `cross-modal process reward`。除非数学上确实定义 loss，否则不要使用 `procedural loss`。

3. **拼写错误**
   - Table II 中出现 `Doubao-1.5-vison`
   - 应修改为 `Doubao-1.5-vision`

4. **需要降调的强表述**
   - `strong cross-modal transferability`
   - `remarkable cross-modal transferability`
   - `foundational model for clinical applications`
   - `clear evidence of interpretability`
   - `trustworthiness`
   - `generalization capabilities`

### 建议修改方向

- 将 `remarkable cross-modal transferability` 改为：
  - `promising transfer on dermoscopy but limited generalization to modalities with distinct imaging principles`
- 将 `foundational model for clinical applications` 改为：
  - `a pathology VQA model with potential clinical decision-support value`
- 将 `clear evidence of interpretability` 改为：
  - `a qualitative indication of visual-textual alignment`
- 将 `procedural loss` 改为：
  - `process reward`

### 下一步

- 需要对 LaTeX 源文件或论文 Word/Overleaf 源进行全局替换。
- 修改后应重新 grep 检查所有混用术语是否清除。

状态：**已完成定位，待进入论文源文件修改。**

---

## 3. 已执行：PathMMU 本地子集清单

执行时间：2026-07-05

### 已有 PathMMU JSON 子集

| 文件 | 样本数/结构 | 备注 |
|---|---:|---|
| `PathMMU/data.json` | dict, 5 keys | 原始大表，含多个子集 |
| `PathMMU/converted_data.json` | 5385 | 转换后数据 |
| `PathMMU/without_cot_data.json` | 5385 | 无 CoT 全量候选 |
| `PathMMU/with_cot_data_sample3000.json` | 3000 | SFT 原始格式候选 |
| `PathMMU/sft_with_cot_data_sample3000.json` | 3000 | SFT 消息格式候选 |
| `PathMMU/with_cot_data_3000remaining.json` | 2385 | 3000 后剩余 |
| `PathMMU/with_cot_data_remaining.json` | 4385 | 1000 后剩余 |
| `PathMMU/without_cot_1000_train.json` | 1000 | 训练候选 |
| `PathMMU/withcot_1000_train.json` | 1000 | 训练候选 |
| `PathMMU/without_cot_2000_val.json` | 1385 | 测试/验证候选 |
| `PathMMU/withcot_2000_val.json` | 1385 | 带 CoT 验证候选 |
| `PathMMU/noanswer_without_cot_2000_val.json` | 1385 | 无答案版本候选 |
| `PathMMU/without_cot_2000_val_medr1.json` | 1385 | MedR1 格式候选 |
| `PathMMU/socialpath_mapping.json` | val 155, test 1620, test_tiny 235 | SocialPath mapping |

### 结论

本地已经有不少 PathMMU 子集，不应马上重复下载。第一步应先确认论文中每个数字使用哪个文件：

- 3000 SFT：很可能是 `with_cot_data_sample3000.json` / `sft_with_cot_data_sample3000.json`
- 1000 RL：不确定；`processed_data/picked.json` 与 PathMMU 图像不重叠，可能不是论文 PathMMU RL 数据
- 1385 test：很可能是 `without_cot_2000_val.json`

### 下一步

- 查训练脚本、运行日志和输出 JSON，确认 1000 RL 训练实际用的是哪个文件。
- 若需要补充更多 PathMMU 子集，优先从官方 PathMMU 获取 official split，而不是重复下载已有图片。

状态：**已有数据较多，暂不建议盲目下载；先确认 canonical split。**

---

## 4. 已执行：现有下载脚本与基线资产初查

执行时间：2026-07-05

### 当前发现

初步搜索到的直接相关文件较少：

- `./download.py`
- `./LLaMA-Factory/data/dataset_info.json`

工作区中已经存在：

- `LLaMA-Factory/`
- `LLaVA/`
- `HuatuoGPT-Vision/`
- 多个模型的输出 JSON 和 scored JSON
- `papers/VLM-R1-main/`

### 结论

不建议马上继续 `git clone` 大型仓库到 `/home/wjy` 根目录。当前根目录已经很乱，`LLaMA-Factory` 约 439G，`papers` 约 45G。新增基线应放到统一目录，例如：

```text
/home/wjy/revision_runs/04_baselines/repos/
/home/wjy/revision_runs/04_baselines/configs/
/home/wjy/revision_runs/04_baselines/outputs/
```

### 建议优先基线

按可行性排序：

1. **已有输出的模型重新整理**
   - Qwen2.5-VL-7B
   - Qwen2.5-VL-3B
   - Lingshu-7B
   - InternVL3-8B
   - MedGemma-4B
   - MedVLM-R1
   - HuatuoGPT-Vision-7B

2. **病理专用相关工作优先写作定位**
   - UNI
   - CONCH
   - PathChat

3. **若能获取权重/环境，再尝试运行病理 VLM 基线**
   - 先确认模型许可、显存需求、输入格式、是否支持 VQA。
   - 不建议在未确认前直接下载大模型。

### 下一步

- 建立 `baseline_candidate_table.md`，列出每个候选基线：是否已有代码、是否已有权重、是否支持 VQA、预计显存、是否能直接跑 PathMMU。
- 优先整理已有模型输出，减少新增下载。

状态：**已完成初查；建议先建基线清单再拉代码。**

---

## 5. 其他模态数据集下载计划

对应待办：修正跨模态能力相关结论，需要补充/核验其他模态数据集。

### 论文中已有域外数据集

- ChestCT
- ISIC 2020
- Retinal OCT-C8
- Diabetic Retinopathy
- 来源描述为 OmniMedVQA 或天池等公开数据

### 当前策略

1. 先检查本地是否已有 OmniMedVQA 或四个域外数据集缓存。
2. 若没有，优先下载 metadata 和小样本验证集，不直接下载全量大图。
3. 每个数据集单独记录：来源 URL、下载命令、样本数、标签格式、许可证、论文表 IV 对应样本数。
4. 下载目标目录建议：

```text
/home/wjy/revision_runs/datasets_out_of_domain/
  chest_ct/
  isic2020/
  retinal_oct_c8/
  diabetic_retinopathy/
```

5. 下载完成后应生成 manifest：

```text
/home/wjy/revision_runs/00_manifests/out_of_domain_datasets_manifest.md
```

### 暂不立即下载的原因

- 当前尚未确认已有缓存位置。
- 数据集可能较大，直接下载会加剧版本混乱。
- 审稿人的核心要求不一定是更多域外数据，而是修正过强结论、解释下降、补充分模态统计。

状态：**已形成下载计划；下一步先查缓存和数据许可。**

---

## 6. 立刻可执行的下一批任务

### 6.1 生成正式数据泄漏检查脚本

建议新建：

```text
/home/wjy/Majorrevision/scripts/check_split_leakage.py
```

输出：

```text
/home/wjy/revision_runs/01_data_checks/split_leakage_summary.tsv
/home/wjy/revision_runs/01_data_checks/split_leakage_examples.json
```

优先级：必做。

### 6.2 确认 1000 RL 数据实际文件

需要检查：

- 训练脚本 `--dataset_name`
- 训练日志
- 远程 `/root/VLM-R1` 中 `run_grpo_rec.sh` 和 data_config
- 本机 `papers/VLM-R1-main/src/open-r1-multimodal/data_config/*.yaml`

优先级：必做。

### 6.3 建立 baseline candidate table

建议新建：

```text
/home/wjy/Majorrevision/baseline_candidate_table.md
```

列：模型、类型、是否病理专用、是否支持 VQA、代码地址、权重地址、显存需求、当前状态、能否纳入论文。

优先级：强烈建议。

### 6.4 论文术语替换清单

建议新建：

```text
/home/wjy/Majorrevision/terminology_fix_list.md
```

内容：旧术语、新术语、出现位置、修改理由。

优先级：必做。

---

## 7. 当前判断

当前最紧急、最可立即推进的不是下载更多数据或拉更多仓库，而是：

1. 固定 canonical 数据 split。
2. 解决 SFT/Test 之间 437 张图像重叠的问题。
3. 明确论文中的 1000 RL 数据到底对应哪个文件。
4. 建立实验 manifest，保证每个论文表格数字可追溯。
5. 在此基础上再决定是否下载更多 PathMMU 子集、域外数据集或新增病理基线。

如果不先完成这些管理动作，后续新增实验可能会继续堆出更多版本分歧。

---

## 8. 已执行：基线代码拉取与环境文档初版

执行时间：2026-07-05

执行位置：本机 `lab`，目录 `/home/wjy`。

### 已创建目录

```text
/home/wjy/revision_runs/04_baselines/repos
/home/wjy/revision_runs/04_baselines/configs
/home/wjy/revision_runs/04_baselines/outputs
/home/wjy/Majorrevision/env_doc
```

### 已拉取代码

| 名称 | 本地目录 | commit | 大小 | 状态 |
|---|---|---|---:|---|
| CONCH | `/home/wjy/revision_runs/04_baselines/repos/CONCH` | `141cc09c7d4ff33d8eda562bd75169b457f71a62` | 3.8M | 已克隆 |
| UNI | `/home/wjy/revision_runs/04_baselines/repos/UNI` | `42715efc11722a496e0a67f3369505a8f277206c` | 12M | 已克隆 |
| PLIP | `/home/wjy/revision_runs/04_baselines/repos/PLIP` | `f010f3d0bef20f4e8cc64cc26c301cbd26305fa1` | 1.5M | 已克隆 |
| LLaVA-Med | `/home/wjy/revision_runs/04_baselines/repos/LLaVA-Med` | `30697ca50b5c29a8e955c99330b259776aef27b9` | 1.6G | 已克隆，仓库较大 |

PathChat：尝试访问 `https://github.com/mahmoodlab/PathChat.git`，结果为 `Repository not found`。需要后续从论文、官网、HuggingFace 或作者主页确认是否有公开代码/权重。

### 已写文档

```text
/home/wjy/Majorrevision/env_doc/baseline_repos_and_envs.md
/home/wjy/Majorrevision/env_doc/reproducible_env_commands.md
```

### 关键判断

1. CONCH、UNI、PLIP 更适合作为 pathology foundation / CLIP-like baseline 或 related work 定位，不一定能直接做生成式多选 VQA。
2. LLaVA-Med 最接近可直接作为医学 VQA baseline，但环境较重，建议单独 conda 环境运行。
3. 暂未下载任何模型权重；后续执行前需确认 HuggingFace 权限、磁盘空间和显存。
4. 基线结果必须在 canonical split 明确后再纳入论文主表。

状态：**基线代码和环境文档初版完成，尚未安装环境或运行模型。**
