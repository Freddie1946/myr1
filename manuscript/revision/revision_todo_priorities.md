# PathVLM-R1 大修待办清单与优先级

更新时间：2026-07-05

依据文件：
- 论文：`/home/wjy/Majorrevision/PathVLM_R1__A_Reinforcement_Learning_Driven_Reasoning_Model_for_Pathology_Visual_Language_Tasks.pdf`
- 大修意见：`/home/wjy/Majorrevision/转发：JBHI-06328-2025.pdf`
- 代码版本审计：`/home/wjy/Majorrevision/code_version_audit.md`

当前现实约束：
- 本机算力：4 张 RTX 4090，每张 24GB；当前显存已有占用，不适合大规模 7B 全量重复训练，但可以做推理、离线评分、小规模 LoRA/QLoRA、抽样消融和统计分析。
- 远程服务器：已确认存在 `/root/VLM-R1`、`/root/data` 等目录，可作为代码和数据版本参考；需要进一步确认细节时再连接远程。
- 代码现状：当前可见训练代码完整支持 `Accuracy Reward + Format Reward`；论文 Stage 3 所述 GPT-4o Process Reward 训练接入代码尚未在已检查路径中完整发现。
- 数据现状：本机有 PathMMU、processed_data、train_data、valid_data；`processed_data/picked.json` 为 1000 条，`PathMMU/with_cot_data_sample3000.json` 和 `sft_with_cot_data_sample3000.json` 为 3000 条，`PathMMU/without_cot_2000_val.json` 为 1385 条。

## 优先级定义

- **必做**：不完成会严重影响再次送审，或直接对应多位审稿人的核心质疑。
- **强烈建议**：能显著增强说服力，但可根据算力/时间缩小规模。
- **可选增强**：锦上添花，主要用于提高论文完整度。
- **写作整理**：低算力成本，必须系统完成，但一般不需要新增大实验。

---

## 1. 补充 GPT-4o 奖励与病理专家人工评分的一致性验证

提出者：审稿人 1 第 1、5 点；审稿人 2 第 4 点；审稿人 3 第 7 点。

论文现状：论文只用 Fig.4 证明 GPT-4o 三次评分一致，Spearman 相关系数大于 0.78。但审稿人指出，一致性不等于临床准确性；同时 GPT-4o 既参与奖励构建又参与推理质量评测，存在循环评价问题。

执行措施：
1. 从已有 500 条推理链评分样本中抽取 80-150 条，覆盖正确/错误、不同模型、不同难度和不同分数区间。
2. 请 2-3 名病理专业人员按论文中的两个核心维度打分：reasoning integrity 和 medical knowledge rationality。
3. 计算 GPT-4o 与专家评分之间的一致性：Spearman 相关、Pearson 相关、ICC 或加权 Kappa。
4. 报告专家间一致性，说明专家评分标准。
5. 在论文中新增一张表或补充材料，说明 GPT-4o 评分不是单纯自洽，而是与人工医学判断有可接受一致性。
6. 如果专家资源不足，最低限度也要做小规模双专家验证，并在 limitation 中承认规模有限。

现实约束：主要耗费专家时间和 GPT-4o 评分整理，不需要大量 GPU。当前已有 `evaluate_with_gpt4o_v2.py`、`scorev3.py`、`score4.py` 和若干 scored JSON，可直接抽样。

预期产出：
- 新增表：GPT-4o vs pathologist agreement。
- 新增方法说明：专家评分标准、样本抽样规则、一致性统计。
- rebuttal 中明确回应循环评价问题。

优先级：**必做**

---

## 2. 明确并验证数据划分规则，排查图像级/病例级泄漏

提出者：审稿人 2 第 2 点。

论文现状：论文说明使用 PathMMU 中 PubMed 与 EduContent 共 5385 条，划分为 3000 SFT、1000 RL、1385 testing。但未说明按 Q&A、图像、病例还是患者维度划分。PathMMU 可能一图多问，如果同一图像跨训练/测试出现，会高估泛化。

执行措施：
1. 对 3000 SFT、1000 RL、1385 test 文件做 image 字段去重和交集统计。
2. 如果数据中有病例/患者 ID 或来源字段，补充病例级去重统计；如果没有，明确说明数据集无患者 ID，只能执行图像级去重。
3. 生成一张数据划分表：样本数、图像数、重复图像数、跨 split 图像交集。
4. 如果发现泄漏，重新构建 image-level disjoint split，并至少重新跑核心模型或关键评测；若时间不允许，需在论文中修正结论并说明影响。
5. 将划分脚本和最终 split 文件固定下来，避免后续版本混乱。

现实约束：主要是数据处理，不需要 GPU。当前本机有 `PathMMU/with_cot_data_sample3000.json`、`PathMMU/without_cot_data.json`、`PathMMU/without_cot_2000_val.json` 等候选文件，但 canonical split 仍需确认。

预期产出：
- 新增表：SFT/RL/Test split de-duplication statistics。
- 新增方法描述：split by image-level 或说明无法做到 patient-level 的原因。
- 如有必要，新增附录：数据划分脚本和 hash。

优先级：**必做**

---

## 3. 修正跨模态迁移能力相关结论，并补充分模态分析

提出者：审稿人 1 第 3 点；审稿人 3 第 1 点。

论文现状：论文声称模型具备 remarkable cross-modal transferability，但 Table IV 显示 PathVLM-R1 只在 ISIC 2020 上显著优于 Qwen2.5-VL-7B；在 Chest CT、Retinal OCT-C8、Diabetic Retinopathy 上均低于 Qwen2.5-VL-7B。

关键数值：
- Chest CT：Qwen2.5-VL-7B 48.60，PathVLM-R1 44.00。
- ISIC 2020：Qwen2.5-VL-7B 56.59，PathVLM-R1 80.00。
- Retinal OCT-C8：Qwen2.5-VL-7B 57.19，PathVLM-R1 49.20。
- Diabetic Retinopathy：Qwen2.5-VL-7B 70.80，PathVLM-R1 59.80。

执行措施：
1. 大幅弱化“广泛跨模态迁移能力”表述，改为“在皮肤镜/病理相关视觉模式上显示潜在迁移，但对 CT/OCT/眼底等差异更大的模态存在下降”。
2. 对四个域外数据集做错误分析：图像模态差异、任务类型、选项文本、训练域距离。
3. 补充每个域外数据集的置信区间，避免单点结果支撑过强结论。
4. 如果算力允许，补充一个轻量的域外 prompt sensitivity 或 few-shot prompt 对比，证明下降不是纯提示词问题。
5. 若不能新增大规模实验，应在 Discussion 和 Limitation 中主动解释性能下降。

现实约束：域外重新推理可能需要 GPU，但可先基于已有结果做统计和文字修正。若要补充实验，建议优先做推理而非训练。

预期产出：
- 修改 Abstract、Conclusion、Discussion 中 cross-modal 相关表述。
- 新增分模态分析段落。
- 新增或更新 Table IV，加入 CI 或样本数。

优先级：**必做**

---

## 4. 补充统计显著性、置信区间和重复实验

提出者：审稿人 2 第 5 点；审稿人 3 第 6 点。

论文现状：当前主要报告单点 accuracy 和 GPT-4o 评分，没有置信区间、显著性检验或多随机种子重复实验。测试样本规模有限，且结果受 prompt 和解析规则影响。

执行措施：
1. 对所有主要 accuracy 报告 Wilson 95% confidence interval 或 bootstrap CI。
2. 对 PathVLM-R1 vs Qwen2.5-VL-7B、PathVLM-R1 vs PathVLM-β 等关键比较做 McNemar test 或 bootstrap paired test。
3. 如果已有不同输出文件或多次运行结果，整理成 repeated-run mean ± std。
4. 若没有时间重复训练，至少重复推理/解析 3 次或做 bootstrap，并在 rebuttal 中说明训练重复的算力限制。
5. 对 GPT-4o reasoning scores 报告均值、标准差和置信区间。

现实约束：统计分析主要 CPU 即可。重复训练 7B 成本高；重复推理和 bootstrap 更现实。

预期产出：
- 更新 Table II、Table IV、Table V，加入 CI 或显著性标记。
- 新增统计方法段落。
- rebuttal 中回应 point estimates 问题。

优先级：**必做**

---

## 5. 补充或修正 Stage 3 Process Reward 训练实现说明

提出者：审稿人 2 第 4 点；审稿人 3 第 3、7 点。

论文现状：论文明确描述 Stage 3 使用 GPT-4o Process Reward 在线或动态生成 cross-modal rewards。但代码审计发现，当前本机和远程已确认的主训练脚本 `grpo_rec.py` 只完整支持 `accuracy_reward` 和 `format_reward`，尚未发现完整接入 GPT-4o Process Reward 的 GRPO 训练代码。

执行措施：
1. 继续在本机和远程搜索 Stage 3 训练脚本，关键词包括 `process_reward`、`integrity_score`、`knowledge_score`、`gpt4o_reward`、`dual_reward`、`OpenAI + GRPOTrainer`。
2. 如果找到缺失脚本，固定版本，记录 commit/hash、训练命令、环境、模型路径和数据路径。
3. 如果找不到，需要在论文和 rebuttal 中谨慎处理：要么补全可复现实装代码，要么调整论文描述，避免声称不存在的训练实现。
4. 明确 GPT-4o reward 是在线计算、离线缓存，还是先 rollout 再评分。
5. 报告 GPT-4o 模型版本、temperature、max tokens、prompt、解析规则、失败兜底、调用次数和成本估计。

现实约束：如果真正在线接入 GPT-4o 训练，成本和速度很高，不适合直接大规模重跑。更现实的方案是确认历史实现或做小规模复现实验；必要时使用离线缓存 reward。

预期产出：
- 新增训练实现说明表。
- 新增 reward prompt 和解析规则附录。
- 修正代码/论文版本不一致风险。

优先级：**必做**

---

## 6. 增加失败案例分析，尤其是罕见病或困难病理场景

提出者：审稿人 1 第 2 点。

论文现状：论文主要展示成功案例 Fig.1 和 Fig.7，缺少典型失败案例。审稿人要求说明模型缺陷和未来改进方向。

执行措施：
1. 从 PathVLM-R1 错误预测样本中抽取 6-10 个案例。
2. 按错误类型归类：视觉特征识别错误、选项概念混淆、推理链正确但答案错、答案正确但推理错误、罕见病/低频概念、跨模态失败。
3. 每类选 1-2 个代表案例，给出问题、标准答案、模型输出、错误原因。
4. 请病理专家对其中 3-5 个关键失败案例给简短医学解释。
5. 在论文中新增 Failure Case Analysis 小节，或放入补充材料。

现实约束：不需要训练，只需已有结果文件和人工整理。专家参与会提高说服力。

预期产出：
- 新增失败案例表。
- 新增 limitation 中关于 rare disease 和 open-ended diagnosis 的说明。

优先级：**必做**

---

## 7. 增强基线模型，尤其是病理专用模型与非 RL 强基线

提出者：审稿人 2 第 3 点；审稿人 3 第 4 点。

论文现状：Table II 包含多种通用/医学 VLM，但缺少足够的病理专用 VLM/LMM 基线。审稿人明确点名 UNI、CONCH、PathChat 系列工作。

执行措施：
1. 相关工作中系统补充 UNI、CONCH、PathChat，并说明它们与本文任务的差异。
2. 若能跑模型，优先选择可访问且适合 VQA/对话的病理模型，例如 PathChat 类或其他开源病理 VLM。
3. 若 UNI/CONCH 主要是 encoder/representation model，无法直接做多选 VQA，应说明不能直接端到端比较，并可做 linear probe/retrieval 或文献定位比较。
4. 增加一个“非 RL 但领域适配强”的对照：例如同一基座的 SFT-only、继续 SFT、或者 LLaMA-Factory 中已有 SFT checkpoint。
5. 表 II 增加 prompt/config 说明，回应审稿人 1 第 7 点。

现实约束：直接跑大型病理模型可能受权重、显存和接口限制；至少应完成相关工作和定位比较。当前本机已有 LLaMA-Factory 和多个模型输出 JSON，可优先整理已有结果。

预期产出：
- 更新 Related Work。
- 更新 Table II 或新增 pathology-specific baseline table。
- 新增 baseline prompt/config 说明。

优先级：**强烈建议**

---

## 8. 扩大评测范围：完整 held-out set、官方 PathMMU split 或独立病理基准

提出者：审稿人 2 第 5 点。

论文现状：Table II 显示主要对比为 500 samples；论文称 performance testing 为 1385 条，但表格未充分展示完整 held-out set 的全部对比。审稿人要求不要只用 500 样本。

执行措施：
1. 优先在 1385 条 held-out set 上补全关键模型结果：Qwen2.5-VL-7B、PathVLM-β、PathVLM-R1、至少一个强基线。
2. 如果已有完整输出，重新汇总并替换或补充 500-sample 表。
3. 若算力不足，保留 500-sample 作为 reasoning quality 子集，但 accuracy 必须给完整 1385 结果。
4. 如果能获得 PathMMU 官方 split，增加官方 split 评测；否则说明使用子集原因。
5. 尝试增加一个独立病理 benchmark；若无法完成，至少在 Limitation 中承认。

现实约束：1385 条推理在 4x4090 上可行性较高，训练不必重跑。GPT-4o 六维评分全量成本较高，可只对代表性子集评分。

预期产出：
- 新增完整 held-out accuracy 表。
- 保留 500-sample reasoning 评分作为补充，而不是唯一主要证据。

优先级：**强烈建议**

---

## 9. 补充消融实验：证明 Process Reward 的独立贡献

提出者：审稿人 3 第 3 点；审稿人 2 第 4 点。

论文现状：Table V 包含 Alpha/Beta/Gamma/Delta/Epsilon/R1，但审稿人仍认为需要更强消融，尤其要证明 GPT-4o Process Reward 相比 SFT-only、outcome-only、简单规则/模型 reward 有统计显著提升。

执行措施：
1. 固定数据 split 和基座，整理已有模型：SFT-only、Acc/Format GRPO、Acc/Format/Process GRPO。
2. 如果 Stage 3 训练代码缺失，先解决版本问题，否则无法严谨补消融。
3. 加入简单替代 reward 对照：仅答案 reward、仅 format reward、答案+format、答案+format+LLM answer similarity、答案+format+GPT process。
4. 若训练成本过高，做小规模 500/1000 RL sample 的 pilot 消融，并明确是补充证据。
5. 对消融结果做 CI 和显著性检验。

现实约束：完整重训成本较高；当前 4x4090 不适合大规模 7B 全量重复训练，建议 LoRA/QLoRA 或小步数 pilot。远程如有更强算力可补确认。

预期产出：
- 新增 ablation table。
- 明确 Process Reward 的增益是否稳定。

优先级：**强烈建议**

---

## 10. 数据规模消融：验证 3000 SFT + 1000 RL 的鲁棒性

提出者：审稿人 3 第 5 点；审稿人 2 第 1 点。

论文现状：论文强调少量数据即可获得强性能，但没有系统测试不同 SFT/RL 数据规模。

执行措施：
1. 设计轻量数据规模实验：SFT 数据量 500、1000、2000、3000；RL 数据量可选 250、500、1000。
2. 如果完整矩阵太贵，做最小可行矩阵：SFT 500/1000/3000 + RL 500/1000。
3. 每个设置不一定完整训练 18000/2000 steps，可做固定较短 steps 的趋势验证，并明确其目的。
4. 汇报 accuracy 和 reasoning score 随数据规模变化的趋势。
5. 若无法训练，需弱化“仅需 3000/1000 即具备 foundational 能力”的表述。

现实约束：这是最耗算力的新增实验之一。4x4090 下建议 LoRA/QLoRA、减少 seed 或只跑关键点；优先级低于专家验证、数据划分、统计分析。

预期产出：
- 新增 data scaling figure/table。
- 支撑或修正 small-data claim。

优先级：**强烈建议**

---

## 11. 增加可解释性/视觉 grounding 的定量验证

提出者：审稿人 3 第 2 点；审稿人 1 第 8 点。

论文现状：Fig.7 只有一张注意力热力图，且缺少颜色刻度条。审稿人认为单一 attention map 不能证明可信度。

执行措施：
1. 至少补充 6-12 个代表病例的 heatmap，包括成功与失败案例。
2. 图 7 增加 color scale bar，并提升分辨率。
3. 若有区域标注，计算 heatmap 与专家 ROI 的 overlap/pointing game/IoU。
4. 若没有区域标注，请病理专家对 heatmap 是否关注关键区域进行 0-2 或 1-5 分评价。
5. 可补充 deletion/insertion 或遮挡实验：遮挡高注意力区域后准确率下降，遮挡低注意力区域影响较小。

现实约束：生成 heatmap 需要模型 forward 和注意力提取，但样本量可控；专家 ROI 标注成本中等。

预期产出：
- 更新 Fig.7。
- 新增 quantitative grounding/faithfulness 表。

优先级：**强烈建议**

---

## 12. 明确 GPT-4o 奖励细节：在线/离线、prompt、参数、成本、异常处理

提出者：审稿人 2 第 4 点；审稿人 1 第 4 点。

论文现状：论文给出 Fig.3 prompt 思路和 0.4 扣分规则，但没有充分说明具体 GPT 模型版本、解码参数、调用次数、成本、解析失败兜底、在线/离线缓存。

执行措施：
1. 整理 GPT-4o prompt 原文，放入 Appendix。
2. 说明 model version、temperature、max_tokens、base_url/API provider 如可公开、调用批次。
3. 说明 JSON parsing、Markdown fenced JSON、fallback 逻辑。
4. 解释每缺失一步扣 0.4 的依据：可作为 heuristic，并补充敏感性分析，例如 0.3/0.4/0.5 是否改变模型排序。
5. 如果没有训练时在线调用记录，不能含糊其辞，需要明确实际执行方式。

现实约束：主要是整理和少量敏感性分析。现有 `evaluate_with_gpt4o_v2.py` 可作为 prompt 和解析规则来源，但要注意它是离线脚本。

预期产出：
- 新增 Appendix：GPT-4o evaluation prompt and parsing rules。
- 新增 sensitivity analysis 或文字解释 0.4 penalty。

优先级：**必做**

---

## 13. 重新表述创新点，避免被认为只是套用标准 SFT + GRPO 流程

提出者：审稿人 3 第 3 点。

论文现状：审稿人认为 SFT + GRPO + format/accuracy/GPT reward 是近期推理模型常见范式，需要明确本文独有贡献。

执行措施：
1. 将创新点从“使用 GRPO”转移到“病理 VQA 中的过程奖励设计、阶段式知识注入、医学推理质量评价、低样本适配”。
2. 删除或弱化“first/foundational/broadly generalizable”等过强说法。
3. 在 Introduction 和 Discussion 中明确本文边界：PathMMU 多选病理 VQA，而不是通用病理基础模型。
4. 用消融和专家验证支撑 process reward 的真实贡献。

现实约束：写作任务，不需要 GPU，但依赖新增实验支撑。

预期产出：
- 修改 Contribution bullets。
- 修改 Abstract/Conclusion 夸大表述。

优先级：**必做**

---

## 14. 完善相关工作：UNI、CONCH、PathChat 及病理 VLM 定位

提出者：审稿人 3 第 4 点；审稿人 2 第 3 点。

论文现状：相关工作对通用 VLM、医学 VLM、RL 后训练有介绍，但对主流病理 foundation model 和 pathology VLM 讨论不足。

执行措施：
1. 新增一节或扩展 Related Work：Pathology foundation models and pathology VLMs。
2. 重点讨论 UNI、CONCH、PathChat，并补充其他相关病理图文模型。
3. 从任务类型、训练数据、是否支持 VQA/对话、是否支持推理链、是否可解释、是否临床部署等维度对比。
4. 若不能直接实验对比，明确原因。

现实约束：文献整理为主，不需要 GPU。

预期产出：
- 新增相关工作段落。
- 可能新增定位表。

优先级：**写作整理**

---

## 15. 全文术语、图表和排版统一

提出者：审稿人 1 第 6、8、9 点；审稿人 3 第 8 点。

论文现状：存在术语混用和图表可读性问题，例如 `PathVLM-α` vs `PathVLM-Alpha`、`procedural loss` vs `process reward`、`Doubao-1.5-vison` 拼写错误、Fig.4 太小、Fig.7 无色条。

执行措施：
1. 统一模型命名：建议全文使用 `PathVLM-α`、`PathVLM-β`、`PathVLM-R1`，首次出现时括号说明 Alpha/Beta。
2. 统一 reward 术语：建议使用 `process reward`，避免 `procedural loss`，除非确实指 loss。
3. 修正 `Doubao-1.5-vison` 为 `Doubao-1.5-vision`。
4. 重画 Fig.4，放大文字，提高分辨率。
5. Fig.7 加 color scale bar，并补充说明 heatmap 归一化方式。
6. Table II 说明 large-parameter 和 comparable-parameter models 的 prompt、推理设置、样本数、解析规则。

现实约束：低成本，必须完成。

预期产出：
- 修订版 manuscript 全文术语一致。
- 图表质量达送审标准。

优先级：**必做**

---

## 16. 建立最终版本追踪表，避免代码和论文再次错位

提出者：不是审稿人直接要求，但由当前代码审计发现。

论文现状：论文 Stage 3 描述与当前可见训练代码存在潜在不一致；本机和远程代码、脚本路径、模型尺寸 3B/7B 命名混杂。

执行措施：
1. 建立 `final_experiment_manifest.md`，记录每个实验对应的代码文件、commit/hash、数据文件、模型 checkpoint、运行命令、输出 JSON。
2. 对 Table II-IV-V 每个数字建立来源索引。
3. 明确最终论文使用 7B 还是 3B，清理脚本中的 3B/7B 混用说明。
4. 统一本地和远程关键脚本路径，避免远程脚本引用 `/home/wjy/...` 这类本机路径。

现实约束：主要是整理，但非常重要。否则 rebuttal 或复现实验很容易再次混乱。

预期产出：
- 实验 manifest。
- 表格数字可追溯。

优先级：**必做**

---

## 建议执行顺序

第一阶段：低算力但必须立即完成
1. 数据划分与去重检查。
2. 统计显著性和置信区间。
3. GPT-4o reward 细节整理。
4. 跨模态结论降调与分模态分析。
5. 术语、图表、拼写修正。
6. 建立实验 manifest。

第二阶段：中等成本，最能回应核心质疑
1. 病理专家评分一致性验证。
2. 失败案例分析。
3. 完整 1385 held-out set 关键模型评测。
4. 可解释性多案例 heatmap/专家定位评分。

第三阶段：高成本，根据时间和算力选择
1. 病理专用模型基线。
2. Process Reward 消融。
3. 数据规模消融。
4. 官方 PathMMU split 或独立病理 benchmark。

## 最关键风险提示

当前最需要优先澄清的是：论文声称的 Stage 3 `Accuracy + Format + GPT-4o Process Reward` 训练实现，在已检查的本机和远程代码中尚未完整定位。如果最终找不到该代码，必须在补实验或修稿前先解决这个版本一致性问题，否则新增实验和 rebuttal 都会建立在不稳固的代码基础上。
