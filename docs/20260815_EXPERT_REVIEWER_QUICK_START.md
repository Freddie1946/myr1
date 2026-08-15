# PathVLM-R1 专家复核包简明使用指南

## 一、开始前

1. 将收到的 `.tar.gz` 压缩包完整解压到一个可写目录，不要直接在压缩包内打开文件。
2. 向负责人领取固定匿名 `Reviewer ID`，例如 `pathologist_A`。全程使用同一编号，不要填写姓名、邮箱，也不要自行改变拼写。
3. 由负责人指定使用“独立盲评”还是“外部参考辅助评审”。同一位专家不要在两种模式间切换；辅助模式结果必须标记为 `reference-assisted`。

## 二、启动评审网页

Linux/macOS：进入解压后的 `two_pass_expert_review_20260815` 目录，运行：

```bash
bash serve_review.sh 8765
```

Windows PowerShell：进入同一目录，运行：

```powershell
python serve_expert_review_app.py --root . --output .\expert_submissions --host 127.0.0.1 --port 8765
```

然后用浏览器访问：

```text
http://127.0.0.1:8765/
```

本工具只在本机运行，不会自动向网络上传评分。评审结束后可在终端按 `Ctrl+C` 停止。

## 三、完成三项评审

### A. 奖励机制复核（60例）

1. 进入负责人指定的奖励复核入口。
2. 输入固定 `Reviewer ID`，逐题阅读图片、题干、选项和待评分回答。
3. 对六个事件分别点击 `True/False`；外部模型意见只供参考，最终判断以专家意见为准。
4. 每题必须点击“保存本题评分到本地文件”，看到保存成功后再点“下一题”。
5. 中断后重新启动网页，输入相同 ID，点击“载入本题已保存评分”即可继续。
6. 完成60例后点击“导出我的评分包”。最终文件名应以 `_complete60.zip` 结尾；`_partialN.zip` 仅用于中途备份，不能作为最终提交。

### B. 可解释性区域复核（20例）

逐题查看 `EIR-001` 至 `EIR-020`，填写并保存两张表：

- `case_ratings_template.csv`：病例整体、候选区域覆盖及是否漏框；
- `region_ratings_template.csv`：每个候选框的病理相关性和边界质量。

不要修改表头、`case_id` 或 `region_id`。建议另存为：

```text
roi_<Reviewer ID>_case_completed.csv
roi_<Reviewer ID>_region_completed.csv
```

### C. Stage2/Stage3 生成质量盲评（100例）

逐题比较匿名回答 A/B，填写 `stage2_stage3_blind_pairwise_100/ratings_template.csv`：

- 总体偏好只能填写 `A`、`B` 或 `Tie`；
- 信心、医学事实、视觉依据和推理质量均按表中要求填写 1–5；
- 不因回答更长而加分；两者同样好或同样差时填写 `Tie`。

建议另存为：

```text
generation_<Reviewer ID>_completed.csv
```

## 四、提交结果

请将以下四个文件原样交给负责人：

1. `pathvlm_reward_review_<Reviewer ID>_complete60.zip`
2. `roi_<Reviewer ID>_case_completed.csv`
3. `roi_<Reviewer ID>_region_completed.csv`
4. `generation_<Reviewer ID>_completed.csv`

可将四个文件放入文件夹 `pathvlm_review_return_<Reviewer ID>/` 后整体压缩发送，但不要解开或改名第1项奖励 ZIP。提交前请检查：Reviewer ID 一致、奖励包为 `complete60`、两张 ROI 表及生成质量表均无漏题。

如遇页面无法打开、保存失败或导出文件不是 `complete60`，请保留当前解压目录，不要重装或删除 `expert_submissions/`，并联系负责人处理。
