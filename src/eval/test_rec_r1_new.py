import os
import re
import json
import random
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块可用

# -------------------------------
# 参数设置
# -------------------------------
steps = 200
print("Steps:", steps)
MODEL_PATH = f"path/to/Qwen2.5-VL-3B-GRPO-REC/checkpoint-{steps}"
OUTPUT_PATH = "./logs/rec_results_{DATASET}_qwen2_5vl_3b_instruct_r1_{STEPS}.json"
BSZ = 32
DATA_ROOT = "/data/shz/project/vlm-r1/VLM-R1/src/data/rec_jsons_processed"
TEST_DATASETS = ['refgta_subsample']
IMAGE_ROOT = "/data/shz/dataset/refgta"
sample_num=5

random.seed(42)

# -------------------------------
# 加载模型与处理器
# -------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# -------------------------------
# 定义评估函数：检查模型输出中 <answer> 块仅包含一个 yes/no 且与参考答案一致
# -------------------------------
# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     对每个样本的生成结果进行评估：
#     - 提取 <answer> 块内的内容
#     - 检查生成的答案中是否恰好出现一个 "yes" 或 "no"
#     - 如果与参考答案中的 yes/no 完全匹配，则视为正确
#     返回：
#       rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 提取参考答案中的 <answer> 块
#             sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
#             gold_answer_block = sol_answer_match.group(1).strip() if sol_answer_match else sol.strip()

#             # 提取生成内容中的 <answer> 块（必须存在）
#             content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not content_answer_match:
#                 raise ValueError("生成内容中缺失 <answer> 标签")
#             student_answer_block = content_answer_match.group(1).strip()

#             # 提取参考答案中的 yes/no（转换为小写）
#             gold_bool_match = re.search(r'\b(yes|no)\b', gold_answer_block, re.IGNORECASE)
#             gold_answer = gold_bool_match.group(1).lower() if gold_bool_match else None

#             # 从生成内容中提取所有出现的 yes/no
#             student_bool_matches = re.findall(r'\b(yes|no)\b', student_answer_block, re.IGNORECASE)
#             # 检查生成内容中是否恰好只有一个 yes 或 no
#             if len(student_bool_matches) != 1:
#                 raise ValueError("生成内容中应仅包含一个 'yes' 或 'no'")
#             student_answer = student_bool_matches[0].lower()

#             if student_answer == gold_answer:
#                 reward = 1.0

#         except Exception as e:
#             if debug:
#                 log_path = os.getenv("LOG_PATH", "debug_log.txt")
#                 with open(log_path, "a") as f:
#                     f.write(f"------------- {current_time} ERROR: {str(e)} -------------\n")
#                     f.write(f"生成内容: {content}\n")
#                     f.write(f"参考答案: {sol}\n")
#             reward = 0.0

#         rewards.append(reward)
#         if debug:
#             log_path = os.getenv("LOG_PATH", "debug_log.txt")
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Reward: {reward} -------------\n")
#                 f.write(f"生成内容: {content}\n")
#                 f.write(f"参考答案: {sol}\n")

#     accuracy = sum(rewards) / len(rewards) * 100
#     return rewards, accuracy


# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     对每个样本的生成结果进行评估：
#     - 提取生成内容中的 <answer> 块内容
#     - 与参考答案的完整文本进行全字匹配
#     返回：
#       rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 标准答案直接使用原始文本（不含任何标签）
#             gold_answer = sol.strip()

#             # 提取生成内容中的 <answer> 块（必须存在）
#             content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not content_answer_match:
#                 raise ValueError("生成内容中缺失 <answer> 标签")
#             student_answer = content_answer_match.group(1).strip()

#             # 严格全字匹配验证
#             if student_answer == gold_answer:
#                 reward = 1.0

#         except Exception as e:
#             if debug:
#                 log_path = os.getenv("LOG_PATH", "debug_log.txt")
#                 with open(log_path, "a") as f:
#                     f.write(f"------------- {current_time} ERROR: {str(e)} -------------\n")
#                     f.write(f"生成内容: {content}\n")
#                     f.write(f"参考答案: {sol}\n")
#             reward = 0.0

#         rewards.append(reward)
#         if debug:
#             log_path = os.getenv("LOG_PATH", "debug_log.txt")
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Reward: {reward} -------------\n")
#                 f.write(f"生成内容: {content}\n")
#                 f.write(f"参考答案: {sol}\n")

#     accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
#     return rewards, accuracy


import re
import os
from datetime import datetime

def evaluate_yes_no(completions, solutions, debug=False):
    """
    对每个样本的生成结果进行评估：
    - 同时提取生成内容和参考答案中的 <answer> 块内容
    - 进行全字匹配验证
    返回：
      rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
      accuracy: 整体准确率（百分比）
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            # 提取参考答案中的 <answer> 块内容
            gold_answer_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            if not gold_answer_match:
                raise ValueError("参考答案中缺失 <answer> 标签")
            gold_answer = gold_answer_match.group(1).strip()

            # 提取生成内容中的 <answer> 块内容
            content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if not content_answer_match:
                raise ValueError("生成内容中缺失 <answer> 标签")
            student_answer = content_answer_match.group(1).strip()

            # 严格全字匹配验证
            if student_answer == gold_answer:
                reward = 1.0

        except Exception as e:
            if debug:
                log_path = os.getenv("LOG_PATH", "debug_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} ERROR: {str(e)} -------------\n")
                    f.write(f"生成内容: {content}\n")
                    f.write(f"参考答案: {sol}\n")
            reward = 0.0

        rewards.append(reward)
        if debug:
            log_path = os.getenv("LOG_PATH", "debug_log.txt")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Reward: {reward} -------------\n")
                f.write(f"生成内容: {content}\n")
                f.write(f"参考答案: {sol}\n")

    accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
    return rewards, accuracy



# -------------------------------
# 从测试数据集中读取数据并构造对话消息
# -------------------------------
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds} ...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    data = data[:sample_num]
    # 设置问题模板，其中问题部分由数据中 'problem' 字段填充
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

    messages = []
    # 构造对话消息，每条消息包含图像和文本（用户角色）
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}"
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                    }
                ]
            }
        ]
        messages.append(message)

    # -------------------------------
    # 批量生成模型回答
    # -------------------------------
    all_outputs = []  # 存储所有生成的答案文本
    for i in tqdm(range(0, len(messages), BSZ), desc="Generating answers"):
        batch_messages = messages[i:i + BSZ]
        
        # 使用处理器生成输入文本
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        # 模型推理，生成答案
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        # 由于生成的结果包括输入部分，因此去掉输入 token 部分
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_outputs.extend(batch_output_text)
    
    # -------------------------------
    # 模型评估
    # -------------------------------
    # 假设数据中 solution 字段存放参考答案，其中应包含 <answer> 标签和 yes/no 信息
    solutions = [x['solution'] for x in data]
    rewards, acc = evaluate_yes_no(all_outputs, solutions, debug=True)
    print(f"\nAccuracy of {ds}: {acc:.2f}%")

    # 构造详细结果列表
    final_output = []
    for input_example, model_output, r in zip(data, all_outputs, rewards):
        result = {
            'question': input_example['problem'],
            'ground_truth': input_example['solution'],
            'model_output': model_output,
            'correct': bool(r)
        }
        final_output.append(result)
    
    # 保存结果到输出文件
    output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': acc,
            'results': final_output
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print("-" * 100)
