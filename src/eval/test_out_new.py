import os
import re
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块可用
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/output/Qwen2.5-VL-3B-GRPO_3.270259_mmucontinue_4500/checkpoint-1600",
                        help="Path to the model checkpoint")
    parser.add_argument("--output_path", type=str, 
                        default="/root/autodl-tmp/output/logs/7b-continue-1600.json",
                        help="Output file path")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size")
    parser.add_argument("--data_root", type=str, 
                        default="/root/PathMMU/dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--test_datasets", type=str, default="without_cot_2000_val",
                        help="Comma-separated list of test datasets")
    parser.add_argument("--image_root", type=str, 
                        default="/root/PathMMU/images",
                        help="Root directory of images")
    parser.add_argument("--sample_num", type=int, default=5,
                        help="Number of samples")
    
    parser.add_argument("--cuda", type=int, default=0,
                        help="Number of samples")
    return parser.parse_args()
# -------------------------------
# 参数设置
# -------------------------------
# steps = 1600
# print("Steps:", steps)
# MODEL_PATH = "/root/autodl-tmp/output/Qwen2.5-VL-3B-GRPO_3.270259_mmucontinue_4500/checkpoint-1600"
# OUTPUT_PATH = "/root/autodl-tmp/output/logs/7b-continue-1600.json"
# BSZ = 32
# DATA_ROOT = "/root/PathMMU/dataset"
# TEST_DATASETS = ['without_cot_2000_val']
# IMAGE_ROOT = "/root/PathMMU/images"
# sample_num=500
args = parse_args()
    
# 使用解析后的参数
cuda_n = args.cuda
steps = args.steps
print("Steps:", steps)
MODEL_PATH = args.model_path
OUTPUT_PATH = args.output_path
BSZ = args.bsz
DATA_ROOT = args.data_root
TEST_DATASETS = args.test_datasets.split(',')  # 将逗号分隔的字符串转为列表
IMAGE_ROOT = args.image_root
sample_num = args.sample_num
random.seed(42)

# -------------------------------
# 加载模型与处理器
# -------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=f"cuda:{cuda_n}",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# # -------------------------------
# # 定义评估函数：检查模型输出中 <answer> 块仅包含一个 yes/no 且与参考答案一致
# # -------------------------------
# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     对每个样本的生成结果进行评估：
#     - 提取 <answer> 块内的内容
#     - 检查生成的答案中是否恰好出现一个 "yes" 或 "no"
#     - 如果与参考答案中的 yes/no 完全匹配，则视为正确
#     返回：
#       rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
#       accuracy: 整体准确率（百分比）+
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



def evaluate_yes_no(completions, solutions, debug=False):
    """
    对每个样本的生成结果进行评估：
    - 提取生成内容中的 <answer> 块内容
    - 与参考答案的完整文本进行全字匹配
    返回：
      rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
      accuracy: 整体准确率（百分比）
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            # 标准答案直接使用原始文本（不含任何标签）
            gold_answer = re.search(r'<answer>(.*?)</answer>', sol.strip(), re.DOTALL)
            gold_answer = gold_answer.group(1).strip() if gold_answer else sol.strip()
            print(gold_answer)
            # gold_answer = sol.strip()

            # 提取生成内容中的 <answer> 块（必须存在）
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


import re
import os
from datetime import datetime

# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     对每个样本的生成结果进行评估：
#     - 同时提取生成内容和参考答案中的 <answer> 块内容
#     - 进行全字匹配验证
#     返回：
#       rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 提取参考答案中的 <answer> 块内容
#             gold_answer_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
#             if not gold_answer_match:
#                 raise ValueError("参考答案中缺失 <answer> 标签")
#             gold_answer = gold_answer_match.group(1).strip()

#             # 提取生成内容中的 <answer> 块内容
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

# def evaluate_yes_no(completions, solutions, debug=False):
#     """Reward function that strictly checks matching of <answer> block contents"""
    
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for content, sol in zip(contents, solutions):
#         reward = 0.0
#         try:
#             # 从标准答案提取<answer>内容
#             gold_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
#             if not gold_match:
#                 raise ValueError("Missing required <answer> tags in solution")
#             gold_answer = gold_match.group(1).strip()

#             # 从生成内容提取<answer>内容
#             content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not content_match:
#                 raise ValueError("Missing required <answer> tags in generated content")
#             student_answer = content_match.group(1).strip()

#             # 严格全字匹配验证
#             if student_answer == gold_answer:
#                 reward = 1.0

#         except Exception as e:
#             if os.getenv("DEBUG_MODE") == "true":
#                 log_path = os.getenv("LOG_PATH")
#                 with open(log_path, "a") as f:
#                     f.write(f"------------- {current_time} Accuracy reward ERROR: {str(e)} -------------\n")
#                     f.write(f"Content: {content}\n")
#                     f.write(f"Solution: {sol}\n")
#             reward = 0.0

#         rewards.append(reward)

#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             with open(log_path, "a") as f:
#                 f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Solution: {sol}\n")
#     return rewards

# import re
# import os
# from datetime import datetime

# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     评估生成结果：
#     - 检查 <answer> 块是否存在
#     - 确保 <answer> 块内包含 <bool> 块
#     - 提取 <bool> 块内的 yes/no，且仅有一个
#     - 比较提取出的 yes/no 是否与参考答案一致
#     返回：
#       rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 提取参考答案中的 yes/no（转换为小写）
#             gold_answer_match = re.search(r'\b(yes|no)\b', sol, re.IGNORECASE)
#             gold_answer = gold_answer_match.group(1).lower() if gold_answer_match else None

#             # 提取生成内容中的 <answer> 块（必须存在）
#             content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not content_answer_match:
#                 raise ValueError("生成内容中缺失 <answer> 标签")
#             student_answer_block = content_answer_match.group(1).strip()

#             # 提取 <bool> 块（必须存在）
#             bool_match = re.search(r'<bool>(.*?)</bool>', student_answer_block, re.DOTALL)
#             if not bool_match:
#                 raise ValueError("<answer> 块中缺少 <bool> 块")
#             bool_content = bool_match.group(1).strip()

#             # 提取 <bool> 内的 yes/no（必须仅有一个）
#             student_bool_matches = re.findall(r'\b(yes|no)\b', bool_content, re.IGNORECASE)
#             if len(student_bool_matches) != 1:
#                 raise ValueError("<bool> 块中应仅包含一个 'yes' 或 'no'")
#             student_answer = student_bool_matches[0].lower()

#             # 仅当答案匹配时给予奖励
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

#     accuracy = sum(rewards) / len(rewards) * 100 if rewards else 0.0
#     return rewards, accuracy





# -------------------------------
# 从测试数据集中读取数据并构造对话消息
# -------------------------------
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds} ...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    # if sample_num != 1385:
    data = data[:sample_num]

    # 设置问题模板，其中问题部分由数据中 'problem' 字段填充
    # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    # QUESTION_TEMPLATE = (
    #         "{Question} A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    #         "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    #         "The reasoning process is enclosed within <think> </think> tags. "
    #         "The final answer must be enclosed within <answer> </answer> tags. "
    #         "The user's questions always require a Yes or No answer. The answer must be explicitly stated as 'Yes' or 'No' and "
    #         "wrapped within <bool> tags inside <answer>, followed by a brief explanation. "
    #         "Example: <think> reasoning here </think> <answer> <bool>Yes</bool> Additional explanations here </answer> "
    #     )
    

    # QUESTION_TEMPLATE = (
    #         "{Question} First, consider the reasoning process in your mind before providing the answer to the user." 
    #         "The reasoning process should be enclosed within <think> </think> tags." 
    #         "The final answer must be enclosed within <answer> </answer> tags, "
    #         "and the answer should be given first with 'Yes' or 'No' wrapped in <bool> </bool> tags."
    #         "Example: <think> reasoning here </think> <answer> <bool>Yes</bool> </answer>"
    #         "Example: <think> reasoning here </think> <answer> <bool>No</bool> </answer>"
    #     )
    
    # QUESTION_TEMPLATE = (
    #         "{Question} First, consider the reasoning process in your mind before providing the answer." 
    #         "The reasoning process should be enclosed within <think></think> tags." 
    #         "Although the question may be difficult to judge, you must provide a final answer of 'Yes' or 'No', enclosed within <answer></answer> tags." 
    #         "Example: <think> Here is the reasoning process </think> <answer> Yes </answer>" 
    #         "Example: <think> Here is the reasoning process </think> <answer> No </answer>"
    #     )
    # QUESTION_TEMPLATE = ("A conversation between User and Assistant. The user provides a picture and asks a question, "
    #                  "and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides "
    #                  "the user \"yes\" or \"no\" as the answer. "
    #                  "The reasoning process is enclosed within <think> </think> tags. "
    #                  "The answer must be enclosed within <answer> </answer> tags. "
    #                  "Followed by two brief right examples.\n "
    #                  "Example1: <think> reasoning here </think><answer> \"yes\" </answer>\n" 
    #                  "Example2: <think> reasoning here </think><answer> \"no\" </answer>\n"
    #                  "{Question}")
    
    # 3.18 早上
    # QUESTION_TEMPLATE = (
    #         "{Question} First, consider the reasoning process in your mind before providing the answer." 
    #         "The reasoning process should be enclosed within <think></think> tags." 
    #         "If the question is difficult to judge, you should think more and provide a definite final answer of 'Yes' or 'No', enclosed within <answer></answer> tags." 
    #         "Example: <think> Here is the reasoning process </think> <answer> Yes </answer>" 
    #         "Example: <think> Here is the reasoning process </think> <answer> No </answer>"
    #     )
    
    # 我们将对同一批样本使用三种 prompt 变体：
    # 1) baseline: 原始 prompt（分析 image features + option elimination）
    # 2) no_image: 明确指示不要分析 image features，只基于选项与医学知识进行推理
    # 3) no_option_elim: 明确指示不要进行逐项 option elimination，只给出结论性推理
    variants = {
        'baseline': (
            "{Question}\n"
            "First, analyze the image and options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        ),
        'no_image': (
            "{Question}\n"
            "First, analyze the options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        ),
        'no_option_elim': (
            "{Question}\n"
            "First, analyze the image within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        )
    }
    # 新增变体：no_think -> 直接提供答案，不输出思维链条
    variants['no_think'] = (
        "{Question}\n"
        "Provide only the final answer enclosed in <answer> tags with the correct answer letter and full option text. Example: <answer>Full text of the correct option</answer>"
    )

    # 存储所有变体的输出与评估结果，方便后续合并
    outputs_by_variant = {}
    rewards_by_variant = {}
    acc_by_variant = {}
    token_counts_by_variant = {}
    avg_tokens_by_variant = {}

    # 逐个变体执行生成与评估
    for vname, vtemplate in variants.items():
        print(f"\nRunning variant: {vname}")

        # 构造对话消息，每条消息包含图像和文本（用户角色）
        messages = []
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
                            "text": vtemplate.format(Question=x['problem'])
                        }
                    ]
                }
            ]
            messages.append(message)

        # -------------------------------
        # 批量生成模型回答（与原流程一致）
        # -------------------------------
        processor.tokenizer.padding_side = "left"
        all_outputs = []  # 存储该变体下的所有生成文本
        for i in tqdm(range(0, len(messages), BSZ), desc=f"Generating answers ({vname})"):
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
            inputs = inputs.to(f"cuda:{cuda_n}")

            # 模型推理，生成答案
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            # 由于生成的结果包括输入部分，因此去掉输入 token 部分
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_outputs.extend(batch_output_text)

        # -------------------------------
        # 模型评估 + token 统计
        # -------------------------------
        solutions = [x['solution'] for x in data]
        rewards, acc = evaluate_yes_no(all_outputs, solutions, debug=True)
        print(f"\nVariant {vname} Accuracy of {ds}: {acc:.2f}%")

        # 计算每个输出的 token 数（尽量兼容 tokenizer API）
        token_counts = []
        for txt in all_outputs:
            try:
                ids = processor.tokenizer.encode(txt, add_special_tokens=False)
                cnt = len(ids)
            except Exception:
                try:
                    enc = processor.tokenizer(txt, return_tensors='pt')
                    cnt = int(enc['input_ids'].shape[1])
                except Exception:
                    cnt = len(txt.split())
            token_counts.append(int(cnt))

        avg_tokens = float(np.mean(token_counts)) if len(token_counts) > 0 else 0.0

        # 构造详细结果列表并保存为单独文件（包含 token count）
        final_output = []
        for input_example, model_output, r, tok in zip(data, all_outputs, rewards, token_counts):
            result = {
                'question': input_example['problem'],
                'ground_truth': input_example['solution'],
                'model_output': model_output,
                'correct': bool(r),
                'token_count': int(tok)
            }
            final_output.append(result)

        base_output_path = OUTPUT_PATH.format(DATASET=ds, STEPS=steps)
        variant_output_path = base_output_path.replace('.json', f'_{vname}.json')
        with open(variant_output_path, "w", encoding='utf-8') as f:
            json.dump({
                'accuracy': acc,
                'avg_output_tokens': avg_tokens,
                'results': final_output
            }, f, indent=2, ensure_ascii=False)

        print(f"Results for variant '{vname}' saved to {variant_output_path} (avg tokens: {avg_tokens:.1f})")
        print("-" * 100)

        # 保持在内存中，供合并使用
        outputs_by_variant[vname] = all_outputs
        rewards_by_variant[vname] = rewards
        acc_by_variant[vname] = acc
        token_counts_by_variant[vname] = token_counts
        avg_tokens_by_variant[vname] = avg_tokens

    # -------------------------------
    # 合并保存三种变体的对比结果（每条样本包含三个变体的输出与 correctness）
    # -------------------------------
    combined = []
    for i, x in enumerate(data):
        row = {
            'question': x['problem'],
            'ground_truth': x['solution']
        }
        for vname in variants.keys():
            row[f'model_output_{vname}'] = outputs_by_variant[vname][i]
            row[f'correct_{vname}'] = bool(rewards_by_variant[vname][i])
        combined.append(row)

    combined_output_path = base_output_path.replace('.json', '_combined_variants.json')
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'variant_accuracies': acc_by_variant,
            'per_sample_comparison': combined
        }, f, indent=2, ensure_ascii=False)

    print(f"Combined variant results saved to {combined_output_path}")
    print("-" * 100)
