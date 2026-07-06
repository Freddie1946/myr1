from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import random
from datetime import datetime

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps")
    parser.add_argument("--model_path", type=str, 
                        default="/home/wjy/LLaMA-Factory/saves/qwen2_5_vl-3b-3.270256_without_cot_continue750-1000/full/sft/checkpoint-1250",
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
                        help="Number of cuda")
    return parser.parse_args()

# ---------------------------
# 参数设置
# ---------------------------

# MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
# OUTPUT_PATH = "./logs/rec_results_{DATASET}_3bzeroshot.json"
# # print("250")
# BSZ = 8
# DATA_ROOT = "/home/wjy/PathMMU"
# TEST_DATASETS = ['noanswer_without_cot_2000_val']  # 测试数据集名称列表
# IMAGE_ROOT = "/home/wjy/PathMMU/images"
random.seed(42)
# sample_num = 500
args = parse_args()
    
# 使用解析后的参数
steps = args.steps
print("Steps:", steps)
MODEL_PATH = args.model_path
OUTPUT_PATH = args.output_path
BSZ = args.bsz
DATA_ROOT = args.data_root
TEST_DATASETS = args.test_datasets.split(',')  # 将逗号分隔的字符串转为列表
IMAGE_ROOT = args.image_root
sample_num = args.sample_num
cuda = args.cuda


# ---------------------------
# 加载模型与处理器
# ---------------------------
# 使用 flash_attention_2 加速并节省显存，模型加载到 GPU 0
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=f"cuda:{cuda}",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ---------------------------
# 定义评估函数：严格检查生成答案中的 <answer> 块仅包含一个 yes 或 no，
# 并与参考答案中的 yes/no 进行比较，返回每个样本奖励及整体准确率
# ---------------------------
# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     参数:
#       completions: 模型生成的输出列表，每个元素为字符串
#       solutions: 参考答案列表，每个答案应包含正确的 yes/no 信息
#       debug: 是否开启调试日志（默认 False）
      
#     返回:
#       rewards: 每个样本的奖励列表（1.0 表示正确，0.0 表示错误）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 从参考答案中提取 yes 或 no（不区分大小写）
#             gold_bool_match = re.search(r'\b(yes|no)\b', sol, re.IGNORECASE)
#             gold_answer = gold_bool_match.group(1).lower() if gold_bool_match else None
            
#             # 从模型生成的回答中提取所有出现的 yes 或 no
#             student_bool_matches = re.findall(r'\b(yes|no)\b', content, re.IGNORECASE)
            
#             # 检查生成内容中恰好只有一个 yes 或 no
#             if len(student_bool_matches) != 1:
#                 raise ValueError("生成内容中应仅包含一个 'yes' 或 'no'")
            
#             student_answer = student_bool_matches[0].lower()
            
#             # 如果生成的答案与参考答案匹配，则奖励 1
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

# 严格全字匹配
# def evaluate_yes_no(completions, solutions, debug=False):
#     """
#     参数:
#       completions: 模型生成的完整输出列表（字符串格式）
#       solutions: 参考答案列表（完整文本）
#       debug: 是否开启调试日志（默认 False）
      
#     返回:
#       rewards: 每个样本的奖励列表（1.0 表示完全匹配，0.0 表示不匹配）
#       accuracy: 整体准确率（百分比）
#     """
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
#     for content, sol in zip(completions, solutions):
#         reward = 0.0
#         try:
#             # 标准化处理（去除首尾空格）
#             gold_answer = sol.strip()
#             student_answer = content.strip()
            
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
#                 f.write(f"参考答案: {gold_answer}\n")  # 显示完整答案
#                 f.write(f"生成内容: {student_answer}\n")  # 显示完整生成内容
    
#     accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
#     return rewards, accuracy


def evaluate_yes_no(completions, solutions, debug=False):
    """
    修改后的评估函数：
    - 提取标准答案中的 <answer> 块内容
    - 直接使用生成内容的完整文本进行全字匹配
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            # 从标准答案中提取 <answer> 块内容
            gold_match = re.search(r'<answer>(.*?)</answer>', sol.strip(), re.DOTALL)
            if not gold_match:
                raise ValueError("标准答案中缺失 <answer> 标签")
            gold_answer = gold_match.group(1).strip()

            # 生成内容直接使用原始文本（不含任何标签）
            student_answer = content.strip()

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


# import re
# import os
# from datetime import datetime

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
# ---------------------------
# 主评估流程
# ---------------------------
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds} ...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    if sample_num != 1385:
        data = data[:sample_num]
    
    # 设置问题模板：这里仅使用问题文本，您可以根据需要调整模板
    QUESTION_TEMPLATE = (
            "{Question}\n"
            "First, analyze the image and options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        )
    
    messages = []
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        # 构造对话消息：user 的内容包含图像与文本
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
    
    # ---------------------------
    # 批量生成模型回答
    # ---------------------------
    all_outputs = []  # 存储生成的答案文本
    for i in tqdm(range(0, len(messages), BSZ), desc="Generating answers"):
        batch_messages = messages[i:i + BSZ]
        # 对每个消息应用预定义模板转换为输入文本
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        # 处理图像和视频信息
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{cuda}")
        
        # 模型生成输出，设置最大生成 token 数为256，不使用采样（贪婪解码）
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        # 由于生成的结果包含原始输入部分，需去除这些部分，仅保留新生成的 token
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(batch_output_text)
    
    # ---------------------------
    # 严格评估：检查生成回答是否仅包含一个 yes 或 no 并与参考答案匹配
    # ---------------------------
    # 假设数据中 solution 字段存放参考答案，答案中应包含 <answer> 标签及正确的 yes/no 信息
    solutions = [x['solution'] for x in data]
    rewards, acc = evaluate_yes_no(all_outputs, solutions, debug=True)
    print(f"\nAccuracy of {ds}: {acc:.2f}%")
    
    # 构造详细结果列表，记录每个样本的评估信息
    final_output = []
    for input_example, model_output, r in zip(data, all_outputs, rewards):
        # 从生成内容中提取 yes/no 作为 extracted_answer
        try:
            answer_match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            extracted = answer_match.group(1).strip() if answer_match else ""
            bool_matches = re.findall(r'\b(yes|no)\b', extracted, re.IGNORECASE)
            extracted_answer = bool_matches[0].lower() if len(bool_matches) == 1 else "error"
        except Exception:
            extracted_answer = "error"
            
        result = {
            'question': input_example['problem'],
            'ground_truth': input_example['solution'],
            'model_output': model_output,
            'extracted_answer': extracted_answer,
            'correct': bool(r)
        }
        final_output.append(result)
    
    # ---------------------------
    # 保存评估结果
    # ---------------------------
    output_path = OUTPUT_PATH.format(DATASET=ds)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': acc,
            'results': final_output
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print("-" * 100)
