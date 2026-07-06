import os
import re
import json
import random
import torch
from tqdm import tqdm
from datetime import datetime
import argparse
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

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
                        help="CUDA device number")
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

if torch.cuda.is_available():
    target_device = f"cuda:{cuda_n}"
    torch.cuda.set_device(cuda_n)
else:
    target_device = "cpu"

# -------------------------------
# 加载模型与处理器
# -------------------------------

# 官方推荐方式：只用 load_pretrained_model 返回的 tokenizer, model, image_processor
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name="llava-med-v1.5-mistral-7b",
    device=target_device,
    device_map=target_device if target_device != "cuda" else "auto"
)

model_name = get_model_name_from_path(MODEL_PATH)
conv_mode = "mistral_instruct"
print(f"[DEBUG] Forced conv_mode: {conv_mode}")
print(f"[DEBUG] conv_templates['mistral_instruct'] system: {conv_templates['mistral_instruct'].system}")

image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
vision_tower = model.get_vision_tower()
try:
    vision_tower.to(device=target_device, dtype=torch.float16)
except Exception:
    pass
vision_device = getattr(vision_tower, "device", target_device)
vision_dtype = getattr(vision_tower, "dtype", torch.float16)


def build_prompt(question_text):
    qs = question_text
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()
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
    
    QUESTION_TEMPLATE = (
            "{Question}\n"
            "First, analyze the image and options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        )
    
    samples = []
    for x in data:
        samples.append(
            {
                "image_path": os.path.join(IMAGE_ROOT, x['image']),
                "question_text": QUESTION_TEMPLATE.format(Question=x['problem'])
            }
        )

    # -------------------------------
    # 批量生成模型回答
    # -------------------------------

    all_outputs = []  # 存储所有生成的答案文本
    from PIL import Image
    for i in tqdm(range(0, len(samples), BSZ), desc="Generating answers"):
        batch_samples = samples[i:i + BSZ]
        input_ids_list = []
        input_lengths = []
        prompts = []
        images = []
        image_sizes = []
        for sample in batch_samples:
            prompt = build_prompt(sample["question_text"])
            prompts.append(prompt)
            ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            input_ids_list.append(ids)
            input_lengths.append(ids.shape[0])
            image = Image.open(sample["image_path"]).convert("RGB")
            images.append(image)
            image_sizes.append(image.size)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_id
        ).to(target_device)
        image_tensors = process_images(
            images, image_processor, model.config
        ).to(vision_device, dtype=vision_dtype)
        attention_mask = (input_ids != pad_token_id).long().to(target_device)

        print(">>> NEW GENERATION PATH <<<", prompts[0][:120].replace("\n", " "))
        print(f"input_ids shape: {input_ids.shape if input_ids is not None else None}")
        print(f"image_tensors shape: {image_tensors.shape if image_tensors is not None else None}")
        assert input_ids is not None and len(input_ids.shape) == 2, f"input_ids is None or not 2D: {input_ids}"
        assert image_tensors is not None, "image_tensors is None"

        outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=image_tensors,
            image_sizes=image_sizes,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
        )
        # 直接解码完整序列，然后去掉前缀 prompt 文本
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cleaned = [text[len(pmpt):] if text.startswith(pmpt) else text for text, pmpt in zip(decoded, prompts)]
        print(f"Batch {i // BSZ}: decoded lengths {[len(x) for x in decoded]} cleaned lengths {[len(x) for x in cleaned]}")
        all_outputs.extend(cleaned)
    
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
