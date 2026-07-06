import os
import re
import json
import random
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import LlavaQwen2ForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块可用
from llava.model import *
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

# -------------------------------
# 加载模型与处理器
# -------------------------------
model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=f"cuda:{cuda_n}",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)



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
    
    QUESTION_TEMPLATE = (
            "{Question}\n"
            "First, analyze the image and options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        )
    
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
    processor.tokenizer.padding_side = "left"
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
        inputs = inputs.to(f"cuda:{cuda_n}")
        

        # 模型推理，生成答案
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
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
