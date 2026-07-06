import os
import re
import json
import random
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块可用
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--steps", type=int, default=2, help="Number of forced budgeting steps (chain-of-thought steps)")
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
    return parser.parse_args()

args = parse_args()
    
# 使用解析后的参数
steps = args.steps
print("Forced budgeting steps:", steps)
MODEL_PATH = args.model_path
OUTPUT_PATH = args.output_path
BSZ = args.bsz
DATA_ROOT = args.data_root
TEST_DATASETS = args.test_datasets.split(',')  # 将逗号分隔的字符串转为列表
IMAGE_ROOT = args.image_root
# sample_num = 10
random.seed(42)

# -------------------------------
# 加载模型与处理器
# -------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:3",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

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
    # data = data[:sample_num]

    QUESTION_TEMPLATE = (
            "{Question}\n"
            # "First, analyze the image and options step-by-step within <think> tags. "
            # "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            # "Example format:\n"
            # "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
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
    import copy
    processor.tokenizer.padding_side = "left"
    all_outputs = []  # 存储所有生成的答案文本
    for i in tqdm(range(0, len(messages), BSZ), desc="Generating answers"):
        batch_messages = messages[i:i + BSZ]
        
        # 使用处理器生成输入文本
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        # print("text:", text)
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:3")
        # daima
        if steps > 0:
            
            batch_buffer = copy.deepcopy(batch_messages)
            for buffer in batch_buffer:
                    for bf in buffer:
                        if bf["role"] == "user":
                            for content_item in bf["content"]:
                                if content_item["type"] == "text":
                                    content_item["text"] += "First, analyze the image and options step-by-step, then provide the correct answer. <|im_start|>Analyze the image and options step-by-step:\n"
                                    # content_item["text"] += "<|im_start|>Analyze the image and options step-by-step:\n"
                                    # print("buffer:", buffer)
                                    break
            # 多步生成中间推理文本，每一步都更新生成上下文
            for step in range(steps):
                print("step:", step)
                print("batch_buffer:", batch_buffer)
                print("\n")
                forced_text = []
                forced_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_buffer]
                
                # print("forced_text:\n\n", forced_text)
                print("force text:", forced_text)
                print("\n")
                forced_inputs = processor(
                    text=forced_text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda:3")
                # 每步生成一定数量的 token（这里设为512，可根据实际情况调整）
                new_generated_ids = model.generate(**forced_inputs, use_cache=True, max_new_tokens=512, do_sample=False)

                new_generated = processor.batch_decode(new_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print("new_generated:\n\n", new_generated)

                new_generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(forced_inputs.input_ids, new_generated_ids)]
                new_generated_texts = processor.batch_decode(new_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                tags = ["<think>", "</think>", "<answer>", "</answer>"]

                # 创建一个空列表用于存储处理后的字符串
                cleaned_texts = []

                # 遍历列表中的每个字符串
                for text in new_generated_texts:
                    # 依次移除每个标签
                    for tag in tags:
                        text = text.replace(tag, "")
                    # 将处理后的字符串添加到新列表中
                    cleaned_texts.append(text)
                print("new_generated_texts:\n", new_generated_texts)
                print("\n")
                print("cleaned_texts:",cleaned_texts)
                print("\n")
                # 更新 forced_text，追加生成内容以及一个 "Wait" 标记以便突破停止 token 限制
                for buffer, nt in zip(batch_buffer, cleaned_texts):
                    for bf in buffer:
                        if bf["role"] == "user":
                            for content_item in bf["content"]:
                                if content_item["type"] == "text":
                                    content_item["text"] += nt
                                    content_item["text"] += " Wait, "
                                    # print("buffer:", buffer)
                                    break
                print("new_batch_buffer:", batch_buffer)
                print("\n")
            for buffer in batch_buffer:
                    for bf in buffer:
                        if bf["role"] == "user":
                            for content_item in bf["content"]:
                                if content_item["type"] == "text":
                                    content_item["text"] += "\nProvide the correct answer letter with full option text in <answer></answer> tags."
                                    # print("buffer:", buffer)
                                    break
            final_buffer = copy.deepcopy(batch_buffer)

            # print("final buffer:\n", final_buffer)
            forced_text = [ ]
            
            forced_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in final_buffer]
            print("Final_input:", forced_text)
            print("\n")
            final_inputs = processor(
                    text=forced_text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda:3")
            final_generated_ids = model.generate(**final_inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            final_generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(final_inputs.input_ids, final_generated_ids)]
            batch_output_text = processor.batch_decode(final_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("batch_output_text:\n", batch_output_text)
            print("\n")
        else:
            # ----- 普通生成 -----
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