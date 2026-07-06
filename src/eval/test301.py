import os
import json
import random
import base64
import time
import argparse
from datetime import datetime
from openai import OpenAI

# -------------------------------
# 参数解析
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps")
    parser.add_argument("--output_path", type=str, default="./logs/api_test_results.json", help="Output file path")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Root directory of the dataset")
    parser.add_argument("--test_datasets", type=str, default="without_cot_2000_val", help="Comma-separated list of test datasets")
    parser.add_argument("--image_root", type=str, default="./images", help="Root directory of images")
    parser.add_argument("--sample_num", type=int, default=5, help="Number of samples")
    parser.add_argument("--api_key", type=str, default="sk-0yL4gU76Tis0cOqUa3vAbkhOwpIYHb3IGcWeIPCZlVgO6VJ3", help="API key for model")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="API model name")
    return parser.parse_args()

args = parse_args()
random.seed(42)

# -------------------------------
# API 客户端初始化
# -------------------------------
client = OpenAI(
    base_url="https://api.302.ai/v1/",
    api_key=args.api_key
)

# -------------------------------
# 工具函数
# -------------------------------
def encode_image(image_path):
    """读取本地图片并转 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# -------------------------------
# 评估函数（保留原逻辑）
# -------------------------------
import re
def evaluate_yes_no(completions, solutions, debug=False):
    """
    评估生成结果：
    - 检查生成内容中的 <answer> 块
    - 如果 <answer> 中包含参考答案首字母（A/B/C/D）即视为正确
    返回：
    rewards: 每个样本的奖励（1.0 为正确，0.0 为错误）
    accuracy: 整体准确率（百分比）
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            # 提取参考答案 <answer> 内的首字母
            gold_match = re.search(r'<answer>\s*([A-Z])', sol, re.DOTALL | re.IGNORECASE)
            if not gold_match:
                raise ValueError("参考答案中缺少选项字母")
            gold_option = gold_match.group(1).upper()
            
            # 提取生成内容中的 <answer> 块
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if not content_match:
                raise ValueError("生成内容中缺失 <answer> 标签")
            student_answer_block = content_match.group(1).strip()
            
            # 判断生成内容是否包含参考答案字母
            if re.search(r'\b' + re.escape(gold_option) + r'\b', student_answer_block, re.IGNORECASE):
                reward = 1.0
        except Exception as e:
            if debug:
                log_path = os.getenv("LOG_PATH", "debug_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} ERROR: {str(e)} -------------\n")
                    f.write(f"生成内容: {content}\n参考答案: {sol}\n")
            reward = 0.0
        
        rewards.append(reward)

    accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
    return rewards, accuracy

# -------------------------------
# 处理测试数据集
# -------------------------------
TEST_DATASETS = args.test_datasets.split(',')
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds} ...")
    ds_path = os.path.join(args.data_root, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    data = data[:args.sample_num]

    # 构造问题模板
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "First, analyze the image and options step-by-step within <think> tags. "
        "Then, provide the correct answer letter with full option text in <answer> tags.\n"
        "Example format:\n"
        "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
    )

    # 构造消息列表
    messages = []
    for x in data:
        image_path = os.path.join(args.image_root, x['image'])
        base64_image = encode_image(image_path)
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x['problem'])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        messages.append(msg)

    # -------------------------------
    # 批量调用 API 生成回答
    # -------------------------------
    all_outputs = []
    for i in range(0, len(messages), args.bsz):
        batch = messages[i:i + args.bsz]
        for msg in batch:
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=msg,
                    temperature=0
                )
                # API 返回结果
                content = response.choices[0].message.content
                all_outputs.append(content)
            except Exception as e:
                print(f"API request failed: {e}")
                all_outputs.append("")

    # -------------------------------
    # 模型评估
    # -------------------------------
    solutions = [x['solution'] for x in data]
    rewards, acc = evaluate_yes_no(all_outputs, solutions, debug=True)
    print(f"\nAccuracy of {ds}: {acc:.2f}%")

    # -------------------------------
    # 保存结果
    # -------------------------------
    final_output = []
    for input_example, model_output, r in zip(data, all_outputs, rewards):
        final_output.append({
            'question': input_example['problem'],
            'ground_truth': input_example['solution'],
            'model_output': model_output,
            'correct': bool(r)
        })

    output_path = args.output_path
    with open(output_path, "w") as f:
        json.dump({'accuracy': acc, 'results': final_output}, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-" * 100)
