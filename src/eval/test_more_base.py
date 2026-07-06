import os
import re
import json
import random
import base64
import http.client
from tqdm import tqdm
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps")
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
    parser.add_argument("--api_token", type=str, default="sk-iAswseARFAisfM3jl4DhWcEJG61sKDj6Qd3chI6AsYoy9Ni1",
                        help="API token for model access")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Model name to call via API")
    return parser.parse_args()

args = parse_args()
random.seed(42)

# -------------------------------
# 参数设置
# -------------------------------
steps = args.steps
BSZ = args.bsz
DATA_ROOT = args.data_root
TEST_DATASETS = args.test_datasets.split(',')
IMAGE_ROOT = args.image_root
sample_num = args.sample_num
API_TOKEN = args.api_token
MODEL_NAME = args.model_name
OUTPUT_PATH = args.output_path

# -------------------------------
# 定义评估函数
# -------------------------------
def evaluate_yes_no(completions, solutions, debug=False):
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            gold_answer_match = re.search(r'<answer>(.*?)</answer>', sol.strip(), re.DOTALL)
            gold_answer = gold_answer_match.group(1).strip() if gold_answer_match else sol.strip()

            content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if not content_answer_match:
                raise ValueError("生成内容中缺失 <answer> 标签")
            student_answer = content_answer_match.group(1).strip()

            if student_answer == gold_answer:
                reward = 1.0

        except Exception as e:
            if debug:
                log_path = os.getenv("LOG_PATH", "debug_log.txt")
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} ERROR: {str(e)} -------------\n")
                    f.write(f"生成内容: {content}\n参考答案: {sol}\n")
            reward = 0.0

        rewards.append(reward)
        if debug:
            log_path = os.getenv("LOG_PATH", "debug_log.txt")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Reward: {reward} -------------\n")
                f.write(f"生成内容: {content}\n参考答案: {sol}\n")

    accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
    return rewards, accuracy

# -------------------------------
# Helper: 将本地图片编码为Base64
# -------------------------------
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

# -------------------------------
# Helper: 调用API获取回答
# -------------------------------
# def call_api(messages):
#     conn = http.client.HTTPSConnection("api2.aigcbest.top")
#     payload = json.dumps({
#         "model": MODEL_NAME,
#         "messages": messages
#     })
#     headers = {
#         'Accept': 'application/json',
#         'Authorization': f'Bearer {API_TOKEN}',
#         'Content-Type': 'application/json'
#     }
#     conn.request("POST", "/v1/chat/completions", payload, headers)
#     res = conn.getresponse()
#     data = res.read()
#     response_json = json.loads(data)
#     # 返回模型生成的文本列表
#     outputs = []
#     for choice in response_json.get("choices", []):
#         # 不同API版本字段可能略有不同
#         content = choice.get("message", {}).get("content", "")
#         outputs.append(content)
#     return outputs
def call_api(messages):
    proxy_host = "10.128.100.233"
    proxy_port = 7890

    # 建立 HTTPSConnection 到代理
    conn = http.client.HTTPSConnection(proxy_host, proxy_port)
    # 设置隧道到目标服务器
    conn.set_tunnel("api2.aigcbest.top", 443)

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_json = json.loads(data)
    print(response_json)

    # 返回模型生成的文本列表
    outputs = []
    for choice in response_json.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        outputs.append(content)
    return outputs

# -------------------------------
# 从测试数据集中读取数据并构造对话消息
# -------------------------------
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds} ...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    data = data[:sample_num]

    QUESTION_TEMPLATE = (
            "{Question}\n"
            "First, analyze the image and options step-by-step within <think> tags. "
            "Then, provide the correct answer letter with full option text in <answer> tags.\n"
            "Example format:\n"
            "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        )

    messages_list = []
    for x in data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        image_base64 = encode_image_base64(image_path)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x['problem'])},
                    {"type": "image_base64", "image_base64": image_base64}
                ]
            }
        ]
        messages_list.append(message)

    # -------------------------------
    # 批量生成模型回答
    # -------------------------------
    all_outputs = []
    for i in tqdm(range(0, len(messages_list), BSZ), desc="Generating answers"):
        batch_messages = messages_list[i:i+BSZ]
        # 直接调用API
        batch_outputs = call_api(batch_messages)
        all_outputs.extend(batch_outputs)

    # -------------------------------
    # 模型评估
    # -------------------------------
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
