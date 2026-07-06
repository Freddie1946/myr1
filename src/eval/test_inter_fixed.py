import os
import re
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
import argparse
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_args():
    parser = argparse.ArgumentParser(description="Program parameters")
    parser.add_argument("--model_path", type=str, 
                        default="OpenGVLab/InternVL3-8B",
                        help="Path to the model checkpoint")
    parser.add_argument("--output_path", type=str, 
                        default="./InternVL3-8B.json",
                        help="Output file path")
    parser.add_argument("--bsz", type=int, default=6, help="Batch size")
    parser.add_argument("--data_root", type=str, 
                        default="/home/wjy/PathMMU/dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--image_root", type=str, 
                        default="/home/wjy/PathMMU/images",
                        help="Root directory of images")
    parser.add_argument("--sample_num", type=int, default=500,
                        help="Number of samples")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device number")
    return parser.parse_args()

args = parse_args()
random.seed(42)

# 加载模型
print(f"Loading model from {args.model_path}...")
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=f'cuda:{args.cuda}'
).eval()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
print("Model loaded successfully!")

# 加载数据
TEST_DATASETS = ['without_cot_2000_val']
QUESTION_TEMPLATE = (
    "{Question}\n"
    "First, analyze the image and options step-by-step within <think> tags. "
    "Then, provide the correct answer letter with full option text in <answer> tags.\n"
    "Example format:\n"
    "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
)

def evaluate_yes_no(completions, solutions):
    """评估生成结果"""
    rewards = []
    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            gold_answer = re.search(r'<answer>(.*?)</answer>', sol.strip(), re.DOTALL)
            gold_answer = gold_answer.group(1).strip() if gold_answer else sol.strip()
            
            content_answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if not content_answer_match:
                raise ValueError("生成内容中缺失 <answer> 标签")
            student_answer = content_answer_match.group(1).strip()
            
            if student_answer == gold_answer:
                reward = 1.0
        except:
            reward = 0.0
        rewards.append(reward)
    
    accuracy = sum(rewards) / len(rewards) * 100 if len(rewards) > 0 else 0.0
    return rewards, accuracy

# 处理数据集
for ds in TEST_DATASETS:
    print(f"Processing dataset: {ds}...")
    ds_path = os.path.join(args.data_root, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.shuffle(data)
    data = data[:args.sample_num]
    
    all_outputs = []
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # 批量处理
    for i in tqdm(range(0, len(data), args.bsz), desc="Generating answers"):
        batch_data = data[i:i + args.bsz]
        batch_pixel_values = []
        batch_questions = []
        num_patches_list = []
        
        for x in batch_data:
            image_path = os.path.join(args.image_root, x['image'])
            question = '<image>\n' + QUESTION_TEMPLATE.format(Question=x['problem'])
            
            # 加载图像
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(f'cuda:{args.cuda}')
            batch_pixel_values.append(pixel_values)
            batch_questions.append(question)
            num_patches_list.append(pixel_values.size(0))
        
        # 合并 pixel_values
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        
        # 批量生成
        responses = model.batch_chat(
            tokenizer, 
            pixel_values,
            num_patches_list=num_patches_list,
            questions=batch_questions,
            generation_config=generation_config
        )
        
        all_outputs.extend(responses)
    
    # 评估
    solutions = [x['solution'] for x in data]
    rewards, acc = evaluate_yes_no(all_outputs, solutions)
    print(f"\nAccuracy of {ds}: {acc:.2f}%")
    
    # 保存结果
    final_output = []
    for input_example, model_output, r in zip(data, all_outputs, rewards):
        result = {
            'question': input_example['problem'],
            'ground_truth': input_example['solution'],
            'model_output': model_output,
            'correct': bool(r)
        }
        final_output.append(result)
    
    with open(args.output_path, "w") as f:
        json.dump({
            'accuracy': acc,
            'results': final_output
        }, f, indent=2)
    
    print(f"Results saved to {args.output_path}")
