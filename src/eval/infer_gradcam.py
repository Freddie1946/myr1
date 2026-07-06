import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from gradcam import generate_and_save_gradcam


DEFAULT_TEMPLATE = (
    "{Question}\n"
    "First, analyze the image within <think> tags. "
    "Then, provide the correct answer letter with full option text in <answer> tags.\n"
    "Example: <think>Your reasoning here...</think><answer>Full text of the correct option</answer>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and produce Grad-CAM heatmaps")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the pretrained model")
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing dataset json files")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset json filename (without .json)")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory storing images")
    parser.add_argument("--output_dir", type=str, default="gradcam_outputs", help="Directory to store outputs")
    parser.add_argument("--sample_num", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument(
        "--gradcam_device",
        type=str,
        default=None,
        help="Device for Grad-CAM computation (defaults to same as --device)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_root: str, dataset: str) -> List[Dict[str, Any]]:
    ds_path = os.path.join(data_root, f"{dataset}.json")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset json not found: {ds_path}")
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_message(image_path: str, question: str, template: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": template.format(Question=question)},
            ],
        }
    ]


def select_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    return torch.bfloat16


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = select_dtype(args.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=args.device,
        trust_remote_code=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    data = load_dataset(args.data_root, args.dataset)
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    random.shuffle(data)

    results = []
    inference_device = args.device
    grad_device = args.gradcam_device or args.device
    if grad_device.startswith("cpu") and getattr(model.config, "attn_implementation", "") == "flash_attention_2":
        raise ValueError(
            "Grad-CAM device set to CPU but model uses flash_attention_2 kernels which require CUDA. "
            "Please rerun with --gradcam_device set to a CUDA device (e.g., cuda:0) or reload the model with"
            " --dtype float32 and attn_implementation='eager'."
        )

    processed = 0
    for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        if processed >= args.sample_num:
            break
        image_path = os.path.join(args.image_root, sample["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        message = build_message(image_path, sample["problem"], DEFAULT_TEMPLATE)
        chat_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(inference_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        # 去除输入部分 token
        gen_only = output_ids[0][len(inputs["input_ids"][0]) :]
        output_text = processor.decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if inference_device.startswith("cuda"):
            del output_ids
            torch.cuda.empty_cache()

        del inputs, gen_only, image_inputs, video_inputs

        result_entry = {
            "id": sample.get("id", processed),
            "question": sample.get("problem"),
            "ground_truth": sample.get("solution"),
            "model_output": output_text,
        }

        if grad_device != inference_device:
            model.to(grad_device)
            if inference_device.startswith("cuda"):
                torch.cuda.empty_cache()

        heatmap_path = os.path.join(
            args.output_dir,
            f"gradcam_{args.dataset}_{processed:03d}.png",
        )
        generate_and_save_gradcam(
            model=model,
            processor=processor,
            image_path=image_path,
            chat_text=chat_text,
            save_path=heatmap_path,
            device=grad_device,
        )
        result_entry["heatmap_path"] = heatmap_path

        if grad_device != inference_device:
            model.to(inference_device)
            if grad_device.startswith("cuda"):
                torch.cuda.empty_cache()

        if inference_device.startswith("cuda"):
            torch.cuda.empty_cache()

    results.append(result_entry)
    processed += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(args.output_dir, f"gradcam_results_{timestamp}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} samples to {output_json}")


if __name__ == "__main__":
    main()
