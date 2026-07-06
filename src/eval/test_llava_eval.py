import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model on PathMMU datasets")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_datasets", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=5)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def evaluate_yes_no(completions: List[str], solutions: List[str], debug: bool = False):
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(completions, solutions):
        reward = 0.0
        try:
            gold_answer_match = re.search(r"<answer>(.*?)</answer>", sol.strip(), re.DOTALL)
            gold_answer = gold_answer_match.group(1).strip() if gold_answer_match else sol.strip()

            content_answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if not content_answer_match:
                raise ValueError("Missing <answer> tag in model output")
            student_answer = content_answer_match.group(1).strip()

            if student_answer == gold_answer:
                reward = 1.0
        except Exception as exc:  # pylint: disable=broad-except
            if debug:
                log_path = os.getenv("LOG_PATH", "debug_log.txt")
                with open(log_path, "a", encoding="utf-8") as fout:
                    fout.write(f"------------- {current_time} ERROR: {exc} -------------\n")
                    fout.write(f"Model output: {content}\n")
                    fout.write(f"Reference: {sol}\n")
            reward = 0.0

        rewards.append(reward)

        if debug:
            log_path = os.getenv("LOG_PATH", "debug_log.txt")
            with open(log_path, "a", encoding="utf-8") as fout:
                fout.write(f"------------- {current_time} Reward: {reward} -------------\n")
                fout.write(f"Model output: {content}\n")
                fout.write(f"Reference: {sol}\n")

    accuracy = sum(rewards) / len(rewards) * 100 if rewards else 0.0
    return rewards, accuracy


def main() -> None:
    args = parse_args()

    random.seed(42)
    torch.cuda.set_device(args.cuda)
    disable_torch_init()

    target_device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=target_device,
        device_map=target_device if target_device != "cuda" else "auto",
    )

    lower_name = model_name.lower()
    if "llama-2" in lower_name:
        conv_mode = "llava_llama_2"
    elif "mistral" in lower_name:
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in lower_name:
        conv_mode = "chatml_direct"
    elif "v1" in lower_name:
        conv_mode = "llava_v1"
    elif "mpt" in lower_name:
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv_mode = "mistral_instruct"

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    test_datasets = args.test_datasets.split(",")

    for ds in test_datasets:
        print(f"Processing dataset: {ds} ...")
        ds_path = os.path.join(args.data_root, f"{ds}.json")
        with open(ds_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        random.shuffle(data)
        data = data[: args.sample_num]

        question_template = (
            "{Question}\n"
            "Provide the correct answer letter.\n"
            "Example format:\n"
            "A"
        )

        # question_template = (
        #     "{Question}\n"
        #     "First, analyze the image and options step-by-step within <think> tags. "
        #     "Then, provide the correct answer letter with full option text in <answer> tags.\n"
        #     "Example format:\n"
        #     "<think>Your analytical reasoning here...</think><answer>Full text of the correct option</answer>"
        # )

        samples = []
        for item in data:
            question_text = question_template.format(Question=item["problem"])
            image_path = os.path.join(args.image_root, item["image"])
            samples.append({"question_text": question_text, "image_path": image_path})

        all_outputs: List[str] = []
        batch_size = args.bsz

        for start in tqdm(range(0, len(samples), batch_size), desc="Generating answers"):
            batch = samples[start : start + batch_size]

            prompts = []
            input_ids_list = []
            images = []
            image_sizes = []

            for sample in batch:
                qs = sample["question_text"]
                if IMAGE_PLACEHOLDER in qs:
                    if model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    prefix = image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
                    qs = prefix + "\n" + qs

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)

                ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).squeeze(0)
                input_ids_list.append(ids)

                image = Image.open(sample["image_path"]).convert("RGB")
                images.append(image)
                image_sizes.append(image.size)

            padded = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=pad_token_id
            ).to(model.device)
            attention_mask = (padded != pad_token_id).long().to(model.device)

            image_tensors = process_images(images, image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=padded,
                    attention_mask=attention_mask,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            decoded = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            cleaned = []
            for text, prompt in zip(decoded, prompts):
                cleaned.append(text[len(prompt) :].strip() if text.startswith(prompt) else text.strip())
            all_outputs.extend(cleaned)

        solutions = [item["solution"] for item in data]
        rewards, acc = evaluate_yes_no(all_outputs, solutions, debug=True)
        print(f"\nAccuracy of {ds}: {acc:.2f}%")

        final_output = []
        for item, model_output, reward in zip(data, all_outputs, rewards):
            final_output.append(
                {
                    "question": item["problem"],
                    "ground_truth": item["solution"],
                    "model_output": model_output,
                    "correct": bool(reward),
                }
            )

        output_path = args.output_path.format(DATASET=ds, STEPS=args.steps)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump({"accuracy": acc, "results": final_output}, fout, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
        print("-" * 100)


if __name__ == "__main__":
    main()
