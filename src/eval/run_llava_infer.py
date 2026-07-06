import argparse
import re
from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image

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


def parse_image_list(image_file: str, sep: str) -> List[str]:
    return image_file.split(sep)


def load_image(path: str) -> Image.Image:
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(path).convert("RGB")


def load_images(paths: List[str]) -> List[Image.Image]:
    return [load_image(p) for p in paths]


def build_prompt(raw_prompt: str, model_cfg) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    prompt = raw_prompt

    if IMAGE_PLACEHOLDER in prompt:
        replacement = image_token_se if model_cfg.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        prompt = re.sub(IMAGE_PLACEHOLDER, replacement, prompt)
    else:
        prefix = image_token_se if model_cfg.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        prompt = prefix + "\n" + prompt
    return prompt


def infer(args: argparse.Namespace) -> None:
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device=args.device,
        device_map=args.device if args.device != "cuda" else "auto",
    )

    prompt = build_prompt(args.query, model.config)

    # Conversation template selection
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
    if args.conv_mode is not None and args.conv_mode != conv_mode:
        print(
            f"[WARNING] inferred conv_mode={conv_mode}, but overriding with args.conv_mode={args.conv_mode}"
        )
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    image_paths = parse_image_list(args.image_file, args.sep)
    images = load_images(image_paths)
    image_sizes = [img.size for img in images]
    image_tensors = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    output_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    print(output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaVA inference with official loader")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    infer(args)


if __name__ == "__main__":
    main()
