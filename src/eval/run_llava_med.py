import argparse
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


@dataclass
class ConversationTurn:
    role: str
    content: List[dict]


def build_conversation(question: str) -> List[dict]:
    """Create a multimodal conversation template accepted by the processor."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful medical vision assistant."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "text": None,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaVA-Med inference with Transformers API")
    parser.add_argument(
        "--model-id",
        default="llava-hf/llava-med-v1.5-mistral-7b",
        help="Hugging Face model identifier for LLaVA-Med.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. 'cuda', 'cuda:1', 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()

    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    image = Image.open(args.image).convert("RGB")
    conversation = build_conversation(args.question)
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # Remove the prompt portion to keep only the assistant answer.
    answer = generated_text[len(prompt):].strip()
    print("=== LLaVA-Med Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
