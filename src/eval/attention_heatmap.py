import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


@dataclass
class HeatmapTile:
    layer_idx: int
    image_idx: int
    heatmap: np.ndarray  # (H, W)
    score: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.heatmap.shape  # pragma: no cover - trivial accessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize text-to-vision attention heatmaps for Qwen2.5-VL"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="attn_heatmaps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--sample_num", type=int, default=3)
    parser.add_argument(
        "--template",
        type=str,
        default=(
            "{Question}\n"
            "Think carefully about the image inside <think> tags, then provide the final answer"
            " wrapped in <answer> tags."
        ),
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=8,
        help="Number of transformer layers (from the top) to visualize. Use -1 for all layers.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Opacity for overlaying heatmaps onto images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample shuffling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate when capturing model output.",
    )
    return parser.parse_args()


def select_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_samples(data_root: str, dataset: str) -> List[Dict[str, str]]:
    ds_path = Path(data_root) / f"{dataset}.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    with ds_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset should be a list of samples, got {type(data)}")
    return data


def build_message(image_path: str, question: str, template: str) -> List[Dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{image_path}"},
                {"type": "text", "text": template.format(Question=question)},
            ],
        }
    ]


def extract_image_segments(
    input_ids: torch.LongTensor,
    vision_start_id: int,
    vision_end_id: int,
) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    seq = input_ids.tolist()
    for idx, token in enumerate(seq):
        if token == vision_start_id:
            start = idx + 1
        elif token == vision_end_id and start is not None:
            segments.append((start, idx))  # end exclusive
            start = None
    return segments


def build_text_mask(
    input_ids: torch.LongTensor,
    segments: Sequence[Tuple[int, int]],
    special_token_ids: Sequence[int],
) -> torch.BoolTensor:
    device = input_ids.device
    seq_len = input_ids.size(0)
    mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    for start, end in segments:
        mask[start:end] = False
    if special_token_ids:
        special_tensor = torch.tensor(list(special_token_ids), device=device)
        mask &= ~torch.isin(input_ids, special_tensor)
    if not mask.any():
        mask[-1] = True
    return mask


def infer_grid_shape(seg_len: int, hint: Optional[Sequence[int]]) -> Tuple[int, int]:
    if hint and len(hint) == 3:
        total = int(hint[1]) * int(hint[2])
        if total >= seg_len and seg_len > 0:
            scale = math.sqrt(seg_len / max(total, 1))
            h = max(1, int(round(int(hint[1]) * scale)))
            w = max(1, int(math.ceil(seg_len / h)))
            if h * w >= seg_len:
                return h, w
    if seg_len <= 0:
        return 1, 1
    h = int(math.sqrt(seg_len)) or 1
    w = int(math.ceil(seg_len / h))
    return h, w


def compute_heatmaps(
    attentions: Sequence[torch.Tensor],
    input_ids: torch.LongTensor,
    segments: Sequence[Tuple[int, int]],
    special_token_ids: Sequence[int],
    grid_hints: Optional[Sequence[Sequence[int]]],
    max_layers: int,
) -> List[HeatmapTile]:
    if not segments:
        return []

    seq_len = input_ids.size(0)
    text_mask = build_text_mask(input_ids, segments, special_token_ids)
    text_positions = text_mask.nonzero(as_tuple=False).squeeze(-1)
    if text_positions.numel() == 0:
        text_positions = torch.tensor([seq_len - 1], device=input_ids.device)

    tiles: List[HeatmapTile] = []
    total_layers = len(attentions)
    layer_indices = range(total_layers)
    if max_layers > 0:
        start = max(0, total_layers - max_layers)
        layer_indices = range(start, total_layers)

    for layer_idx in layer_indices:
        attn = attentions[layer_idx]  # [B, heads, S, S]
        attn = attn.mean(dim=1, keepdim=False)  # [B, S, S]
        attn_vec = attn[0, text_positions, :].mean(dim=0)  # [S]

        for img_idx, (start, end) in enumerate(segments):
            vec = attn_vec[start:end]
            if vec.numel() == 0:
                continue
            vec = vec - vec.min()
            denom = vec.max().clamp(min=1e-6)
            vec = vec / denom

            grid_hint = None
            if grid_hints and img_idx < len(grid_hints):
                grid_hint = grid_hints[img_idx]
            h, w = infer_grid_shape(vec.numel(), grid_hint)
            padded = torch.zeros(h * w, dtype=vec.dtype, device=vec.device)
            padded[: vec.numel()] = vec
            heatmap = padded.view(h, w).detach().cpu().numpy()
            score = float(vec.mean().item())
            tiles.append(HeatmapTile(layer_idx=layer_idx, image_idx=img_idx, heatmap=heatmap, score=score))
    return tiles


def overlay_heatmap(
    image_path: str,
    heatmap: np.ndarray,
    out_path: Path,
    alpha: float = 0.45,
    cmap_name: str = "jet",
) -> None:
    image = Image.open(image_path).convert("RGB")
    heat = heatmap.astype(np.float32)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_img = Image.fromarray(np.uint8(heat * 255))
    heat_img = heat_img.resize(image.size, resample=Image.BILINEAR)
    colormap = cm.get_cmap(cmap_name)
    colored = colormap(np.array(heat_img) / 255.0)[..., :3]
    overlay = (1.0 - alpha) * np.asarray(image) / 255.0 + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


def visualize_sample(
    model,
    processor,
    sample: Dict[str, str],
    sample_idx: int,
    template: str,
    image_root: str,
    device: str,
    alpha: float,
    max_layers: int,
    max_new_tokens: int,
    out_dir: Path,
) -> Dict[str, object]:
    image_path = sample.get("image")
    if not image_path:
        raise ValueError("Sample missing 'image' field")
    if not os.path.isabs(image_path):
        image_path = os.path.join(image_root, image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    question = sample.get("problem") or sample.get("question")
    if not question:
        raise ValueError("Sample missing problem/question text")

    message = build_message(image_path, question, template)
    chat_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)
    model_inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in model_inputs.items()}

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    input_len = model_inputs["input_ids"].shape[-1]
    gen_tokens = generation[:, input_len:]
    generated_text = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
    del generation

    special_tokens = {
        processor.tokenizer.pad_token_id,
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|im_start|>"),
        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        processor.tokenizer.convert_tokens_to_ids("<|vision_pad|>"),
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    }
    special_tokens = {tok for tok in special_tokens if tok is not None}

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if vision_start_id is None or vision_end_id is None:
        raise ValueError("Tokenizer does not define <|vision_start|> / <|vision_end|> tokens")

    segments = extract_image_segments(model_inputs["input_ids"][0], vision_start_id, vision_end_id)
    if not segments:
        raise RuntimeError("Unable to locate image token segments in input_ids")

    grid_hints = None
    if "image_grid_thw" in model_inputs:
        grid_hints = model_inputs["image_grid_thw"].tolist()

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = outputs.attentions
    del outputs
    heatmaps = compute_heatmaps(
        attentions=attentions,
        input_ids=model_inputs["input_ids"][0],
        segments=segments,
        special_token_ids=list(special_tokens),
        grid_hints=grid_hints,
        max_layers=max_layers,
    )

    result_entry: Dict[str, object] = {
        "question": question,
        "image_path": image_path,
        "model_output": generated_text,
        "heatmaps": [],
    }

    sample_tag = sample.get("id")
    if sample_tag is None:
        sample_tag = f"{sample_idx:04d}"
    else:
        sample_tag = str(sample_tag)

    for tile in heatmaps:
        out_path = out_dir / f"sample_{sample_tag}_{tile.layer_idx:02d}_{tile.image_idx:02d}.png"
        overlay_heatmap(image_path, tile.heatmap, out_path, alpha=alpha)
        result_entry["heatmaps"].append(
            {
                "layer": tile.layer_idx,
                "image_index": tile.image_idx,
                "score": tile.score,
                "path": str(out_path),
            }
        )

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result_entry


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)

    dtype = select_dtype(args.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    data = load_samples(args.data_root, args.dataset)
    np.random.shuffle(data)
    data = data[: args.sample_num]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    processed = 0
    for idx, sample in enumerate(tqdm(data, desc="Visualizing attention")):
        if processed >= args.sample_num:
            break
        try:
            entry = visualize_sample(
                model=model,
                processor=processor,
                sample=sample,
                sample_idx=idx,
                template=args.template,
                image_root=args.image_root,
                device=args.device,
                alpha=args.alpha,
                max_layers=args.max_layers,
                max_new_tokens=args.max_new_tokens,
                out_dir=out_dir,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"[WARN] Skipping sample due to error: {exc}")
            continue
        results.append(entry)
        processed += 1

    summary_path = out_dir / "attention_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples to {summary_path}")

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
