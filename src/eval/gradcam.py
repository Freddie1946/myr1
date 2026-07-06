import os
from contextlib import nullcontext
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _pick_last_block(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    """尝试在常见属性中找到最后一个视觉 block/层。"""

    if module is None:
        return None

    # torch.nn.ModuleList / list / tuple 直接取最后一个并递归
    if isinstance(module, (torch.nn.ModuleList, list, tuple)):
        if len(module) == 0:
            return None
        return _pick_last_block(module[-1])

    for attr in ("layers", "blocks", "h", "stages"):
        layers = getattr(module, attr, None)
        if isinstance(layers, (list, tuple, torch.nn.ModuleList)) and len(layers) > 0:
            return layers[-1]
        if hasattr(layers, "__len__") and len(layers) > 0:
            try:
                return layers[-1]
            except Exception:  # pragma: no cover
                pass

    if hasattr(module, "encoder"):
        enc_last = _pick_last_block(module.encoder)
        if enc_last is not None:
            return enc_last

    if hasattr(module, "conv"):
        return module.conv

    if hasattr(module, "linear"):
        return module.linear

    return None


def _find_vision_block(model: torch.nn.Module) -> torch.nn.Module:
    """定位视觉编码器的最后一层 block，用于注册 Hook。"""

    candidates = [
        getattr(model, "vision_model", None),
        getattr(model, "vision_tower", None),
        getattr(model, "visual_encoder", None),
        getattr(model, "vision_layers", None),
        getattr(model, "visual", None),
    ]

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        candidates.extend(
            [
                getattr(inner_model, "vision_model", None),
                getattr(inner_model, "vision_tower", None),
                getattr(inner_model, "visual_encoder", None),
                getattr(inner_model, "visual", None),
            ]
        )

    for cand in candidates:
        block = _pick_last_block(cand)
        if block is not None:
            return block

    # 兜底：遍历 named_modules
    for name, module in model.named_modules():
        if "vision" in name.lower():
            block = _pick_last_block(module)
            if block is not None:
                return block

    vision_names = [name for name, _ in model.named_modules() if "vision" in name.lower()]
    raise RuntimeError(
        "未找到可用于 Grad-CAM 的视觉模块, 可用名称: "
        + (", ".join(vision_names[:10]) if vision_names else "<无>")
    )

def _register_hooks(target_module, activations, gradients):
    def forward_hook(module, inp, out):
        activations['value'] = out.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    handle_f = target_module.register_forward_hook(forward_hook)
    handle_b = target_module.register_full_backward_hook(backward_hook)
    return handle_f, handle_b

def _compute_gradcam(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    # acts: (B, C, H, W) or (B, Npatch, C)
    # grads: same shape
    if acts.dim() == 4:
        # CNN: (B, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze(1)
    elif acts.dim() == 3:
        # ViT: (B, Npatch, C) -> reshape为(B, H, W, C)
        B, N, C = acts.shape
        H = W = int(N ** 0.5)
        acts = acts[:, :H*W, :].reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        grads = grads[:, :H*W, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1)
    else:
        raise ValueError('未知特征张量形状')
    # 归一化到[0,1]
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return cam.detach().cpu().numpy()

def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    # img: PIL.Image, cam: (H, W) numpy
    cam = np.uint8(255 * cam)
    cam_img = Image.fromarray(cam).resize(img.size, resample=Image.BILINEAR)
    cam_img = np.array(cam_img)
    cmap = plt.get_cmap('jet')
    heatmap = np.uint8(cmap(cam_img / 255.0) * 255)
    heatmap = Image.fromarray(heatmap).convert('RGBA')
    img = img.convert('RGBA')
    overlay = Image.blend(img, heatmap, alpha)
    return overlay

def _prepare_inputs(
    processor,
    chat_text: str,
    image: Image.Image,
    device: str,
    image_inputs: Optional[Tuple] = None,
    video_inputs: Optional[Tuple] = None,
):
    if image_inputs is None or video_inputs is None:
        inputs = processor(text=[chat_text], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


def generate_and_save_gradcam(
    model,
    processor,
    image_path: str,
    chat_text: str,
    save_path: str,
    device: str = "cuda:0",
    image_inputs: Optional[Tuple] = None,
    video_inputs: Optional[Tuple] = None,
    alpha: float = 0.45,
):
    model.eval()
    # 1. 处理输入
    image = Image.open(image_path).convert('RGB')
    inputs = _prepare_inputs(processor, chat_text, image, device, image_inputs, video_inputs)
    # 3. 注册hook
    acts, grads = {}, {}
    target_module = _find_vision_block(model)
    handle_f, handle_b = _register_hooks(target_module, acts, grads)
    # 4. 前向生成logits
    amp_context = nullcontext()
    if device.startswith("cuda"):
        model_dtype = next(model.parameters()).dtype
        amp_context = torch.cuda.amp.autocast(dtype=model_dtype)

    with torch.enable_grad():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.requires_grad_(True)
        with amp_context:
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, V)
            target_logit = logits[:, -1, :].max()
        model.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=False)
    # 5. 计算Grad-CAM
    if "value" not in acts or "value" not in grads:
        handle_f.remove()
        handle_b.remove()
        raise RuntimeError("未捕获到视觉特征或梯度，无法生成 Grad-CAM")
    cam = _compute_gradcam(acts['value'], grads['value'])[0]  # (H, W)
    # 6. 叠加可视化
    overlay = overlay_cam_on_image(image, cam, alpha=alpha)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    overlay.save(save_path)
    # 7. 清理hook
    handle_f.remove()
    handle_b.remove()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return save_path
