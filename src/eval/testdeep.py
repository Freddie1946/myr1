from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
import torch

# 初始化
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path='/home/wjy/papers/VLM-R1-main/llava1',
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b'
)

# 直接推理
def simple_inference(image_path, question):
    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    
    # 准备文本输入
    if DEFAULT_IMAGE_TOKEN not in question:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
    
    # Tokenize
    input_ids = tokenizer_image_token(
        question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_tensor[0].shape[-2:]],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
    
    # 解码
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 去掉输入问题，只保留回答
    answer = response[len(question):].strip()
    return answer

# 使用
result = simple_inference("test_image.jpg", "What's in this image?")
print(result)