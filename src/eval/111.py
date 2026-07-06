from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# llava-1.5预训练权重
model_path = "/home/wjy/papers/VLM-R1-main/llava1"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
print(get_model_name_from_path(model_path))
# 文本提示
prompt = "what is in this image?"
# 测试图片路径
image_file = "/home/wjy/PathMMU/images/0a0b54f0bdb5f74b05f0d613beb1aa77534cad5f5e20f400aedabb8cbd323b2f.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)