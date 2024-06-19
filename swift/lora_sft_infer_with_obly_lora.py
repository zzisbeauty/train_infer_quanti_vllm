import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

ckpt_dir = 'vx-xxx/checkpoint-100'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')