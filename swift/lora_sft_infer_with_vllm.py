import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (ModelType, get_vllm_engine, get_default_template_type, get_template, inference_vllm)

model_type = ModelType.chatglm3_6b
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)

# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 3000

import pickle

with open(os.path.join(
    os.getcwd(),'datasets/trainconv240101-0529/原始promp3塞入对话后再进行缩短后得到的指令数据，用于模型测试.list.pick'),'rb') as f:
    testdata = pickle.load(f)

for i in testdata[:20]:
    request_list = [{'query': i}]
    resp_list = inference_vllm(llm_engine, template, request_list)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print()
        print('*'*50)
        print()
        print(f"response: {resp['response']}")
        a = 1



# 多轮对话下使用
"""
request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}] # 你好之后会等待模型答复；模型答复之后才会进行浙江省会的继续询问
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")
"""