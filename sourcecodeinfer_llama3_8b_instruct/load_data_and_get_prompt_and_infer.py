import sys, os

proPath = os.getcwd()
sys.path.append('/')
sys.path.append(proPath)

from publicfunctions import *



# load yl
ylDataPath = os.path.join(proPath,'datasets/giikin_this_task_data-v1/read_and_clean_yl_trans_data/lang_type_trans_data_ori_tar_more_clean.pick')
with open(ylDataPath,'rb') as f:
    ylData = pickle.load(f)

# ###########################################################


# from swift.llm import (get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,)
# from swift.utils import seed_everything

# model_type = ModelType.qwen_7b_chat
# # model_type = "llama3-8b-instruct"
# template_type = get_default_template_type(model_type)

# kwargs = {}
# # kwargs['use_flash_attn'] = True   # 使用 flash_attn
# model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# model.generation_config.max_new_tokens = 128  # 修改 max_new_tokens

# template = get_template(template_type, tokenizer)
# seed_everything(42)
# query = '浙江的省会在哪里？'
# response, history = inference(model, template, query)
# print(f'query: {query}')
# print(f'response: {response}')
# query = '这有什么好吃的？'
# response, history = inference(model, template, query, history)
# print(f'query: {query}')
# print(f'response: {response}')
# print(f'history: {history}')

# """Out[0]
# query: 浙江的省会在哪里？
# response: 浙江省的省会是杭州。
# query: 这有什么好吃的？
# response: 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。
# history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。')]
# """

# # 流式输出对话模板
# inference(model, template, '第一个问题是什么', history, verbose=True, stream=True)
# """Out[1]
# [PROMPT]<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# 浙江的省会在哪里？<|im_end|>
# <|im_start|>assistant
# 浙江省的省会是杭州。<|im_end|>
# <|im_start|>user
# 这有什么好吃的？<|im_end|>
# <|im_start|>assistant
# 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。<|im_end|>
# <|im_start|>user
# 第一个问题是什么<|im_end|>
# <|im_start|>assistant
# [OUTPUT]你的第一个问题是“浙江的省会在哪里？”<|im_end|>
# """



from swift.llm import (ModelType, get_vllm_engine, get_default_template_type,get_template, inference_vllm)

# model_type = ModelType.qwen_7b_chat
# model_type = ModelType.llama3_8b_instruct
# llm_engine = get_vllm_engine(model_type)


llm_engine = get_vllm_engine(
    model_type='llama3-8b-instruct',
    model_id_or_path=os.path.join(proPath, 'untrackfiles/Meta-Llama-3-8B-Instruct'),
    max_model_len=1024
)


template_type = get_default_template_type(model_type='llama3-8b-instruct')
template = get_template(template_type, llm_engine.hf_tokenizer)

# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 1024

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
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

"""Out[0]
query: 你好!
response: 你好！很高兴为你服务。有什么我可以帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省会是杭州市。
query: 这有什么好吃的
response: 杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。
history: [('浙江的省会在哪？', '浙江省会是杭州市。'), ('这有什么好吃的', '杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。')]
"""