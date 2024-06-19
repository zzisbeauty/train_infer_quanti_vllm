import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (ModelType, get_vllm_engine, get_default_template_type, get_template, inference_vllm)

# new glm3 model
ckpt_dir = '/home/train_infer_quanti_vllm/untrackfiles/output-new-glm3-with-data-trainconv240101-0529/chatglm3-6b-32k/v0-20240606-102752/checkpoint-1060-merged'
# old glm3 model
# ckpt_dir = '/home/train_infer_quanti_vllm/untrackfiles/output-old-glm3/chatglm3-6b-32k/v0-20240608-072736/checkpoint-1060-merged'
model_type = ModelType.chatglm3_6b_32k
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir)
tokenizer = llm_engine.hf_tokenizer
template = get_template(template_type, tokenizer)

import pickle

with open(os.path.join(
    os.getcwd(),'datasets/trainconv240101-0529/原始promp3塞入对话后再进行缩短后得到的指令数据，用于模型测试.list.pick'
    ),'rb') as f:
    testdata = pickle.load(f)


newmodelres = []
for query in testdata[40:500]:
    resp = inference_vllm(llm_engine, template, [{'query': query}])[0]
    print(f"response: {resp['response']}")
    newmodelres.append(resp['response'])
    print()
    print('*'*50)
    print()
    # print(f"history: {resp['history']}")
    # a = 1

with open('new-model-res-index-40-500-vllm.pick','wb') as f:
    pickle.dump(newmodelres,f)