import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from swift.llm import (get_model_tokenizer, get_template, inference, ModelType, get_default_template_type)

# new-model-path
ckpt_dir = '/home/train_infer_quanti_vllm/untrackfiles/output-new-glm3-with-data-trainconv240101-0529/chatglm3-6b-32k/v0-20240606-102752/checkpoint-1060-merged'
model_type = ModelType.chatglm3_6b_32k
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, model_id_or_path=ckpt_dir)

template = get_template(template_type, tokenizer)


with open(os.path.join(
    os.getcwd(),'datasets/trainconv240101-0529/原始promp3塞入对话后再进行缩短后得到的指令数据，用于模型测试.list.pick'
    ),'rb') as f:
    testdata = pickle.load(f)

newmodelres = []
for query in testdata[40:500]:
    response, history = inference(model, template, query)
    print(f'response: {response}')
    newmodelres.append(response)
    print()
    print('*'*50)
    print()
    # print(f'history: {history}')
    a = 1

with open('new-model-res-index-40-500-usual.pick','wb') as f:
    pickle.dump(newmodelres,f)