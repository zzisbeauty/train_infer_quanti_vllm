import os, sys

lastpath = os.path.abspath('..')  
nowPath = os.getcwd()
os.path.join(nowPath)

from publicfunctions import *

logs_dir = os.path.join(nowPath,'untrackfiles/save_logs')
logger_init(log_file_name='monitor', log_level=logging.INFO, log_dir=logs_dir, only_file=False)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import torch
from swift.llm import (DatasetName, InferArguments, ModelType, SftArguments, infer_main, sft_main, app_ui_main)

output_dir = os.path.join(nowPath,'sourcecodesft_llama3_8b_instruct/output')


# lora_target_modules=['ALL', 'EMBEDDING'] # ['DEFAULT']
lora_target_modules=['DEFAULT'] # ['DEFAULT']


# run time setting
max_steps = -1  # 训练的 max_steps 数, 默认为-1. 如果 max_steps >= 0, 则覆盖 num_train_epochs
if max_steps <= 0: # 不使用最大步数训练。而是使用轮次
    num_train_epochs = 10
else: # 使用
    num_train_epochs = -1


# 确定评估时的 batch_size, 默认为 None。当 predict_with_generate 为 True 时, 设置为 1, 为 False 时, 设置为 batch_size
predict_with_generate = True 
if predict_with_generate:
    eval_batch_size = 1
    max_new_tokens = 5000  # 当 predict_with_generate = True 时生效 
else:
    eval_batch_size = 10

# model_type = ModelType.qwen_7b_chat # 注册模型
model_type = ModelType.llama3_8b_instruct

sft_args = SftArguments(
    # base model params
    sft_type='lora',
    model_type=model_type,
    model_id_or_path=os.path.join(nowPath,'untrackfiles/Meta-Llama-3-8B-Instruct'),
    output_dir=output_dir,

    # base data params
    batch_size=1,
    max_length=1024,
    max_new_tokens=1024,
    eval_batch_size=eval_batch_size,
    train_dataset_sample=2000, # -1 all data ，不对数据进行采样
    # 已经注册的数据集
    dataset=[],
    dataset=[DatasetName.blossom_math_zh],
    # 自定义数据集
    custom_train_dataset_path = os.path.join(nowPath, 'datasets/giikin_jsonl_data_demo_standard_data/chatgml.jsonl'),

    # run time params
    max_steps=max_steps,
    num_train_epochs=num_train_epochs,
    
    # 验证过程控制，由上诉  predict_with_generate 控制
    predict_with_generate=predict_with_generate,
    eval_batch_size=eval_batch_size,
    max_new_tokens=max_new_tokens
)


result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
logging.info(f'============= best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()