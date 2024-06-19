# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

import torch

from swift.llm import (DatasetName, InferArguments, ModelType, SftArguments,infer_main, sft_main, app_ui_main)


sft_args = SftArguments(
    # 模型
    model_type=ModelType.chatglm3_6b_32k,
    model_id_or_path='/home/train_infer_quanti_vllm/untrackfiles/old-model-chatglm3-6b-32k-base',

    # dataset=[f'{DatasetName.blossom_math_zh}#2000'],  # 注册数据集
    # dataset=['/home/train_infer_quanti_vllm/datasets/trainconv240101-0529/train-data-base-p3-no-url.jsonl'],  # 本地数据集
    dataset='/home/train_infer_quanti_vllm/datasets/tranconv240610/train-data-base-p3-no-url-quli.jsonl',
    max_length=10000,
    # truncation_strategy='delete',

    # 训练参数
    # dtype='bf16',
    # save_only_model=True,
    # deepspeed='/home/train_infer_quanti_vllm/swift/llm/ds_config/zero3_offload.json',

    num_train_epochs=5,
    gradient_checkpointing=True,
    sft_type='lora', # 设定 lora 后，关于 lora 的其它参数保持默认
    learning_rate=1e-4,
    tuner_backend='peft',
    output_dir='/home/train_infer_quanti_vllm/untrackfiles/output-old-model-glm3-with-data-24016',
)
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()
# 训练完毕zm