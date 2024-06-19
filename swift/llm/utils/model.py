# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import math
import os
import sys
from contextlib import nullcontext
from functools import partial, update_wrapper, wraps
from types import MethodType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from modelscope import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                        GenerationConfig, GPTQConfig, snapshot_download)
from packaging import version
from torch import Tensor
from torch import dtype as Dtype
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import strtobool
from transformers.utils.versions import require_version

from swift import get_logger
from swift.hub.utils.utils import get_cache_dir
from swift.utils import get_dist_setting, safe_ddp_context, subprocess_run, use_torchacc
from .template import TemplateType
from .utils import get_max_model_len, is_unsloth_available

logger = get_logger()

# Model Home: 'https://modelscope.cn/models/{model_id_or_path}/summary'
MODEL_MAPPING: Dict[str, Dict[str, Any]] = {}


class ModelType:
    # qwen
    qwen_1_8b = 'qwen-1_8b'
    qwen_1_8b_chat = 'qwen-1_8b-chat'
    qwen_1_8b_chat_int4 = 'qwen-1_8b-chat-int4'
    qwen_1_8b_chat_int8 = 'qwen-1_8b-chat-int8'
    qwen_7b = 'qwen-7b'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen_7b_chat_int4 = 'qwen-7b-chat-int4'
    qwen_7b_chat_int8 = 'qwen-7b-chat-int8'
    qwen_14b = 'qwen-14b'
    qwen_14b_chat = 'qwen-14b-chat'
    qwen_14b_chat_int4 = 'qwen-14b-chat-int4'
    qwen_14b_chat_int8 = 'qwen-14b-chat-int8'
    qwen_72b = 'qwen-72b'
    qwen_72b_chat = 'qwen-72b-chat'
    qwen_72b_chat_int4 = 'qwen-72b-chat-int4'
    qwen_72b_chat_int8 = 'qwen-72b-chat-int8'
    # modelscope_agent
    modelscope_agent_7b = 'modelscope-agent-7b'
    modelscope_agent_14b = 'modelscope-agent-14b'
    # qwen1.5
    qwen1half_0_5b = 'qwen1half-0_5b'
    qwen1half_1_8b = 'qwen1half-1_8b'
    qwen1half_4b = 'qwen1half-4b'
    qwen1half_7b = 'qwen1half-7b'
    qwen1half_14b = 'qwen1half-14b'
    qwen1half_32b = 'qwen1half-32b'
    qwen1half_72b = 'qwen1half-72b'
    qwen1half_110b = 'qwen1half-110b'
    codeqwen1half_7b = 'codeqwen1half-7b'
    qwen1half_moe_a2_7b = 'qwen1half-moe-a2_7b'
    qwen1half_0_5b_chat = 'qwen1half-0_5b-chat'
    qwen1half_1_8b_chat = 'qwen1half-1_8b-chat'
    qwen1half_4b_chat = 'qwen1half-4b-chat'
    qwen1half_7b_chat = 'qwen1half-7b-chat'
    qwen1half_14b_chat = 'qwen1half-14b-chat'
    qwen1half_32b_chat = 'qwen1half-32b-chat'
    qwen1half_72b_chat = 'qwen1half-72b-chat'
    qwen1half_110b_chat = 'qwen1half-110b-chat'
    qwen1half_moe_a2_7b_chat = 'qwen1half-moe-a2_7b-chat'
    codeqwen1half_7b_chat = 'codeqwen1half-7b-chat'

    # qwen1.5 gptq
    qwen1half_0_5b_chat_int4 = 'qwen1half-0_5b-chat-int4'
    qwen1half_1_8b_chat_int4 = 'qwen1half-1_8b-chat-int4'
    qwen1half_4b_chat_int4 = 'qwen1half-4b-chat-int4'
    qwen1half_7b_chat_int4 = 'qwen1half-7b-chat-int4'
    qwen1half_14b_chat_int4 = 'qwen1half-14b-chat-int4'
    qwen1half_32b_chat_int4 = 'qwen1half-32b-chat-int4'
    qwen1half_72b_chat_int4 = 'qwen1half-72b-chat-int4'
    qwen1half_110b_chat_int4 = 'qwen1half-110b-chat-int4'
    qwen1half_0_5b_chat_int8 = 'qwen1half-0_5b-chat-int8'
    qwen1half_1_8b_chat_int8 = 'qwen1half-1_8b-chat-int8'
    qwen1half_4b_chat_int8 = 'qwen1half-4b-chat-int8'
    qwen1half_7b_chat_int8 = 'qwen1half-7b-chat-int8'
    qwen1half_14b_chat_int8 = 'qwen1half-14b-chat-int8'
    qwen1half_72b_chat_int8 = 'qwen1half-72b-chat-int8'
    qwen1half_moe_a2_7b_chat_int4 = 'qwen1half-moe-a2_7b-chat-int4'

    # qwen1.5 awq
    qwen1half_0_5b_chat_awq = 'qwen1half-0_5b-chat-awq'
    qwen1half_1_8b_chat_awq = 'qwen1half-1_8b-chat-awq'
    qwen1half_4b_chat_awq = 'qwen1half-4b-chat-awq'
    qwen1half_7b_chat_awq = 'qwen1half-7b-chat-awq'
    qwen1half_14b_chat_awq = 'qwen1half-14b-chat-awq'
    qwen1half_32b_chat_awq = 'qwen1half-32b-chat-awq'
    qwen1half_72b_chat_awq = 'qwen1half-72b-chat-awq'
    qwen1half_110b_chat_awq = 'qwen1half-110b-chat-awq'
    codeqwen1half_7b_chat_awq = 'codeqwen1half-7b-chat-awq'

    # qwen2
    qwen2_0_5b = 'qwen2-0_5b'
    qwen2_0_5b_instruct = 'qwen2-0_5b-instruct'
    qwen2_0_5b_instruct_int4 = 'qwen2-0_5b-instruct-int4'
    qwen2_0_5b_instruct_int8 = 'qwen2-0_5b-instruct-int8'
    qwen2_0_5b_instruct_awq = 'qwen2-0_5b-instruct-awq'
    qwen2_1_5b = 'qwen2-1_5b'
    qwen2_1_5b_instruct = 'qwen2-1_5b-instruct'
    qwen2_1_5b_instruct_int4 = 'qwen2-1_5b-instruct-int4'
    qwen2_1_5b_instruct_int8 = 'qwen2-1_5b-instruct-int8'
    qwen2_1_5b_instruct_awq = 'qwen2-1_5b-instruct-awq'
    qwen2_7b = 'qwen2-7b'
    qwen2_7b_instruct = 'qwen2-7b-instruct'
    qwen2_7b_instruct_int4 = 'qwen2-7b-instruct-int4'
    qwen2_7b_instruct_int8 = 'qwen2-7b-instruct-int8'
    qwen2_7b_instruct_awq = 'qwen2-7b-instruct-awq'
    qwen2_72b = 'qwen2-72b'
    qwen2_72b_instruct = 'qwen2-72b-instruct'
    qwen2_72b_instruct_int4 = 'qwen2-72b-instruct-int4'
    qwen2_72b_instruct_int8 = 'qwen2-72b-instruct-int8'
    qwen2_72b_instruct_awq = 'qwen2-72b-instruct-awq'
    qwen2_57b_a14b = 'qwen2-57b-a14b'
    qwen2_57b_a14b_instruct = 'qwen2-57b-a14b-instruct'
    qwen2_57b_a14b_instruct_int4 = 'qwen2-57b-a14b-instruct-int4'

    # qwen-vl
    qwen_vl = 'qwen-vl'
    qwen_vl_chat = 'qwen-vl-chat'
    qwen_vl_chat_int4 = 'qwen-vl-chat-int4'
    # qwen-audio
    qwen_audio = 'qwen-audio'
    qwen_audio_chat = 'qwen-audio-chat'
    # chatglm
    chatglm2_6b = 'chatglm2-6b'
    chatglm2_6b_32k = 'chatglm2-6b-32k'
    chatglm3_6b_base = 'chatglm3-6b-base'
    chatglm3_6b = 'chatglm3-6b'
    chatglm3_6b_32k = 'chatglm3-6b-32k'
    chatglm3_6b_128k = 'chatglm3-6b-128k'
    codegeex2_6b = 'codegeex2-6b'
    glm4v_9b_chat = 'glm4v-9b-chat'
    glm4_9b = 'glm4-9b'
    glm4_9b_chat = 'glm4-9b-chat'
    glm4_9b_chat_1m = 'glm4-9b-chat-1m'
    # llama2
    llama2_7b = 'llama2-7b'
    llama2_7b_chat = 'llama2-7b-chat'
    llama2_13b = 'llama2-13b'
    llama2_13b_chat = 'llama2-13b-chat'
    llama2_70b = 'llama2-70b'
    llama2_70b_chat = 'llama2-70b-chat'
    llama2_7b_aqlm_2bit_1x16 = 'llama2-7b-aqlm-2bit-1x16'  # aqlm
    # llama3
    llama3_8b = 'llama3-8b'
    llama3_8b_instruct = 'llama3-8b-instruct'
    llama3_8b_instruct_int4 = 'llama3-8b-instruct-int4'
    llama3_8b_instruct_int8 = 'llama3-8b-instruct-int8'
    llama3_8b_instruct_awq = 'llama3-8b-instruct-awq'
    llama3_70b = 'llama3-70b'
    llama3_70b_instruct = 'llama3-70b-instruct'
    llama3_70b_instruct_int4 = 'llama3-70b-instruct-int4'
    llama3_70b_instruct_int8 = 'llama3-70b-instruct-int8'
    llama3_70b_instruct_awq = 'llama3-70b-instruct-awq'
    # chinese-llama-alpaca
    chinese_llama_2_1_3b = 'chinese-llama-2-1_3b'
    chinese_llama_2_7b = 'chinese-llama-2-7b'
    chinese_llama_2_7b_16k = 'chinese-llama-2-7b-16k'
    chinese_llama_2_7b_64k = 'chinese-llama-2-7b-64k'
    chinese_llama_2_13b = 'chinese-llama-2-13b'
    chinese_llama_2_13b_16k = 'chinese-llama-2-13b-16k'
    chinese_alpaca_2_1_3b = 'chinese-alpaca-2-1_3b'
    chinese_alpaca_2_7b = 'chinese-alpaca-2-7b'
    chinese_alpaca_2_7b_16k = 'chinese-alpaca-2-7b-16k'
    chinese_alpaca_2_7b_64k = 'chinese-alpaca-2-7b-64k'
    chinese_alpaca_2_13b = 'chinese-alpaca-2-13b'
    chinese_alpaca_2_13b_16k = 'chinese-alpaca-2-13b-16k'
    llama_3_chinese_8b = 'llama-3-chinese-8b'
    llama_3_chinese_8b_instruct = 'llama-3-chinese-8b-instruct'
    # atom
    atom_7b = 'atom-7b'
    atom_7b_chat = 'atom-7b-chat'
    # llava
    llava1_6_mistral_7b_instruct = 'llava1_6-mistral-7b-instruct'
    llava1_6_yi_34b_instruct = 'llava1_6-yi-34b-instruct'
    llama3_llava_next_8b = 'llama3-llava-next-8b'
    llava_next_72b = 'llava-next-72b'
    llava_next_110b = 'llava-next-110b'
    # yi
    yi_6b = 'yi-6b'
    yi_6b_200k = 'yi-6b-200k'
    yi_6b_chat = 'yi-6b-chat'
    yi_6b_chat_awq = 'yi-6b-chat-awq'
    yi_6b_chat_int8 = 'yi-6b-chat-int8'
    yi_9b = 'yi-9b'
    yi_9b_200k = 'yi-9b-200k'
    yi_34b = 'yi-34b'
    yi_34b_200k = 'yi-34b-200k'
    yi_34b_chat = 'yi-34b-chat'
    yi_34b_chat_awq = 'yi-34b-chat-awq'
    yi_34b_chat_int8 = 'yi-34b-chat-int8'
    # yi1.5
    yi_1_5_6b = 'yi-1_5-6b'
    yi_1_5_6b_chat = 'yi-1_5-6b-chat'
    yi_1_5_9b = 'yi-1_5-9b'
    yi_1_5_9b_chat = 'yi-1_5-9b-chat'
    yi_1_5_9b_chat_16k = 'yi-1_5-9b-chat-16k'
    yi_1_5_34b = 'yi-1_5-34b'
    yi_1_5_34b_chat = 'yi-1_5-34b-chat'
    yi_1_5_34b_chat_16k = 'yi-1_5-34b-chat-16k'
    yi_1_5_6b_chat_awq_int4 = 'yi-1_5-6b-chat-awq-int4'
    yi_1_5_6b_chat_gptq_int4 = 'yi-1_5-6b-chat-gptq-int4'
    yi_1_5_9b_chat_awq_int4 = 'yi-1_5-9b-chat-awq-int4'
    yi_1_5_9b_chat_gptq_int4 = 'yi-1_5-9b-chat-gptq-int4'
    yi_1_5_34b_chat_awq_int4 = 'yi-1_5-34b-chat-awq-int4'
    yi_1_5_34b_chat_gptq_int4 = 'yi-1_5-34b-chat-gptq-int4'
    # yi-vl
    yi_vl_6b_chat = 'yi-vl-6b-chat'
    yi_vl_34b_chat = 'yi-vl-34b-chat'
    # llava-llama
    llava_llama3_8b_v1_1 = 'llava-llama-3-8b-v1_1'
    # internlm
    internlm_7b = 'internlm-7b'
    internlm_7b_chat = 'internlm-7b-chat'
    internlm_7b_chat_8k = 'internlm-7b-chat-8k'
    internlm_20b = 'internlm-20b'
    internlm_20b_chat = 'internlm-20b-chat'
    # internlm2
    internlm2_1_8b = 'internlm2-1_8b'
    internlm2_1_8b_sft_chat = 'internlm2-1_8b-sft-chat'
    internlm2_1_8b_chat = 'internlm2-1_8b-chat'
    internlm2_7b_base = 'internlm2-7b-base'
    internlm2_7b = 'internlm2-7b'
    internlm2_7b_sft_chat = 'internlm2-7b-sft-chat'
    internlm2_7b_chat = 'internlm2-7b-chat'
    internlm2_20b_base = 'internlm2-20b-base'
    internlm2_20b = 'internlm2-20b'
    internlm2_20b_sft_chat = 'internlm2-20b-sft-chat'
    internlm2_20b_chat = 'internlm2-20b-chat'
    # internlm2-math
    internlm2_math_7b = 'internlm2-math-7b'
    internlm2_math_7b_chat = 'internlm2-math-7b-chat'
    internlm2_math_20b = 'internlm2-math-20b'
    internlm2_math_20b_chat = 'internlm2-math-20b-chat'
    # internlm-xcomposer2
    internlm_xcomposer2_7b_chat = 'internlm-xcomposer2-7b-chat'
    # internvl
    internvl_chat_v1_5 = 'internvl-chat-v1_5'
    internvl_chat_v1_5_int8 = 'internvl-chat-v1_5-int8'
    mini_internvl_chat_2b_v1_5 = 'mini-internvl-chat-2b-v1_5'
    mini_internvl_chat_4b_v1_5 = 'mini-internvl-chat-4b-v1_5'
    # deepseek
    deepseek_7b = 'deepseek-7b'
    deepseek_7b_chat = 'deepseek-7b-chat'
    deepseek_moe_16b = 'deepseek-moe-16b'
    deepseek_moe_16b_chat = 'deepseek-moe-16b-chat'
    deepseek_67b = 'deepseek-67b'
    deepseek_67b_chat = 'deepseek-67b-chat'
    # deepseek-coder
    deepseek_coder_1_3b = 'deepseek-coder-1_3b'
    deepseek_coder_1_3b_instruct = 'deepseek-coder-1_3b-instruct'
    deepseek_coder_6_7b = 'deepseek-coder-6_7b'
    deepseek_coder_6_7b_instruct = 'deepseek-coder-6_7b-instruct'
    deepseek_coder_33b = 'deepseek-coder-33b'
    deepseek_coder_33b_instruct = 'deepseek-coder-33b-instruct'
    # deepseek2-coder
    deepseek_coder_v2_instruct = 'deepseek-coder-v2-instruct'
    deepseek_coder_v2_lite_instruct = 'deepseek-coder-v2-lite-instruct'
    # deepseek-math
    deepseek_math_7b = 'deepseek-math-7b'
    deepseek_math_7b_instruct = 'deepseek-math-7b-instruct'
    deepseek_math_7b_chat = 'deepseek-math-7b-chat'
    # deepseek-vl
    deepseek_vl_1_3b_chat = 'deepseek-vl-1_3b-chat'
    deepseek_vl_7b_chat = 'deepseek-vl-7b-chat'
    # deepseek-v2
    deepseek_v2_chat = 'deepseek-v2-chat'
    deepseek_v2_lite = 'deepseek-v2-lite'
    deepseek_v2_lite_chat = 'deepseek-v2-lite-chat'
    # gemma
    gemma_2b = 'gemma-2b'
    gemma_7b = 'gemma-7b'
    gemma_2b_instruct = 'gemma-2b-instruct'
    gemma_7b_instruct = 'gemma-7b-instruct'
    # paligemma
    paligemma_3b_pt_224 = 'paligemma-3b-pt-224'
    paligemma_3b_pt_448 = 'paligemma-3b-pt-448'
    paligemma_3b_pt_896 = 'paligemma-3b-pt-896'
    paligemma_3b_mix_224 = 'paligemma-3b-mix-224'
    paligemma_3b_mix_448 = 'paligemma-3b-mix-448'
    # minicpm
    minicpm_1b_sft_chat = 'minicpm-1b-sft-chat'
    minicpm_2b_sft_chat = 'minicpm-2b-sft-chat'
    minicpm_2b_chat = 'minicpm-2b-chat'
    minicpm_2b_128k = 'minicpm-2b-128k'
    minicpm_moe_8x2b = 'minicpm-moe-8x2b'
    # minicpm-v
    minicpm_v_3b_chat = 'minicpm-v-3b-chat'
    minicpm_v_v2_chat = 'minicpm-v-v2-chat'
    minicpm_v_v2_5_chat = 'minicpm-v-v2_5-chat'
    # openbuddy
    openbuddy_llama_65b_chat = 'openbuddy-llama-65b-chat'
    openbuddy_llama2_13b_chat = 'openbuddy-llama2-13b-chat'
    openbuddy_llama2_70b_chat = 'openbuddy-llama2-70b-chat'
    openbuddy_llama3_8b_chat = 'openbuddy-llama3-8b-chat'
    openbuddy_llama3_70b_chat = 'openbuddy-llama3-70b-chat'
    openbuddy_mistral_7b_chat = 'openbuddy-mistral-7b-chat'
    openbuddy_zephyr_7b_chat = 'openbuddy-zephyr-7b-chat'
    openbuddy_deepseek_67b_chat = 'openbuddy-deepseek-67b-chat'
    openbuddy_mixtral_moe_7b_chat = 'openbuddy-mixtral-moe-7b-chat'
    # mistral
    mistral_7b = 'mistral-7b'
    mistral_7b_v2 = 'mistral-7b-v2'
    mistral_7b_instruct = 'mistral-7b-instruct'
    mistral_7b_instruct_v2 = 'mistral-7b-instruct-v2'
    mixtral_moe_7b = 'mixtral-moe-7b'
    mixtral_moe_7b_instruct = 'mixtral-moe-7b-instruct'
    mixtral_moe_7b_aqlm_2bit_1x16 = 'mixtral-moe-7b-aqlm-2bit-1x16'  # aqlm
    mixtral_moe_8x22b_v1 = 'mixtral-moe-8x22b-v1'
    # wizardlm
    wizardlm2_7b_awq = 'wizardlm2-7b-awq'
    wizardlm2_8x22b = 'wizardlm2-8x22b'
    # baichuan
    baichuan_7b = 'baichuan-7b'
    baichuan_13b = 'baichuan-13b'
    baichuan_13b_chat = 'baichuan-13b-chat'
    # baichuan2
    baichuan2_7b = 'baichuan2-7b'
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    baichuan2_7b_chat_int4 = 'baichuan2-7b-chat-int4'
    baichuan2_13b = 'baichuan2-13b'
    baichuan2_13b_chat = 'baichuan2-13b-chat'
    baichuan2_13b_chat_int4 = 'baichuan2-13b-chat-int4'
    # owl
    mplug_owl2_chat = 'mplug-owl2-chat'  # llama
    mplug_owl2_1_chat = 'mplug-owl2_1-chat'  # qwen
    # yuan
    yuan2_2b_instruct = 'yuan2-2b-instruct'
    yuan2_2b_janus_instruct = 'yuan2-2b-janus-instruct'
    yuan2_51b_instruct = 'yuan2-51b-instruct'
    yuan2_102b_instruct = 'yuan2-102b-instruct'
    yuan2_m32 = 'yuan2-m32'
    # xverse
    xverse_7b = 'xverse-7b'
    xverse_7b_chat = 'xverse-7b-chat'
    xverse_13b = 'xverse-13b'
    xverse_13b_chat = 'xverse-13b-chat'
    xverse_65b = 'xverse-65b'
    xverse_65b_v2 = 'xverse-65b-v2'
    xverse_65b_chat = 'xverse-65b-chat'
    xverse_13b_256k = 'xverse-13b-256k'
    xverse_moe_a4_2b = 'xverse-moe-a4_2b'
    # orion
    orion_14b = 'orion-14b'
    orion_14b_chat = 'orion-14b-chat'
    # vivo
    bluelm_7b = 'bluelm-7b'
    bluelm_7b_32k = 'bluelm-7b-32k'
    bluelm_7b_chat = 'bluelm-7b-chat'
    bluelm_7b_chat_32k = 'bluelm-7b-chat-32k'
    # ziya
    ziya2_13b = 'ziya2-13b'
    ziya2_13b_chat = 'ziya2-13b-chat'
    # skywork
    skywork_13b = 'skywork-13b'
    skywork_13b_chat = 'skywork-13b-chat'
    # zephyr
    zephyr_7b_beta_chat = 'zephyr-7b-beta-chat'
    # other
    polylm_13b = 'polylm-13b'
    seqgpt_560m = 'seqgpt-560m'
    sus_34b_chat = 'sus-34b-chat'

    # tongyi-finance
    tongyi_finance_14b = 'tongyi-finance-14b'
    tongyi_finance_14b_chat = 'tongyi-finance-14b-chat'
    tongyi_finance_14b_chat_int4 = 'tongyi-finance-14b-chat-int4'
    # codefuse
    codefuse_codellama_34b_chat = 'codefuse-codellama-34b-chat'
    codefuse_codegeex2_6b_chat = 'codefuse-codegeex2-6b-chat'
    codefuse_qwen_14b_chat = 'codefuse-qwen-14b-chat'
    # phi
    phi2_3b = 'phi2-3b'
    phi3_4b_4k_instruct = 'phi3-4b-4k-instruct'
    phi3_4b_128k_instruct = 'phi3-4b-128k-instruct'
    phi3_small_128k_instruct = 'phi3-small-128k-instruct'
    phi3_medium_128k_instruct = 'phi3-medium-128k-instruct'

    phi3_vision_128k_instruct = 'phi3-vision-128k-instruct'
    # cogagent
    cogvlm_17b_chat = 'cogvlm-17b-chat'
    cogvlm2_19b_chat = 'cogvlm2-19b-chat'  # chinese
    cogvlm2_en_19b_chat = 'cogvlm2-en-19b-chat'
    cogagent_18b_chat = 'cogagent-18b-chat'
    cogagent_18b_instruct = 'cogagent-18b-instruct'
    # mamba
    mamba_130m = 'mamba-130m'
    mamba_370m = 'mamba-370m'
    mamba_390m = 'mamba-390m'
    mamba_790m = 'mamba-790m'
    mamba_1_4b = 'mamba-1.4b'
    mamba_2_8b = 'mamba-2.8b'
    # teleAI
    telechat_7b = 'telechat-7b'
    telechat_12b = 'telechat-12b'
    telechat_12b_v2 = 'telechat-12b-v2'
    telechat_12b_v2_gptq_int4 = 'telechat-12b-v2-gptq-int4'
    # grok-1
    grok_1 = 'grok-1'
    # dbrx
    dbrx_instruct = 'dbrx-instruct'
    dbrx_base = 'dbrx-base'
    # mengzi
    mengzi3_13b_base = 'mengzi3-13b-base'
    # c4ai
    c4ai_command_r_v01 = 'c4ai-command-r-v01'
    c4ai_command_r_plus = 'c4ai-command-r-plus'
    # codestral
    codestral_22b = 'codestral-22b'

    @classmethod
    def get_model_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_model_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


class LoRATM(NamedTuple):
    # default lora target modules. qkv
    baichuan = ['W_pack']
    chatglm = ['query_key_value']
    llama = ['q_proj', 'k_proj', 'v_proj']
    qwen = ['c_attn']
    polylm = ['c_attn']
    bloom = ['query_key_value']
    cogagent = [
        'vision_expert_query_key_value', 'vision_expert_dense', 'language_expert_query_key_value',
        'language_expert_dense', 'query', 'key_value', 'dense'
    ]
    cogvlm = [
        'vision_expert_query_key_value', 'vision_expert_dense', 'language_expert_query_key_value',
        'language_expert_dense'
    ]
    glm4v = ['self_attention.query_key_value']
    phi = ['Wqkv']
    phi3 = ['qkv_proj']
    phi3_small = ['query_key_value']  # what the hell???
    internlm2 = ['wqkv']
    mamba = ['in_proj', 'x_proj', 'embeddings', 'out_proj']
    telechat = ['key_value', 'query']
    dbrx = ['attn.Wqkv']
    mplug_owl2 = [
        'q_proj',
        'k_proj.multiway.0',
        'k_proj.multiway.1',
        'v_proj.multiway.0',
        'v_proj.multiway.1',
    ]
    mplug_owl2_1 = [
        'c_attn.multiway.0',
        'c_attn.multiway.1',
    ]
    deepseek2 = [
        'q_a_proj',
        'q_b_proj',
        'kv_a_proj_with_mqa',
        'kv_b_proj',
        'o_proj',
    ]
    # compat
    llama2 = llama


GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]


def register_model(
        model_type: str,
        model_id_or_path: Optional[str],
        lora_target_modules: Optional[List[str]] = None,
        template: str = TemplateType.default,
        get_function: Optional[GetModelTokenizerFunction] = None,
        *,
        requires: Optional[List[str]] = None,
        torch_dtype: Optional[Dtype] = None,
        hf_model_id: Optional[str] = None,
        revision: Optional[str] = None,  # only modelscope
        ignore_file_pattern: Optional[List[str]] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        exist_ok: bool = False,
        eos_token: Optional[str] = None,
        **kwargs) -> Optional[Callable[[GetModelTokenizerFunction], GetModelTokenizerFunction]]:
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if requires is None:
        requires = []
    if function_kwargs is None:
        function_kwargs = {}
    if revision is None:
        revision = 'master'
    model_info = {
        'model_id_or_path': model_id_or_path,
        'lora_target_modules': lora_target_modules,
        'template': template,
        'requires': requires,
        'torch_dtype': torch_dtype,
        'ignore_file_pattern': ignore_file_pattern,
        'hf_model_id': hf_model_id,
        'revision': revision,
        'eos_token': eos_token,
        **kwargs
    }

    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        model_info['get_function'] = get_function
        MODEL_MAPPING[model_type] = model_info
        return

    def _register_model(get_function: GetModelTokenizerFunction) -> GetModelTokenizerFunction:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        model_info['get_function'] = get_function
        MODEL_MAPPING[model_type] = model_info
        return _old_get_function

    return _register_model


def _check_awq_ext() -> None:
    try:
        from awq.utils.packing_utils import dequantize_gemm
        import awq_ext  # with CUDA kernels (AutoAWQ_kernels)
    except ImportError as e:
        raise ImportError('You are training awq models, remember installing awq_ext by '
                          '`git clone https://github.com/casper-hansen/AutoAWQ_kernels '
                          '&& cd AutoAWQ_kernels && pip install -e .`') from e


def _check_gptq_model(bits: int, model_config, model_kwargs: Dict[str, Any]) -> None:
    assert model_kwargs.get('quantization_config') is None
    if bits == 0:
        bits = model_config.quantization_config['bits']
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        model_kwargs['quantization_config'] = GPTQConfig(bits=bits, use_exllama=False)
    else:
        model_kwargs['quantization_config'] = GPTQConfig(bits=bits, disable_exllama=True)

    # fix quantlinear bug
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    __old_forward = QuantLinear.forward

    def _new_forward(self, x):
        if not self.training or not self.autogptq_cuda_available:
            return self.__old_forward(x)
        # fix sft no grad
        self.autogptq_cuda_available = False
        res = self.__old_forward(x)
        self.autogptq_cuda_available = True
        return res

    if not hasattr(QuantLinear, '__old_forward'):  # avoid double patching
        QuantLinear.__old_forward = __old_forward
        QuantLinear.forward = _new_forward


@register_model(
    ModelType.atom_7b,
    'FlagAlpha/Atom-7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='FlagAlpha/Atom-7B')
@register_model(
    ModelType.atom_7b_chat,
    'FlagAlpha/Atom-7B-Chat',
    LoRATM.llama,
    TemplateType.atom,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='FlagAlpha/Atom-7B-Chat')
@register_model(
    ModelType.internlm_20b,
    'Shanghai_AI_Laboratory/internlm-20b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='internlm/internlm2-20b')
@register_model(
    ModelType.internlm_7b,
    'Shanghai_AI_Laboratory/internlm-7b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='internlm/internlm-7b')
@register_model(
    ModelType.bluelm_7b_chat_32k,
    'vivo-ai/BlueLM-7B-Chat-32K',
    LoRATM.llama,
    TemplateType.bluelm,
    hf_model_id='vivo-ai/BlueLM-7B-Chat-32K')
@register_model(
    ModelType.bluelm_7b_chat,
    'vivo-ai/BlueLM-7B-Chat',
    LoRATM.llama,
    TemplateType.bluelm,
    hf_model_id='vivo-ai/BlueLM-7B-Chat')
@register_model(
    ModelType.bluelm_7b_32k,
    'vivo-ai/BlueLM-7B-Base-32K',
    LoRATM.llama,
    TemplateType.default_generation,
    hf_model_id='vivo-ai/BlueLM-7B-Base-32K')
@register_model(
    ModelType.bluelm_7b,
    'vivo-ai/BlueLM-7B-Base',
    LoRATM.llama,
    TemplateType.default_generation,
    hf_model_id='vivo-ai/BlueLM-7B-Base')
@register_model(
    ModelType.seqgpt_560m,
    'damo/nlp_seqgpt-560m',
    LoRATM.bloom,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='DAMO-NLP/SeqGPT-560M')
@register_model(
    ModelType.xverse_13b_chat,
    'xverse/XVERSE-13B-Chat',
    LoRATM.llama,
    TemplateType.xverse,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-13B-Chat')
@register_model(
    ModelType.xverse_13b,
    'xverse/XVERSE-13B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-13B')
@register_model(
    ModelType.xverse_65b,
    'xverse/XVERSE-65B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-65B')
@register_model(
    ModelType.xverse_65b_v2,
    'xverse/XVERSE-65B-2',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-65B-2')
@register_model(
    ModelType.xverse_65b_chat,
    'xverse/XVERSE-65B-Chat',
    LoRATM.llama,
    TemplateType.xverse,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-65B-Chat')
@register_model(
    ModelType.xverse_13b_256k,
    'xverse/XVERSE-13B-256K',
    LoRATM.llama,
    TemplateType.default_generation,
    revision='v1.0.0',
    support_vllm=True,
    hf_model_id='xverse/XVERSE-13B-256K')
@register_model(
    ModelType.xverse_7b_chat,
    'xverse/XVERSE-7B-Chat',
    LoRATM.llama,
    TemplateType.xverse,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-7B-Chat')
@register_model(
    ModelType.xverse_7b,
    'xverse/XVERSE-7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='xverse/XVERSE-7B')
@register_model(
    ModelType.xverse_moe_a4_2b,
    'xverse/XVERSE-MoE-A4.2B',
    LoRATM.llama,
    TemplateType.default_generation,
    hf_model_id='xverse/XVERSE-MoE-A4.2B')
@register_model(
    ModelType.baichuan_13b_chat,
    'baichuan-inc/Baichuan-13B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    requires=['transformers<4.34'],
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan-13B-Chat')
@register_model(
    ModelType.baichuan_7b,
    'baichuan-inc/baichuan-7B',
    LoRATM.baichuan,
    TemplateType.default_generation,
    requires=['transformers<4.34'],
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan-7B')
@register_model(
    ModelType.mengzi3_13b_base,
    'langboat/Mengzi3-13B-Base',
    LoRATM.llama,
    TemplateType.mengzi,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='Langboat/Mengzi3-13B-Base')
@register_model(
    ModelType.c4ai_command_r_v01,
    'AI-ModelScope/c4ai-command-r-v01',
    LoRATM.llama,
    TemplateType.c4ai,
    requires=['transformers>=4.39.1'],
    support_vllm=False,
    support_flash_attn=True,
    hf_model_id='CohereForAI/c4ai-command-r-v01')
@register_model(
    ModelType.c4ai_command_r_plus,
    'AI-ModelScope/c4ai-command-r-plus',
    LoRATM.llama,
    TemplateType.c4ai,
    requires=['transformers>4.39'],
    support_vllm=False,
    support_flash_attn=True,
    hf_model_id='CohereForAI/c4ai-command-r-plus')
@register_model(
    ModelType.chinese_llama_2_1_3b,
    'AI-ModelScope/chinese-llama-2-1.3b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-1.3b')
@register_model(
    ModelType.chinese_llama_2_7b,
    'AI-ModelScope/chinese-llama-2-7b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-7b')
@register_model(
    ModelType.chinese_llama_2_7b_16k,
    'AI-ModelScope/chinese-llama-2-7b-16k',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-7b-16k')
@register_model(
    ModelType.chinese_llama_2_7b_64k,
    'AI-ModelScope/chinese-llama-2-7b-64k',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-7b-64k')
@register_model(
    ModelType.chinese_llama_2_13b,
    'AI-ModelScope/chinese-llama-2-13b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-13b')
@register_model(
    ModelType.chinese_llama_2_13b_16k,
    'AI-ModelScope/chinese-llama-2-13b-16k',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-llama-2-13b-16k')
@register_model(
    ModelType.chinese_alpaca_2_1_3b,
    'AI-ModelScope/chinese-alpaca-2-1.3b',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-1.3b')
@register_model(
    ModelType.chinese_alpaca_2_7b,
    'AI-ModelScope/chinese-alpaca-2-7b',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-7b')
@register_model(
    ModelType.chinese_alpaca_2_7b_16k,
    'AI-ModelScope/chinese-alpaca-2-7b-16k',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-7b-16k')
@register_model(
    ModelType.chinese_alpaca_2_7b_64k,
    'AI-ModelScope/chinese-alpaca-2-7b-64k',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-7b-64k')
@register_model(
    ModelType.chinese_alpaca_2_13b,
    'AI-ModelScope/chinese-alpaca-2-13b',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-13b')
@register_model(
    ModelType.chinese_alpaca_2_13b_16k,
    'AI-ModelScope/chinese-alpaca-2-13b-16k',
    LoRATM.llama,
    TemplateType.llama,
    support_vllm=True,
    support_flash_attn=True,
    hf_model_id='hfl/chinese-alpaca-2-13b-16k')
def get_model_tokenizer_from_repo(model_dir: str,
                                  torch_dtype: Optional[Dtype],
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  tokenizer=None,
                                  automodel_class=AutoModelForCausalLM,
                                  **kwargs):
    """load from an independent repository"""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    is_awq = kwargs.pop('is_awq', False)
    is_aqlm = kwargs.pop('is_aqlm', False)
    gptq_bits = kwargs.pop('gptq_bits', 0)
    if gptq_bits > 0:
        is_gptq = True
    else:
        is_gptq = kwargs.pop('is_gptq', False)
    is_training = kwargs.pop('is_training', False)
    if is_awq and is_training:
        _check_awq_ext()
    if is_gptq and is_training:
        _check_gptq_model(gptq_bits, model_config, model_kwargs)
    context = kwargs.get('context', None)
    if is_aqlm and is_training:
        require_version('transformers>=4.39')
        import aqlm
        context = aqlm.optimize_for_training()
    if context is None:
        context = nullcontext()
    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    pad_token = kwargs.get('pad_token')
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    placeholder_tokens = kwargs.get('placeholder_tokens')
    if placeholder_tokens is not None:
        tokenizer.placeholder_tokens = placeholder_tokens
        tokenizer.placeholder_tokens_id = [tokenizer.convert_tokens_to_ids(token) for token in placeholder_tokens]
    model = None

    rope_scaling = kwargs.pop('rope_scaling', None)
    max_position_embeddings = getattr(model_config, 'max_position_embeddings', None)
    if rope_scaling and max_position_embeddings:
        max_length = kwargs.get('max_length') or max_position_embeddings
        rope_scaling_factor = max(float(math.ceil(max_length / max_position_embeddings)), 1.0)
        setattr(model_config, 'rope_scaling', {'type': rope_scaling, 'factor': rope_scaling_factor})

    if load_model:
        if kwargs.get('use_unsloth', False):
            assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
            if 'qwen' in model_dir:
                logger.warn('If using qwen2 models, please install unsloth with '
                            '`pip install git+https://github.com/yangjianxin1/unsloth`')
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_dir,
                max_seq_length=kwargs.get('max_length', None),
                dtype=torch_dtype,
                load_in_4bit=kwargs.get('load_in_4bit', True),
                trust_remote_code=True,
            )
        else:
            with context:
                model = automodel_class.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


@register_model(
    ModelType.cogvlm2_en_19b_chat,
    'ZhipuAI/cogvlm2-llama3-chat-19B',
    LoRATM.cogvlm,
    TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    placeholder_tokens=['<|reserved_special_token_0|>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm2-llama3-chat-19B')
@register_model(
    ModelType.cogvlm2_19b_chat,
    'ZhipuAI/cogvlm2-llama3-chinese-chat-19B',
    LoRATM.cogvlm,
    TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    placeholder_tokens=['<|reserved_special_token_0|>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm2-llama3-chinese-chat-19B')
def get_model_tokenizer_cogvlm2(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(*args, **kwargs)
    if model is not None:
        # fix device map 4
        def _output_device_map_hook(module, input, output):
            return output.to(input[0].device)

        for layer in model.model.vision.transformer.layers:
            layer.mlp.register_forward_hook(_output_device_map_hook)
            layer.post_attention_layernorm.register_forward_hook(_output_device_map_hook)

        device = next(model.model.vision.linear_proj.parameters()).device
        model.model.vision.boi.data = model.model.vision.boi.to(device)
        model.model.vision.eoi.data = model.model.vision.eoi.to(device)
    return model, tokenizer


@register_model(
    ModelType.llava_llama3_8b_v1_1,
    'AI-ModelScope/llava-llama-3-8b-v1_1-transformers',
    LoRATM.llama,
    TemplateType.llava_llama_instruct,
    support_flash_attn=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='xtuner/llava-llama-3-8b-v1_1-transformers')
def get_model_tokenizer_llava_llama(model_dir: str,
                                    torch_dtype: Dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import LlavaForConditionalGeneration, LlavaConfig, AutoProcessor

    model_config = LlavaConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=LlavaForConditionalGeneration,
        **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


@register_model(
    ModelType.grok_1,
    'colossalai/grok-1-pytorch',
    LoRATM.llama,
    TemplateType.default_generation,
    support_vllm=False,
    support_flash_attn=False,
    hf_model_id='hpcai-tech/grok-1')
def get_model_tokenizer_grok(model_dir: str,
                             torch_dtype: Optional[Dtype],
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             tokenizer=None,
                             automodel_class=AutoModelForCausalLM,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            'AI-ModelScope/grok-1-tokenizer', revision='master', trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


@register_model(
    ModelType.mamba_130m,
    'AI-ModelScope/mamba-130m-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-130m-hf')
@register_model(
    ModelType.mamba_370m,
    'AI-ModelScope/mamba-370m-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-370m-hf')
@register_model(
    ModelType.mamba_390m,
    'AI-ModelScope/mamba-390m-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-390m-hf')
@register_model(
    ModelType.mamba_790m,
    'AI-ModelScope/mamba-790m-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-790m-hf')
@register_model(
    ModelType.mamba_1_4b,
    'AI-ModelScope/mamba-1.4b-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-1.4b-hf')
@register_model(
    ModelType.mamba_2_8b,
    'AI-ModelScope/mamba-2.8b-hf',
    LoRATM.mamba,
    TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-2.8b-hf')
def get_model_tokenizer_mamba(model_dir: str,
                              torch_dtype: Optional[Dtype],
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    logger.info('[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
                'be really slow!')
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


@register_model(
    ModelType.cogvlm_17b_chat,
    'ZhipuAI/cogvlm-chat',
    LoRATM.cogvlm,
    TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm-chat-hf')
@register_model(
    ModelType.cogagent_18b_chat,
    'ZhipuAI/cogagent-chat',
    LoRATM.cogagent,
    TemplateType.cogagent_chat,
    support_gradient_checkpointing=False,
    requires=['timm'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogagent-chat-hf')
@register_model(
    ModelType.cogagent_18b_instruct,
    'ZhipuAI/cogagent-vqa',
    LoRATM.cogagent,
    TemplateType.cogagent_instruct,
    support_gradient_checkpointing=False,
    requires=['timm'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogagent-vqa-hf')
def get_model_tokenizer_cogagent(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/vicuna-7b-v1.5', revision='master', trust_remote_code=True)
    if load_model is True:
        logger.warning('CogAgent with FusedLayerNorm will cause an training loss of NAN, '
                       'to avoid this, please uninstall apex.')
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    logger.info('Please ignore the unimported warning.')
    return model, tokenizer


@register_model(
    ModelType.internlm_20b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-20b',
    LoRATM.llama,
    TemplateType.internlm,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-20b')
@register_model(
    ModelType.internlm_7b_chat_8k,
    'Shanghai_AI_Laboratory/internlm-chat-7b-8k',
    LoRATM.llama,
    TemplateType.internlm,
    support_vllm=True)
@register_model(
    ModelType.internlm_7b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-7b',
    LoRATM.llama,
    TemplateType.internlm,
    support_vllm=True,
    hf_model_id='internlm/internlm-chat-7b')
def get_model_tokenizer_internlm_chat(model_dir: str,
                                      torch_dtype: Dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
        del tokenizer.__class__.eos_token_id
    tokenizer.eos_token = '<eoa>'
    return model, tokenizer


@register_model(
    ModelType.baichuan_13b,
    'baichuan-inc/Baichuan-13B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    requires=['transformers<4.34'],
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan-13B-Base')
def get_model_tokenizer_baichuan_13b(model_dir: str,
                                     torch_dtype: Dtype,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    # baichuan-13b does not implement the `get_input_embeddings` function
    # fix gradient_checkpointing bug
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        model.__class__.get_input_embeddings = lambda self: self.model.embed_tokens
    return model, tokenizer


@register_model(
    ModelType.paligemma_3b_pt_224,
    'AI-ModelScope/paligemma-3b-pt-224',
    LoRATM.llama,
    TemplateType.paligemma,
    support_flash_attn=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-224')
@register_model(
    ModelType.paligemma_3b_pt_448,
    'AI-ModelScope/paligemma-3b-pt-448',
    LoRATM.llama,
    TemplateType.paligemma,
    support_flash_attn=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-448')
@register_model(
    ModelType.paligemma_3b_pt_896,
    'AI-ModelScope/paligemma-3b-pt-896',
    LoRATM.llama,
    TemplateType.paligemma,
    support_flash_attn=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-896')
@register_model(
    ModelType.paligemma_3b_mix_224,
    'AI-ModelScope/paligemma-3b-mix-224',
    LoRATM.llama,
    TemplateType.paligemma,
    support_flash_attn=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-mix-224')
@register_model(
    ModelType.paligemma_3b_mix_448,
    'AI-ModelScope/paligemma-3b-mix-448',
    LoRATM.llama,
    TemplateType.paligemma,
    support_flash_attn=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-mix-448')
def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         torch_dtype: Dtype,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, automodel_class=PaliGemmaForConditionalGeneration, **kwargs)
    tokenizer.processor = processor
    if model is not None:
        model.max_position_embeddings = model.language_model.config.max_position_embeddings
    return model, tokenizer


def _clone_hook(module, input, output):
    return output.requires_grad_(True).clone()


@register_model(
    ModelType.phi3_vision_128k_instruct,
    'LLM-Research/Phi-3-vision-128k-instruct',
    LoRATM.phi3,
    TemplateType.phi3_vl,
    support_flash_attn=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='microsoft/Phi-3-vision-128k-instruct')
def get_model_tokenizer_phi3_vision(model_dir: str,
                                    torch_dtype: Dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor

    if load_model:
        model.model.vision_embed_tokens.wte.register_forward_hook(_clone_hook)

    return model, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat,
    'baichuan-inc/Baichuan2-13B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan2-13B-Chat')
@register_model(
    ModelType.baichuan2_13b,
    'baichuan-inc/Baichuan2-13B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan2-13B-Base')
def get_model_tokenizer_baichuan2_13b(model_dir: str,
                                      torch_dtype: Dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # patch: baichuan2_13b configuration_baichuan.py bug
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    gradient_checkpointing = model_config.gradient_checkpointing
    if isinstance(gradient_checkpointing, (tuple, list)):
        model_config.gradient_checkpointing = gradient_checkpointing[0]
    return get_model_tokenizer_baichuan2(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


def patch_baichuan2_lm_head_forward(self, hidden_states: Tensor) -> Tensor:
    # patch: baichuan2 lm_head (fp32 bug)
    if self.training:
        norm_weight = F.normalize(self.weight).to(self.weight.dtype)
    elif self.first_flag:
        self.first_flag = False
        self.weight.data = F.normalize(self.weight).to(self.weight.dtype)
        norm_weight = self.weight
    else:
        norm_weight = self.weight
    return F.linear(hidden_states, norm_weight)


@register_model(
    ModelType.baichuan2_7b_chat,
    'baichuan-inc/Baichuan2-7B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan2-7B-Chat')
@register_model(
    ModelType.baichuan2_7b,
    'baichuan-inc/Baichuan2-7B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    support_vllm=True,
    hf_model_id='baichuan-inc/Baichuan2-7B-Base')
def get_model_tokenizer_baichuan2(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(model_config, 'z_loss_weight'):
        model_config.z_loss_weight = 0
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    model_ori = model
    if model is not None:
        if not hasattr(model, 'lm_head'):  # fix awq
            model = model.model
        new_forward = MethodType(patch_baichuan2_lm_head_forward, model.lm_head)
        if hasattr(model, '_old_forward'):  # device_map
            model.lm_head._old_forward = new_forward
        else:
            model.lm_head.forward = new_forward
    return model_ori, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat_int4,
    'baichuan-inc/Baichuan2-13B-Chat-4bits',
    LoRATM.baichuan,
    TemplateType.baichuan,
    function_kwargs={'get_baichuan2_function': get_model_tokenizer_baichuan2_13b},
    torch_dtype=torch.bfloat16,
    requires=['bitsandbytes<0.41.2', 'accelerate<0.26'],
    hf_model_id='baichuan-inc/Baichuan2-13B-Chat-4bits')
@register_model(
    ModelType.baichuan2_7b_chat_int4,
    'baichuan-inc/Baichuan2-7B-Chat-4bits',
    LoRATM.baichuan,
    TemplateType.baichuan,
    torch_dtype=torch.bfloat16,
    requires=['bitsandbytes<0.41.2', 'accelerate<0.26'],
    hf_model_id='baichuan-inc/Baichuan2-7B-Chat-4bits')
def get_model_tokenizer_baichuan2_int4(model_dir: str,
                                       torch_dtype: Dtype,
                                       model_kwargs: Dict[str, Any],
                                       load_model: bool = True,
                                       **kwargs):
    logger.info('use `model_config.quantization_config`, ignore bnb arguments')
    model_kwargs.pop('quantization_config', None)

    # fix device_map bug
    import accelerate
    _old_infer_auto_device_map = accelerate.infer_auto_device_map
    device_map = model_kwargs.get('device_map', None)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = lambda *args, **kwargs: device_map
    get_baichuan2_function = kwargs.pop('get_baichuan2_function', get_model_tokenizer_baichuan2)
    model, tokenizer = get_baichuan2_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = _old_infer_auto_device_map
    if model is not None:
        model.config.quantization_config = BitsAndBytesConfig(**model.config.quantization_config)
        model.train()
        model._is_quantized_training_enabled = True
        model.is_loaded_in_4bit = True
    return model, tokenizer


def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase], tokenizer_config: Dict[str, Any]) -> None:
    for k, v in tokenizer_cls.__dict__.items():
        if k.endswith('_token') and isinstance(v, property) and k in tokenizer_config:
            setattr(tokenizer_cls, k, tokenizer_config[k])


@register_model(
    ModelType.glm4_9b,
    'ZhipuAI/glm-4-9b',
    LoRATM.chatglm,
    TemplateType.chatglm_generation,
    support_vllm=True,
    hf_model_id='THUDM/glm-4-9b')
@register_model(
    ModelType.glm4_9b_chat,
    'ZhipuAI/glm-4-9b-chat',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True,
    function_kwargs={'kv_cache_patch': True},
    hf_model_id='THUDM/glm-4-9b-chat')
@register_model(
    ModelType.glm4_9b_chat_1m,
    'ZhipuAI/glm-4-9b-chat-1m',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True,
    function_kwargs={'kv_cache_patch': True},
    hf_model_id='THUDM/glm-4-9b-chat-1m')
@register_model(
    ModelType.glm4v_9b_chat,
    'ZhipuAI/glm-4v-9b',
    LoRATM.glm4v,
    TemplateType.glm4v,
    eos_token='<|endoftext|>',
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/glm-4v-9b')
@register_model(
    ModelType.codefuse_codegeex2_6b_chat,
    'codefuse-ai/CodeFuse-CodeGeeX2-6B',
    LoRATM.chatglm,
    TemplateType.codefuse,
    requires=['transformers<4.34'],
    support_vllm=True,
    tags=['coding'],
    hf_model_id='codefuse-ai/CodeFuse-CodeGeeX2-6B')
@register_model(
    ModelType.chatglm3_6b_32k,
    'ZhipuAI/chatglm3-6b-32k',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True,
    hf_model_id='THUDM/chatglm3-6b-32k')
@register_model(
    ModelType.chatglm3_6b_128k,
    'ZhipuAI/chatglm3-6b-128k',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True,
    hf_model_id='THUDM/chatglm3-6b-128k')
@register_model(
    ModelType.chatglm3_6b,
    'ZhipuAI/chatglm3-6b',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True,
    hf_model_id='THUDM/chatglm3-6b')
@register_model(
    ModelType.chatglm3_6b_base,
    'ZhipuAI/chatglm3-6b-base',
    LoRATM.chatglm,
    TemplateType.chatglm_generation,
    support_vllm=True,
    hf_model_id='THUDM/chatglm3-6b-base')
@register_model(
    ModelType.chatglm2_6b_32k,
    'ZhipuAI/chatglm2-6b-32k',
    LoRATM.chatglm,
    TemplateType.chatglm2,
    support_vllm=True,
    hf_model_id='THUDM/chatglm2-6b-32k')
@register_model(
    ModelType.chatglm2_6b,
    'ZhipuAI/chatglm2-6b',
    LoRATM.chatglm,
    TemplateType.chatglm2,
    support_vllm=True,
    hf_model_id='THUDM/chatglm2-6b')
@register_model(
    ModelType.codegeex2_6b,
    'ZhipuAI/codegeex2-6b',
    LoRATM.chatglm,
    TemplateType.chatglm_generation,
    requires=['transformers<4.34'],
    support_vllm=True,
    tags=['coding'],
    hf_model_id='THUDM/codegeex2-6b')
def get_model_tokenizer_chatglm(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    kv_cache_patch = kwargs.pop('kv_cache_patch', False)
    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.init_kwargs['image_size'] = 1120
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: Tensor, target: Tensor) -> Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward

        if kv_cache_patch:
            device = next(model.parameters()).device.type

            def _output_device_map_hook(module, input, output):
                kv_cache = output[1]
                if kv_cache is not None and isinstance(kv_cache, torch.Tensor):
                    kv_cache = kv_cache.to(f'{device}:0')
                return output[0], kv_cache

            for layer in model.transformer.encoder.layers:
                layer.register_forward_hook(_output_device_map_hook)
    return model, tokenizer


@register_model(
    ModelType.qwen2_57b_a14b,
    'qwen/Qwen2-57B-A14B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    requires=['transformers>=4.40'],
    hf_model_id='Qwen/Qwen2-57B-A14B')
@register_model(
    ModelType.qwen2_0_5b,
    'qwen/Qwen2-0.5B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-0.5B')
@register_model(
    ModelType.qwen2_1_5b,
    'qwen/Qwen2-1.5B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-1.5B')
@register_model(
    ModelType.qwen2_7b,
    'qwen/Qwen2-7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-7B')
@register_model(
    ModelType.qwen2_72b,
    'qwen/Qwen2-72B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-72B')
@register_model(
    ModelType.minicpm_2b_sft_chat,
    'OpenBMB/MiniCPM-2B-sft-fp32',
    LoRATM.llama,
    TemplateType.minicpm,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='openbmb/MiniCPM-2B-sft-fp32')
@register_model(
    ModelType.minicpm_2b_chat,
    'OpenBMB/MiniCPM-2B-dpo-fp32',
    LoRATM.llama,
    TemplateType.minicpm,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='openbmb/MiniCPM-2B-dpo-fp32')
@register_model(
    ModelType.minicpm_1b_sft_chat,
    'OpenBMB/MiniCPM-1B-sft-bf16',
    LoRATM.llama,
    TemplateType.minicpm,
    requires=['transformers>=4.36.0'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='openbmb/MiniCPM-1B-sft-bf16')
@register_model(
    ModelType.minicpm_2b_128k,
    'OpenBMB/MiniCPM-2B-128k',
    LoRATM.llama,
    TemplateType.chatml,
    requires=['transformers>=4.36.0'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='openbmb/MiniCPM-2B-128k')
@register_model(
    ModelType.phi3_4b_128k_instruct,
    'LLM-Research/Phi-3-mini-128k-instruct',
    LoRATM.phi3,
    TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['general'],
    hf_model_id='microsoft/Phi-3-mini-128k-instruct')
@register_model(
    ModelType.phi3_medium_128k_instruct,
    'LLM-Research/Phi-3-medium-128k-instruct',
    LoRATM.phi3,
    TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['general'],
    hf_model_id='microsoft/Phi-3-medium-128k-instruct')
@register_model(
    ModelType.phi3_4b_4k_instruct,
    'LLM-Research/Phi-3-mini-4k-instruct',
    LoRATM.phi3,
    TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=False,
    tags=['general'],
    hf_model_id='microsoft/Phi-3-mini-4k-instruct')
@register_model(
    ModelType.wizardlm2_8x22b,
    'AI-ModelScope/WizardLM-2-8x22B',
    LoRATM.llama,
    TemplateType.wizardlm2,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='alpindale/WizardLM-2-8x22B')
@register_model(
    ModelType.wizardlm2_7b_awq,
    'AI-ModelScope/WizardLM-2-7B-AWQ',
    LoRATM.llama,
    TemplateType.wizardlm2_awq,
    requires=['transformers>=4.34'],
    torch_dtype=torch.float16,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    hf_model_id='MaziyarPanahi/WizardLM-2-7B-AWQ')
@register_model(
    ModelType.gemma_2b,
    'AI-ModelScope/gemma-2b',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.38'],
    ignore_file_pattern=[r'.+\.gguf$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='google/gemma-2b')
@register_model(
    ModelType.gemma_7b,
    'AI-ModelScope/gemma-7b',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.38'],
    ignore_file_pattern=[r'.+\.gguf$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='google/gemma-7b')
@register_model(
    ModelType.gemma_2b_instruct,
    'AI-ModelScope/gemma-2b-it',
    LoRATM.llama,
    TemplateType.gemma,
    requires=['transformers>=4.38'],
    ignore_file_pattern=[r'.+\.gguf$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='google/gemma-2b-it')
@register_model(
    ModelType.gemma_7b_instruct,
    'AI-ModelScope/gemma-7b-it',
    LoRATM.llama,
    TemplateType.gemma,
    requires=['transformers>=4.38'],
    ignore_file_pattern=[r'.+\.gguf$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='google/gemma-7b-it')
@register_model(
    ModelType.deepseek_math_7b_instruct,
    'deepseek-ai/deepseek-math-7b-instruct',
    LoRATM.llama,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='deepseek-ai/deepseek-math-7b-instruct')
@register_model(
    ModelType.deepseek_math_7b_chat,
    'deepseek-ai/deepseek-math-7b-rl',
    LoRATM.llama,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='deepseek-ai/deepseek-math-7b-rl')
@register_model(
    ModelType.deepseek_math_7b,
    'deepseek-ai/deepseek-math-7b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='deepseek-ai/deepseek-math-7b-base')
@register_model(
    ModelType.qwen1half_0_5b,
    'qwen/Qwen1.5-0.5B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-0.5B')
@register_model(
    ModelType.qwen1half_1_8b,
    'qwen/Qwen1.5-1.8B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-1.8B')
@register_model(
    ModelType.qwen1half_4b,
    'qwen/Qwen1.5-4B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-4B')
@register_model(
    ModelType.qwen1half_7b,
    'qwen/Qwen1.5-7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-7B')
@register_model(
    ModelType.qwen1half_14b,
    'qwen/Qwen1.5-14B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-14B')
@register_model(
    ModelType.qwen1half_32b,
    'qwen/Qwen1.5-32B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-32B')
@register_model(
    ModelType.qwen1half_72b,
    'qwen/Qwen1.5-72B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-72B')
@register_model(
    ModelType.qwen1half_110b,
    'qwen/Qwen1.5-110B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-110B')
@register_model(
    ModelType.codeqwen1half_7b,
    'qwen/CodeQwen1.5-7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/CodeQwen1.5-7B')
@register_model(
    ModelType.qwen1half_moe_a2_7b,
    'qwen/Qwen1.5-MoE-A2.7B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.40'],
    hf_model_id='Qwen/Qwen1.5-MoE-A2.7B')
@register_model(
    ModelType.deepseek_coder_1_3b,
    'deepseek-ai/deepseek-coder-1.3b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-1.3b-base')
@register_model(
    ModelType.deepseek_coder_6_7b,
    'deepseek-ai/deepseek-coder-6.7b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-6.7b-base')
@register_model(
    ModelType.deepseek_coder_33b,
    'deepseek-ai/deepseek-coder-33b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-33b-base')
@register_model(
    ModelType.deepseek_coder_1_3b_instruct,
    'deepseek-ai/deepseek-coder-1.3b-instruct',
    LoRATM.llama,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-1.3b-instruct')
@register_model(
    ModelType.deepseek_coder_6_7b_instruct,
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    LoRATM.llama,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-6.7b-instruct')
@register_model(
    ModelType.deepseek_coder_33b_instruct,
    'deepseek-ai/deepseek-coder-33b-instruct',
    LoRATM.llama,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='deepseek-ai/deepseek-coder-33b-instruct')
@register_model(
    ModelType.openbuddy_deepseek_67b_chat,
    'OpenBuddy/openbuddy-deepseek-67b-v15.2',
    LoRATM.llama,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-deepseek-67b-v15.2')
@register_model(
    ModelType.deepseek_67b_chat,
    'deepseek-ai/deepseek-llm-67b-chat',
    LoRATM.llama,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-llm-67b-chat')
@register_model(
    ModelType.deepseek_67b,
    'deepseek-ai/deepseek-llm-67b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-llm-67b-base')
@register_model(
    ModelType.deepseek_7b_chat,
    'deepseek-ai/deepseek-llm-7b-chat',
    LoRATM.llama,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-llm-7b-chat')
@register_model(
    ModelType.deepseek_7b,
    'deepseek-ai/deepseek-llm-7b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-llm-7b-base')
@register_model(
    ModelType.sus_34b_chat,
    'SUSTC/SUS-Chat-34B',
    LoRATM.llama,
    TemplateType.sus,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='SUSTech/SUS-Chat-34B')
@register_model(
    ModelType.openbuddy_zephyr_7b_chat,
    'OpenBuddy/openbuddy-zephyr-7b-v14.1',
    LoRATM.llama,
    TemplateType.openbuddy,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-zephyr-7b-v14.1')
@register_model(
    ModelType.zephyr_7b_beta_chat,
    'modelscope/zephyr-7b-beta',
    LoRATM.llama,
    TemplateType.zephyr,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='HuggingFaceH4/zephyr-7b-beta')
@register_model(
    ModelType.yi_6b_chat,
    '01ai/Yi-6B-Chat',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-6B-Chat')
@register_model(
    ModelType.yi_6b_chat_awq,
    '01ai/Yi-6B-Chat-4bits',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-6B-Chat-4bits')
@register_model(
    ModelType.yi_6b_chat_int8,
    '01ai/Yi-6B-Chat-8bits',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-6B-Chat-8bits')
@register_model(
    ModelType.yi_34b_chat,
    '01ai/Yi-34B-Chat',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-34B-Chat')
@register_model(
    ModelType.yi_34b_chat_awq,
    '01ai/Yi-34B-Chat-4bits',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-34B-Chat-4bits')
@register_model(
    ModelType.yi_34b_chat_int8,
    '01ai/Yi-34B-Chat-8bits',
    LoRATM.llama,
    TemplateType.yi,
    eos_token='<|im_end|>',
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-34B-Chat-8bits')
@register_model(
    ModelType.yi_34b_200k,
    '01ai/Yi-34B-200K',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-34B-200K')
@register_model(
    ModelType.yi_34b,
    '01ai/Yi-34B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-34B')
@register_model(
    ModelType.yi_6b_200k,
    '01ai/Yi-6B-200K',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-6B-200K')
@register_model(
    ModelType.yi_9b,
    '01ai/Yi-9B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-9B')
@register_model(
    ModelType.yi_9b_200k,
    '01ai/Yi-9B-200K',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-9B-200K')
@register_model(
    ModelType.yi_6b,
    '01ai/Yi-6B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-6B')
@register_model(
    ModelType.yi_1_5_6b,
    '01ai/Yi-1.5-6B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-6B')
@register_model(
    ModelType.yi_1_5_6b_chat,
    '01ai/Yi-1.5-6B-Chat',
    LoRATM.llama,
    TemplateType.yi1_5,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-6B-Chat')
@register_model(
    ModelType.yi_1_5_6b_chat_awq_int4,
    'AI-ModelScope/Yi-1.5-6B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-6B-Chat-AWQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_6b_chat_gptq_int4,
    'AI-ModelScope/Yi-1.5-6B-Chat-GPTQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['auto_gptq>=0.5'],
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-6B-Chat-GPTQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_9b_chat_awq_int4,
    'AI-ModelScope/Yi-1.5-9B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-9B-Chat-AWQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_9b_chat_gptq_int4,
    'AI-ModelScope/Yi-1.5-9B-Chat-GPTQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['auto_gptq>=0.5'],
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-9B-Chat-GPTQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_34b_chat_awq_int4,
    'AI-ModelScope/Yi-1.5-34B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-34B-Chat-AWQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_34b_chat_gptq_int4,
    'AI-ModelScope/Yi-1.5-34B-Chat-GPTQ',
    LoRATM.llama,
    TemplateType.yi1_5,
    requires=['auto_gptq>=0.5'],
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    support_flash_attn=True,
    hf_model_id='modelscope/Yi-1.5-34B-Chat-GPTQ',
    support_vllm=True)
@register_model(
    ModelType.yi_1_5_9b,
    '01ai/Yi-1.5-9B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-9B')
@register_model(
    ModelType.yi_1_5_9b_chat,
    '01ai/Yi-1.5-9B-Chat',
    LoRATM.llama,
    TemplateType.yi1_5,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-9B-Chat')
@register_model(
    ModelType.yi_1_5_9b_chat_16k,
    '01ai/Yi-1.5-9B-Chat',
    LoRATM.llama,
    TemplateType.yi1_5,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-9B-Chat-16K')
@register_model(
    ModelType.yi_1_5_34b,
    '01ai/Yi-1.5-34B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-34B')
@register_model(
    ModelType.yi_1_5_34b_chat,
    '01ai/Yi-1.5-34B-Chat',
    LoRATM.llama,
    TemplateType.yi1_5,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-34B-Chat')
@register_model(
    ModelType.yi_1_5_34b_chat_16k,
    '01ai/Yi-1.5-34B-Chat-16K',
    LoRATM.llama,
    TemplateType.yi1_5,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='01-ai/Yi-1.5-34B-Chat-16K')
@register_model(
    ModelType.ziya2_13b_chat,
    'Fengshenbang/Ziya2-13B-Chat',
    LoRATM.llama,
    TemplateType.ziya,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='IDEA-CCNL/Ziya2-13B-Chat')
@register_model(
    ModelType.ziya2_13b,
    'Fengshenbang/Ziya2-13B-Base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='IDEA-CCNL/Ziya2-13B-Base')
@register_model(
    ModelType.openbuddy_mixtral_moe_7b_chat,
    'OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k',
    LoRATM.llama,
    TemplateType.openbuddy,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    hf_model_id='OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k')
@register_model(
    ModelType.openbuddy_mistral_7b_chat,
    'OpenBuddy/openbuddy-mistral-7b-v17.1-32k',
    LoRATM.llama,
    TemplateType.openbuddy,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-mistral-7b-v17.1-32k')
@register_model(
    ModelType.openbuddy_llama2_70b_chat,
    'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16',
    LoRATM.llama,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-llama2-70b-v10.1-bf16')
@register_model(
    ModelType.openbuddy_llama_65b_chat,
    'OpenBuddy/openbuddy-llama-65b-v8-bf16',
    LoRATM.llama,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-llama-65b-v8-bf16')
@register_model(
    ModelType.openbuddy_llama3_70b_chat,
    'OpenBuddy/openbuddy-llama3-70b-v21.1-8k',
    LoRATM.llama,
    TemplateType.openbuddy2,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-llama3-70b-v21.1-8k')
@register_model(
    ModelType.openbuddy_llama3_8b_chat,
    'OpenBuddy/openbuddy-llama3-8b-v21.1-8k',
    LoRATM.llama,
    TemplateType.openbuddy2,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-llama3-8b-v21.1-8k')
@register_model(
    ModelType.openbuddy_llama2_13b_chat,
    'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
    LoRATM.llama,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='OpenBuddy/openbuddy-llama2-13b-v8.1-fp16')
@register_model(
    ModelType.mistral_7b_instruct,
    'AI-ModelScope/Mistral-7B-Instruct-v0.1',
    LoRATM.llama,
    TemplateType.llama,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='mistralai/Mistral-7B-Instruct-v0.1')
@register_model(
    ModelType.mistral_7b_instruct_v2,
    'AI-ModelScope/Mistral-7B-Instruct-v0.2',
    LoRATM.llama,
    TemplateType.llama,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='mistralai/Mistral-7B-Instruct-v0.2')
@register_model(
    ModelType.mistral_7b,
    'AI-ModelScope/Mistral-7B-v0.1',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='mistralai/Mistral-7B-v0.1')
@register_model(
    ModelType.codestral_22b,
    'huangjintao/Codestral-22B-v0.1',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.34'],
    ignore_file_pattern=['consolidated.safetensors'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='mistralai/Codestral-22B-v0.1')
@register_model(
    ModelType.mistral_7b_v2,
    'AI-ModelScope/Mistral-7B-v0.2-hf',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='alpindale/Mistral-7B-v0.2-hf')
@register_model(
    ModelType.mixtral_moe_7b,
    'AI-ModelScope/Mixtral-8x7B-v0.1',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.36'],
    ignore_file_pattern=[r'.+\.pt$'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    hf_model_id='mistralai/Mixtral-8x7B-v0.1')
@register_model(
    ModelType.mixtral_moe_7b_instruct,
    'AI-ModelScope/Mixtral-8x7B-Instruct-v0.1',
    LoRATM.llama,
    TemplateType.llama,
    requires=['transformers>=4.36'],
    ignore_file_pattern=[r'.+\.pt$'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    hf_model_id='mistralai/Mixtral-8x7B-Instruct-v0.1')
@register_model(
    ModelType.mixtral_moe_8x22b_v1,
    'AI-ModelScope/Mixtral-8x22B-v0.1',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='mistral-community/Mixtral-8x22B-v0.1')
@register_model(
    ModelType.dbrx_base,
    'AI-ModelScope/dbrx-base',
    LoRATM.dbrx,
    TemplateType.dbrx,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    hf_model_id='databricks/dbrx-base')
@register_model(
    ModelType.dbrx_instruct,
    'AI-ModelScope/dbrx-instruct',
    LoRATM.dbrx,
    TemplateType.dbrx,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    hf_model_id='databricks/dbrx-instruct')
def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        torch_dtype: Dtype,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        model_config=None,
                                        **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version.parse(transformers.__version__) >= version.parse('4.36'):
        if use_flash_attn:
            model_config._attn_implementation = 'flash_attention_2'
    else:
        model_config._flash_attn_2_enabled = use_flash_attn
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.phi3_small_128k_instruct,
    'LLM-Research/Phi-3-small-128k-instruct',
    LoRATM.phi3_small,
    TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_gradient_checkpointing=False,
    support_vllm=True,
    tags=['general'],
    hf_model_id='microsoft/Phi-3-small-128k-instruct')
def get_model_tokenizer_phi3_small(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   model_config=None,
                                   **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version.parse(transformers.__version__) >= version.parse('4.36'):
        if use_flash_attn:
            model_config._attn_implementation = 'flash_attention_2'
    else:
        model_config._flash_attn_2_enabled = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)

    def rotary_emb(self, query_states, key_states, **kwargs):
        q_type = query_states.dtype
        k_type = key_states.dtype
        query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
        query_states = query_states.to(q_type)
        key_states = key_states.to(k_type)
        return query_states, key_states

    for i in range(32):
        re = model.model.layers[i].self_attn.rotary_emb
        re.rotory_emb_origin = re.forward
        re.forward = MethodType(rotary_emb, re)
    return model, tokenizer


@register_model(
    ModelType.qwen2_57b_a14b_instruct_int4,
    'qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    requires=['auto_gptq>=0.5', 'transformers>=4.40'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    hf_model_id='Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_57b_a14b_instruct,
    'qwen/Qwen2-57B-A14B-Instruct',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    requires=['transformers>=4.40'],
    hf_model_id='Qwen/Qwen2-57B-A14B-Instruct')
@register_model(
    ModelType.qwen2_0_5b_instruct_int4,
    'qwen/Qwen2-0.5B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_0_5b_instruct_int8,
    'qwen/Qwen2-0.5B-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 8},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_1_5b_instruct_int4,
    'qwen/Qwen2-1.5B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_1_5b_instruct_int8,
    'qwen/Qwen2-1.5B-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 8},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-1_5B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_7b_instruct_int4,
    'qwen/Qwen2-7B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-7B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_7b_instruct_int8,
    'qwen/Qwen2-7B-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 8},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-7B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_72b_instruct_int4,
    'qwen/Qwen2-72B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 4},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-72B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_72b_instruct_int8,
    'qwen/Qwen2-72B-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'gptq_bits': 8},
    torch_dtype=torch.float16,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-72B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_0_5b_instruct_awq,
    'qwen/Qwen2-0.5B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    torch_dtype=torch.float16,
    requires=['transformers>=4.37', 'autoawq'],
    hf_model_id='Qwen/Qwen2-0.5B-Instruct-AWQ')
@register_model(
    ModelType.qwen2_1_5b_instruct_awq,
    'qwen/Qwen2-1.5B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    torch_dtype=torch.float16,
    requires=['transformers>=4.37', 'autoawq'],
    hf_model_id='Qwen/Qwen2-1.5B-Instruct-AWQ')
@register_model(
    ModelType.qwen2_7b_instruct_awq,
    'qwen/Qwen2-7B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    torch_dtype=torch.float16,
    requires=['transformers>=4.37', 'autoawq'],
    hf_model_id='Qwen/Qwen2-7B-Instruct-AWQ')
@register_model(
    ModelType.qwen2_72b_instruct_awq,
    'qwen/Qwen2-72B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    torch_dtype=torch.float16,
    requires=['transformers>=4.37', 'autoawq'],
    hf_model_id='Qwen/Qwen2-72B-Instruct-AWQ')
@register_model(
    ModelType.qwen2_0_5b_instruct,
    'qwen/Qwen2-0.5B-Instruct',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-0.5B-Instruct')
@register_model(
    ModelType.qwen2_1_5b_instruct,
    'qwen/Qwen2-1.5B-Instruct',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-1.5B-Instruct')
@register_model(
    ModelType.qwen2_7b_instruct,
    'qwen/Qwen2-7B-Instruct',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-7B-Instruct')
@register_model(
    ModelType.qwen2_72b_instruct,
    'qwen/Qwen2-72B-Instruct',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen2-72B-Instruct')
@register_model(
    ModelType.qwen1half_0_5b_chat_awq,
    'qwen/Qwen1.5-0.5B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-0.5B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_1_8b_chat_awq,
    'qwen/Qwen1.5-1.8B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-1.8B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_4b_chat_awq,
    'qwen/Qwen1.5-4B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-4B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_7b_chat_awq,
    'qwen/Qwen1.5-7B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-7B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_14b_chat_awq,
    'qwen/Qwen1.5-14B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-14B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_32b_chat_awq,
    'qwen/Qwen1.5-32B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-32B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_72b_chat_awq,
    'qwen/Qwen1.5-72B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-72B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_110b_chat_awq,
    'qwen/Qwen1.5-110B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen1.5-110B-Chat-AWQ')
@register_model(
    ModelType.codeqwen1half_7b_chat_awq,
    'qwen/CodeQwen1.5-7B-Chat-AWQ',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    function_kwargs={'is_awq': True},
    requires=['transformers>=4.37', 'autoawq'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/CodeQwen1.5-7B-Chat-AWQ')
@register_model(
    ModelType.qwen1half_0_5b_chat,
    'qwen/Qwen1.5-0.5B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-0.5B-Chat')
@register_model(
    ModelType.qwen1half_1_8b_chat,
    'qwen/Qwen1.5-1.8B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-1.8B-Chat')
@register_model(
    ModelType.qwen1half_4b_chat,
    'qwen/Qwen1.5-4B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-4B-Chat')
@register_model(
    ModelType.qwen1half_7b_chat,
    'qwen/Qwen1.5-7B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-7B-Chat')
@register_model(
    ModelType.qwen1half_14b_chat,
    'qwen/Qwen1.5-14B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-14B-Chat')
@register_model(
    ModelType.qwen1half_32b_chat,
    'qwen/Qwen1.5-32B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-32B-Chat')
@register_model(
    ModelType.qwen1half_72b_chat,
    'qwen/Qwen1.5-72B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-72B-Chat')
@register_model(
    ModelType.qwen1half_110b_chat,
    'qwen/Qwen1.5-110B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/Qwen1.5-110B-Chat')
@register_model(
    ModelType.qwen1half_moe_a2_7b_chat,
    'qwen/Qwen1.5-MoE-A2.7B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.40'],
    hf_model_id='Qwen/Qwen1.5-MoE-A2.7B-Chat')
@register_model(
    ModelType.codeqwen1half_7b_chat,
    'qwen/CodeQwen1.5-7B-Chat',
    LoRATM.llama,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.37'],
    hf_model_id='Qwen/CodeQwen1.5-7B-Chat')
def get_model_tokenizer_qwen2_chat(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    kwargs['eos_token'] = '<|im_end|>'
    return get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


@register_model(
    ModelType.qwen1half_0_5b_chat_int4,
    'qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_0_5b_chat_int8,
    'qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_1_8b_chat_int4,
    'qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_1_8b_chat_int8,
    'qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_4b_chat_int4,
    'qwen/Qwen1.5-4B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-4B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_4b_chat_int8,
    'qwen/Qwen1.5-4B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-4B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_7b_chat_int4,
    'qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-7B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_7b_chat_int8,
    'qwen/Qwen1.5-7B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-7B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_14b_chat_int4,
    'qwen/Qwen1.5-14B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-14B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_14b_chat_int8,
    'qwen/Qwen1.5-14B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-14B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_32b_chat_int4,
    'qwen/Qwen1.5-32B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-32B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_72b_chat_int4,
    'qwen/Qwen1.5-72B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-72B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_110b_chat_int4,
    'qwen/Qwen1.5-110B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-110B-Chat-GPTQ-Int4')
@register_model(
    ModelType.qwen1half_72b_chat_int8,
    'qwen/Qwen1.5-72B-Chat-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.37'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen1.5-72B-Chat-GPTQ-Int8')
@register_model(
    ModelType.qwen1half_moe_a2_7b_chat_int4,
    'qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5', 'transformers>=4.40'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    hf_model_id='Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4')
def get_model_tokenizer_qwen2_intx(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    kwargs['get_qwen_function'] = get_model_tokenizer_qwen2_chat
    return get_model_tokenizer_qwen_intx(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


@register_model(
    ModelType.internlm2_1_8b,
    'Shanghai_AI_Laboratory/internlm2-1_8b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-1_8b')
@register_model(
    ModelType.internlm2_1_8b_sft_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-1_8b-sft')
@register_model(
    ModelType.internlm2_1_8b_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-1_8b',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-1_8b')
@register_model(
    ModelType.internlm2_math_7b,
    'Shanghai_AI_Laboratory/internlm2-math-base-7b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='internlm/internlm2-math-base-7b')
@register_model(
    ModelType.internlm2_math_20b,
    'Shanghai_AI_Laboratory/internlm2-math-base-20b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='internlm/internlm2-math-base-20b')
@register_model(
    ModelType.internlm2_math_7b_chat,
    'Shanghai_AI_Laboratory/internlm2-math-7b',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='internlm/internlm2-math-7b')
@register_model(
    ModelType.internlm2_math_20b_chat,
    'Shanghai_AI_Laboratory/internlm2-math-20b',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['math'],
    hf_model_id='internlm/internlm2-math-20b')
@register_model(
    ModelType.internlm2_7b_sft_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-7b-sft',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-7b-sft')
@register_model(
    ModelType.internlm2_7b_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-7b',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-7b')
@register_model(
    ModelType.internlm2_20b_sft_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-20b-sft',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-20b-sft')
@register_model(
    ModelType.internlm2_20b_chat,
    'Shanghai_AI_Laboratory/internlm2-chat-20b',
    LoRATM.internlm2,
    TemplateType.internlm2,
    eos_token='<|im_end|>',
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-chat-20b')
@register_model(
    ModelType.internlm2_7b,
    'Shanghai_AI_Laboratory/internlm2-7b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-7b')
@register_model(
    ModelType.internlm2_7b_base,
    'Shanghai_AI_Laboratory/internlm2-base-7b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-base-7b')
@register_model(
    ModelType.internlm2_20b,
    'Shanghai_AI_Laboratory/internlm2-20b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-20b')
@register_model(
    ModelType.internlm2_20b_base,
    'Shanghai_AI_Laboratory/internlm2-base-20b',
    LoRATM.internlm2,
    TemplateType.default_generation,
    requires=['transformers>=4.35'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='internlm/internlm2-base-20b')
def get_model_tokenizer_internlm2(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        model_config.attn_implementation = 'flash_attention_2'

    eos_token = kwargs.pop('eos_token', None)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if eos_token is not None:
        if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
            del tokenizer.__class__.eos_token_id
        tokenizer.eos_token = eos_token

    return model, tokenizer


@register_model(
    ModelType.deepseek_coder_v2_instruct,
    'deepseek-ai/DeepSeek-Coder-V2-Instruct',
    LoRATM.deepseek2,
    TemplateType.deepseek2,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Instruct')
@register_model(
    ModelType.deepseek_coder_v2_lite_instruct,
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    LoRATM.deepseek2,
    TemplateType.deepseek2,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct')
@register_model(
    ModelType.deepseek_v2_lite,
    'deepseek-ai/DeepSeek-V2-Lite',
    LoRATM.deepseek2,
    TemplateType.default_generation,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Lite')
@register_model(
    ModelType.deepseek_v2_lite_chat,
    'deepseek-ai/DeepSeek-V2-Lite-Chat',
    LoRATM.deepseek2,
    TemplateType.deepseek2,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Lite-Chat')
@register_model(
    ModelType.deepseek_v2_chat,
    'deepseek-ai/DeepSeek-V2-Chat',
    LoRATM.deepseek2,
    TemplateType.deepseek2,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Chat')
def get_model_tokenizer_deepseek2(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    model_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if model is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__
        for module in model.modules():
            if isinstance(module, mlp_cls):
                if not hasattr(module, '__old_forward'):  # Avoid double patching
                    __old_forward = module._old_forward if hasattr(module, '_old_forward') else module.forward

                    def _new_forward(hidden_states, *, __old_forward) -> Tensor:
                        dtype = hidden_states.dtype
                        return __old_forward(hidden_states).to(dtype)

                    _new_forward = partial(_new_forward, __old_forward=__old_forward)
                    if hasattr(module, '_old_forward'):  # device_map
                        module._old_forward = _new_forward
                    else:
                        module.forward = _new_forward
                    module.__old_forward = __old_forward
    return model, tokenizer


def fix_internvl_inplace_bug(model) -> None:
    embedding = model.language_model.get_input_embeddings()
    if not hasattr(embedding, '__old_forward'):  # Avoid double patching
        old_forward = embedding.forward

        @wraps(old_forward)
        def _new_forward(*args, **kwargs):
            device = args[0].device
            return old_forward(*args, **kwargs).requires_grad_(True).clone().to(device)

        embedding.__old_forward = old_forward
        embedding.forward = _new_forward


@register_model(
    ModelType.internvl_chat_v1_5,
    'AI-ModelScope/InternVL-Chat-V1-5',
    LoRATM.internlm2,
    TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/InternVL-Chat-V1-5')
@register_model(
    ModelType.internvl_chat_v1_5_int8,
    'AI-ModelScope/InternVL-Chat-V1-5-int8',
    LoRATM.internlm2,
    TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/InternVL-Chat-V1-5-int8')
@register_model(
    ModelType.mini_internvl_chat_2b_v1_5,
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
    LoRATM.internlm2,
    TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/Mini-InternVL-Chat-2B-V1-5')
@register_model(
    ModelType.mini_internvl_chat_4b_v1_5,
    'OpenGVLab/Mini-InternVL-Chat-4B-V1-5',
    LoRATM.phi3,
    TemplateType.internvl_phi3,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/Mini-InternVL-Chat-4B-V1-5')
def get_model_tokenizer_internvl(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    model_config.vision_config.use_flash_attn = use_flash_attn
    model_config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
    model_quant_config = getattr(model_config, 'quantization_config', None)

    use_bnb = False
    if model_quant_config is not None:
        use_bnb = model_quant_config.get('quant_method', None) == 'bitsandbytes'
    quantization_config = model_kwargs.get('quantization_config', None)
    if isinstance(quantization_config, BitsAndBytesConfig):
        use_bnb = True

    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=AutoModel,
        **kwargs)

    if use_bnb and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        model.config.max_position_embeddings = model.language_model.config.max_position_embeddings
        _use_submodel_func(model, 'language_model', ['get_input_embeddings', 'gradient_checkpointing_enable'])
        fix_internvl_inplace_bug(model)
        if not hasattr(model, '__old_forward'):  # Avoid double patching
            forward = model.forward
            model.__old_forward = forward

            @wraps(forward)
            def _new_forward(*args, **kwargs):
                kwargs.pop('inputs_embeds', None)
                return forward(*args, **kwargs)

            model.forward = _new_forward

        if not hasattr(model, '__old_generate'):
            generate = model.generate
            model.__old_generate = generate

            @wraps(generate)
            def _new_generate(*args, **kwargs):
                kwargs.pop('image_flags', None)
                return generate(*args, **kwargs)

            model.generate = _new_generate

        if not hasattr(model, '_old_extract_feature'):
            extract_feature = model.extract_feature
            model._old_extract_feature = extract_feature

            @wraps(extract_feature)
            def _new_extract_feature(pixel_values):
                return extract_feature(pixel_values).to(pixel_values.device).to(pixel_values.dtype)

            model.extract_feature = _new_extract_feature

        if not hasattr(model.language_model, '__old_forward'):  # Avoid double patching
            old_forward = model.language_model.forward
            model.language_model.__old_forward = old_forward

            @wraps(old_forward)
            def _new_forward(*args, **kwargs):
                input_ids = kwargs.get('input_ids', None)
                input_embeds = kwargs.get('inputs_embeds', None)
                device = input_ids.device if input_ids is not None else input_embeds.device
                output = old_forward(*args, **kwargs)
                output['logits'] = output['logits'].to(device)
                return output

            model.language_model.forward = _new_forward

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
        if not hasattr(model.config, 'hidden_size'):
            model.config.hidden_size = model.config.llm_config.hidden_size
    # fix single GPU bug
    if not hasattr(dist, '_old_get_rank'):
        get_rank = dist.get_rank

        @wraps(get_rank)
        def new_get_rank(group=None):
            if not dist.is_initialized() or dist.get_world_size() == 1:
                return -1
            return get_rank(group)

        dist.get_rank = new_get_rank
        dist._old_get_rank = get_rank
    return model, tokenizer


@register_model(
    ModelType.internlm_xcomposer2_7b_chat,
    'Shanghai_AI_Laboratory/internlm-xcomposer2-7b',
    LoRATM.internlm2,
    TemplateType.internlm_xcomposer2,
    eos_token='[UNUSED_TOKEN_145]',
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='internlm/internlm-xcomposer2-7b')
def get_model_tokenizer_internlm_xcomposer2(model_dir: str,
                                            torch_dtype: Dtype,
                                            model_kwargs: Dict[str, Any],
                                            load_model: bool = True,
                                            **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    model_config._flash_attn_2_enabled = use_flash_attn

    eos_token = kwargs.pop('eos_token', None)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if eos_token is not None:
        if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
            del tokenizer.__class__.eos_token_id
        tokenizer.eos_token = eos_token
    if model is not None:
        if use_flash_attn:
            # fix AttributeError: no attribute 'attention_dropout'
            model.model.layers[0].attention.__class__.attention_dropout = 0.

        model_cls = model.__class__
        if not hasattr(model_cls, '__old_encode_img'):  # avoid double patching
            model_cls.__old_encode_img = model_cls.encode_img

            def _new_encode_img(self, image):
                if image is None:
                    return None
                if isinstance(image, str):
                    from PIL import Image
                    image = Image.open(image).convert('RGB')
                    image = self.vis_processor(image).unsqueeze(0).to(self.device)
                else:
                    assert isinstance(image, torch.Tensor)

                img_embeds, atts_img, img_target = self.img2emb(image)
                return img_embeds.to(device=self.device)  # FIX device_map

            model_cls.encode_img = _new_encode_img

    return model, tokenizer


def _git_clone_github(github_url: str, local_repo_name: Optional[str] = None) -> str:
    git_cache_dir = os.path.join(get_cache_dir(), '_github')
    os.makedirs(git_cache_dir, exist_ok=True)
    if local_repo_name is None:
        github_url = github_url.rstrip('/')
        local_repo_name = github_url.rsplit('/', 1)[1]
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context():
        if not os.path.exists(local_repo_path):
            if not github_url.endswith('.git'):
                github_url = f'{github_url}.git'
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            command_str = f"git -C '{git_cache_dir}' clone '{github_url}' {local_repo_name}"
            logger.info(f'Run the command: `{command_str}`')
            subprocess_run(command)
        logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def __prepare_inputs_embeds(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    images_seq_mask: torch.LongTensor,
    images_emb_mask: torch.LongTensor,
    **kwargs,
):
    # for patching deepseek-vl
    from einops import rearrange
    bs, n = pixel_values.shape[0:2]
    images = rearrange(pixel_values, 'b n c h w -> (b n) c h w')
    # [b x n, T2, D]
    images_embeds = self.aligner(self.vision_model(images))

    # [b x n, T2, D] -> [b, n x T2, D]
    images_embeds = rearrange(images_embeds, '(b n) t d -> b (n t) d', b=bs, n=n)
    # [b, n, T2] -> [b, n x T2]
    images_emb_mask = rearrange(images_emb_mask, 'b n t -> b (n t)')

    # [b, T, D]
    input_ids[input_ids < 0] = 0  # ignore the image embeddings
    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    # replace with the image embeddings (FIX)
    inputs_embeds.data[images_seq_mask] = images_embeds[images_emb_mask]

    return inputs_embeds


def _use_submodel_func(model, submodel_name: str, func_list: List[str]) -> None:
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel, func_name)

        @wraps(_old_func)
        def _new_func(*args, **kwargs):
            return _old_func(*args, **kwargs)

        return _new_func

    for key in func_list:
        setattr(model, key, _get_new_func(key))


def _patch_deepseek_vl(model) -> None:
    if not hasattr(model, 'hf_device_map') or len(model.hf_device_map.values()) == 1:
        return
    if hasattr(model.language_model, '__old_forward'):
        # avoid double patching
        return
    # device_map
    __old_forward = model.language_model.forward

    def _new_forward(*args, **kwargs) -> Tensor:
        inputs = kwargs.get('inputs_embeds')
        if inputs is None:
            inputs = kwargs.get('input_ids')
        device = inputs.device
        output = __old_forward(*args, **kwargs)
        if output.logits is not None:
            output.logits = output.logits.to(device)
        if output.loss is not None:
            output.loss = output.loss.to(device)
        return output

    model.language_model.forward = _new_forward
    model.language_model.__old_forward = __old_forward


@register_model(
    ModelType.deepseek_vl_7b_chat,
    'deepseek-ai/deepseek-vl-7b-chat',
    LoRATM.llama,
    TemplateType.deepseek_vl,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    requires=['attrdict'],
    hf_model_id='deepseek-ai/deepseek-vl-7b-chat')
@register_model(
    ModelType.deepseek_vl_1_3b_chat,
    'deepseek-ai/deepseek-vl-1.3b-chat',
    LoRATM.llama,
    TemplateType.deepseek_vl,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    requires=['attrdict'],
    hf_model_id='deepseek-ai/deepseek-vl-1.3b-chat')
def get_model_tokenizer_deepseek_vl(model_dir: str,
                                    torch_dtype: Dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    # compat with python==3.10
    if sys.version_info.minor >= 10:
        import collections
        import collections.abc
        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = _git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
    sys.path.append(os.path.join(local_repo_path))
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    processor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    # flash_attn
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version.parse(transformers.__version__) >= version.parse('4.36'):
        if use_flash_attn:
            model_config.language_config._attn_implementation = 'flash_attention_2'
    else:
        model_config.language_config._flash_attn_2_enabled = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, tokenizer=tokenizer, **kwargs)
    tokenizer.processor = processor
    if load_model:
        _patch_deepseek_vl(model)
        model.prepare_inputs_embeds = MethodType(__prepare_inputs_embeds, model)
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        _use_submodel_func(model, 'language_model', func_list)
        model.generation_config = model.language_model.generation_config
    return model, tokenizer


@register_model(
    ModelType.llama3_70b_instruct_awq,
    'huangjintao/Meta-Llama-3-70B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-70B-Instruct-AWQ')
@register_model(
    ModelType.llama3_70b_instruct_int8,
    'huangjintao/Meta-Llama-3-70b-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.llama3_70b_instruct_int4,
    'huangjintao/Meta-Llama-3-70B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.llama3_8b_instruct_awq,
    'huangjintao/Meta-Llama-3-8B-Instruct-AWQ',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['autoawq'],
    torch_dtype=torch.float16,
    function_kwargs={'is_awq': True},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-8B-Instruct-AWQ')
@register_model(
    ModelType.llama3_8b_instruct_int8,
    'huangjintao/Meta-Llama-3-8B-Instruct-GPTQ-Int8',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.llama3_8b_instruct_int4,
    'huangjintao/Meta-Llama-3-8B-Instruct-GPTQ-Int4',
    LoRATM.llama,
    TemplateType.llama3,
    requires=['auto_gptq'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.llama3_70b_instruct,
    'LLM-Research/Meta-Llama-3-70B-Instruct',
    LoRATM.llama,
    TemplateType.llama3,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Meta-Llama-3-70B-Instruct')
@register_model(
    ModelType.llama3_70b,
    'LLM-Research/Meta-Llama-3-70B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Meta-Llama-3-70B')
@register_model(
    ModelType.llama3_8b_instruct,
    'LLM-Research/Meta-Llama-3-8B-Instruct',
    LoRATM.llama,
    TemplateType.llama3,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Meta-Llama-3-8B-Instruct')
@register_model(
    ModelType.llama3_8b,
    'LLM-Research/Meta-Llama-3-8B',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Meta-Llama-3-8B')
@register_model(
    ModelType.llama_3_chinese_8b,
    'ChineseAlpacaGroup/llama-3-chinese-8b',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='hfl/llama-3-chinese-8b')
@register_model(
    ModelType.llama_3_chinese_8b_instruct,
    'ChineseAlpacaGroup/llama-3-chinese-8b-instruct',
    LoRATM.llama,
    TemplateType.llama3,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='hfl/llama-3-chinese-8b-instruct')
@register_model(
    ModelType.llama2_7b_aqlm_2bit_1x16,
    'AI-ModelScope/Llama-2-7b-AQLM-2Bit-1x16-hf',
    LoRATM.llama,
    TemplateType.default_generation,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    requires=['transformers>=4.38', 'aqlm', 'torch>=2.2.0'],
    support_vllm=False,
    function_kwargs={'is_aqlm': True},
    hf_model_id='ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf')
@register_model(
    ModelType.mixtral_moe_7b_aqlm_2bit_1x16,
    'AI-ModelScope/Mixtral-8x7b-AQLM-2Bit-1x16-hf',
    LoRATM.llama,
    TemplateType.default_generation,
    requires=['transformers>=4.38', 'aqlm', 'torch>=2.2.0'],
    support_flash_attn=True,
    support_vllm=False,
    support_gradient_checkpointing=False,
    function_kwargs={'is_aqlm': True},
    hf_model_id='ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf')
@register_model(
    ModelType.llama2_7b,
    'modelscope/Llama-2-7b-ms',
    LoRATM.llama,
    TemplateType.default_generation,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-7b-hf')
@register_model(
    ModelType.llama2_13b,
    'modelscope/Llama-2-13b-ms',
    LoRATM.llama,
    TemplateType.default_generation,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-13b-hf')
@register_model(
    ModelType.llama2_70b,
    'modelscope/Llama-2-70b-ms',
    LoRATM.llama,
    TemplateType.default_generation,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-70b-hf')
@register_model(
    ModelType.llama2_7b_chat,
    'modelscope/Llama-2-7b-chat-ms',
    LoRATM.llama,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-7b-chat-hf')
@register_model(
    ModelType.llama2_13b_chat,
    'modelscope/Llama-2-13b-chat-ms',
    LoRATM.llama,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-13b-chat-hf')
@register_model(
    ModelType.llama2_70b_chat,
    'modelscope/Llama-2-70b-chat-ms',
    LoRATM.llama,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='meta-llama/Llama-2-70b-chat-hf')
def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: Dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_with_flash_attn(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.polylm_13b,
    'damo/nlp_polylm_13b_text_generation',
    LoRATM.polylm,
    TemplateType.default_generation,
    hf_model_id='DAMO-NLP-MT/polylm-13b')
def get_model_tokenizer_polylm(model_dir: str,
                               torch_dtype: Dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}


def get_model_tokenizer_qwen(model_dir: str,
                             torch_dtype: Dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if torch_dtype is not None:
        k_true = dtype_mapping[torch_dtype]
        for k in dtype_mapping.values():
            v = False
            if k == k_true:
                v = True
            setattr(model_config, k, v)

    if model_kwargs.get('quantization_config') is None or not isinstance(model_kwargs['quantization_config'],
                                                                         BitsAndBytesConfig):
        # not (quantization + bnb)
        torch_dtype = None
    use_flash_attn = kwargs.pop('use_flash_attn', None)
    if use_flash_attn is None:
        use_flash_attn = 'auto'
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer


@register_model(
    ModelType.modelscope_agent_7b,
    'iic/ModelScope-Agent-7B',
    LoRATM.qwen,
    TemplateType.modelscope_agent,
    support_flash_attn=True,
    support_vllm=False)
@register_model(
    ModelType.modelscope_agent_14b,
    'iic/ModelScope-Agent-14B',
    LoRATM.qwen,
    TemplateType.modelscope_agent,
    support_flash_attn=True,
    support_vllm=False)
@register_model(
    ModelType.codefuse_qwen_14b_chat,
    'codefuse-ai/CodeFuse-QWen-14B',
    LoRATM.qwen,
    TemplateType.codefuse,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='codefuse-ai/CodeFuse-QWen-14B')
@register_model(
    ModelType.qwen_1_8b,
    'qwen/Qwen-1_8B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-1_8B')
@register_model(
    ModelType.qwen_72b,
    'qwen/Qwen-72B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-72B')
@register_model(
    ModelType.tongyi_finance_14b,
    'TongyiFinance/Tongyi-Finance-14B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['financial'])
@register_model(
    ModelType.qwen_14b,
    'qwen/Qwen-14B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-14B')
@register_model(
    ModelType.qwen_7b,
    'qwen/Qwen-7B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-7B')
def get_model_tokenizer_qwen_base(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_qwen(*args, **kwargs)
    tokenizer.eos_token_id = tokenizer.eod_id
    return model, tokenizer


@register_model(
    ModelType.qwen_1_8b_chat,
    'qwen/Qwen-1_8B-Chat',
    LoRATM.qwen,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-1_8B-Chat')
@register_model(
    ModelType.qwen_72b_chat,
    'qwen/Qwen-72B-Chat',
    LoRATM.qwen,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-72B-Chat')
@register_model(
    ModelType.tongyi_finance_14b_chat,
    'TongyiFinance/Tongyi-Finance-14B-Chat',
    LoRATM.qwen,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    tags=['financial'],
    hf_model_id='jxy/Tongyi-Finance-14B-Chat')
@register_model(
    ModelType.qwen_14b_chat,
    'qwen/Qwen-14B-Chat',
    LoRATM.qwen,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-14B-Chat')
@register_model(
    ModelType.qwen_7b_chat,
    'qwen/Qwen-7B-Chat',
    LoRATM.qwen,
    TemplateType.qwen,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-7B-Chat')
def get_model_tokenizer_qwen_chat(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_qwen(*args, **kwargs)
    tokenizer.eos_token_id = tokenizer.im_end_id
    return model, tokenizer


def _qwen_vl_visual_block_forward(
    self,
    q_x: torch.Tensor,
    k_x: Optional[torch.Tensor] = None,
    v_x: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

    x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
    z = self.mlp(self.ln_2(x))
    x = x.to(z.device) + z  # FIX
    return x


def fix_qwen_inplace_bug(model) -> None:
    # qwen-vl, qwen-audio
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix in-place operation bug
        if not hasattr(first_drop, '__old_forward'):  # Avoid double patching
            if hasattr(first_drop, '_old_forward'):  # device_map
                __old_forward = first_drop._old_forward
                first_drop._old_forward = lambda *args, **kwargs: __old_forward(*args, **kwargs).clone()
            else:
                __old_forward = first_drop.forward
                first_drop.forward = lambda *args, **kwargs: __old_forward(*args, **kwargs).clone()
            first_drop.__old_forward = __old_forward


def _qwen_vl_audio_decode(self, *args, skip_special_tokens=False, **kwargs) -> str:
    if skip_special_tokens:
        token_ids = kwargs['token_ids']
        while len(token_ids) > 0 and token_ids[-1] in {151645, 151643}:
            token_ids.pop()
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)
    else:
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)


@register_model(
    ModelType.qwen_vl_chat,
    'qwen/Qwen-VL-Chat',
    LoRATM.qwen,
    TemplateType.qwenvl,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen-VL-Chat')
@register_model(
    ModelType.qwen_vl,
    'qwen/Qwen-VL',
    LoRATM.qwen,
    TemplateType.default_generation,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_base},
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen-VL')
def get_model_tokenizer_qwen_vl(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if (model_kwargs.get('quantization_config') is not None
            and isinstance(model_kwargs['quantization_config'], BitsAndBytesConfig)):
        # https://github.com/pytorch/pytorch/issues/58969
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['lm_head', 'attn_pool.attn']
        _TransformerBlock = get_class_from_dynamic_module('visual.TransformerBlock', model_dir)

        def _get_cast_dtype(self) -> torch.dtype:
            return self.resblocks[0].ln_1.weight.dtype

        _TransformerBlock.__old_get_cast_dtype = _TransformerBlock.get_cast_dtype
        _TransformerBlock.get_cast_dtype = _get_cast_dtype

    get_qwen_function = kwargs.pop('get_qwen_function', get_model_tokenizer_qwen_chat)
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.IMAGE_ST = ()  # fix no attr `self.IMAGE_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):  # avoid double patching
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    # fix device_map is 4
    n_gpu = torch.cuda.device_count()
    local_world_size = get_dist_setting()[3]
    if n_gpu // local_world_size >= 4:
        visual_block_cls = get_class_from_dynamic_module('visual.VisualAttentionBlock', model_dir)
        if not hasattr(visual_block_cls, '__old_forward'):  # avoid double patching
            visual_block_cls.__old_forward = visual_block_cls.forward
            visual_block_cls.forward = _qwen_vl_visual_block_forward

    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)
        # fix device_map is 4
        if n_gpu // local_world_size >= 4:
            model.transformer.visual.proj.data = model.transformer.visual.proj.to(
                model.transformer.visual.ln_post.bias.device)
        # fix images cuda:1 bug
        vision_transformer = model.transformer.visual
        if not hasattr(vision_transformer, '__old_forward'):
            _old_forward = vision_transformer.forward

            def _new_forward(x: torch.Tensor):
                return _old_forward(x).to(device=f'{x.device.type}:0')

            vision_transformer.__old_forward = _old_forward
            vision_transformer.forward = _new_forward
    return model, tokenizer


@register_model(
    ModelType.qwen_audio_chat,
    'qwen/Qwen-Audio-Chat',
    LoRATM.qwen,
    TemplateType.qwen_audio,
    support_flash_attn=True,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_chat},
    tags=['multi-modal', 'audio'],
    hf_model_id='Qwen/Qwen-Audio-Chat')
@register_model(
    ModelType.qwen_audio,
    'qwen/Qwen-Audio',
    LoRATM.qwen,
    TemplateType.qwen_audio_generation,
    support_flash_attn=True,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_base},
    tags=['multi-modal', 'audio'],
    hf_model_id='Qwen/Qwen-Audio')
def get_model_tokenizer_qwen_audio(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    get_qwen_function = kwargs.pop('get_qwen_function')
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.AUDIO_ST = ()  # fix no attr `self.AUDIO_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):  # avoid double patching
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)

    return model, tokenizer


@register_model(
    ModelType.qwen_1_8b_chat_int8,
    'qwen/Qwen-1_8B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-1_8B-Chat-Int8')
@register_model(
    ModelType.qwen_1_8b_chat_int4,
    'qwen/Qwen-1_8B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-1_8B-Chat-Int4')
@register_model(
    ModelType.qwen_72b_chat_int8,
    'qwen/Qwen-72B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-72B-Chat-Int8')
@register_model(
    ModelType.qwen_72b_chat_int4,
    'qwen/Qwen-72B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-72B-Chat-Int4')
@register_model(
    ModelType.tongyi_finance_14b_chat_int4,
    'TongyiFinance/Tongyi-Finance-14B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    tags=['financial'],
    hf_model_id='jxy/Tongyi-Finance-14B-Chat-Int4')
@register_model(
    ModelType.qwen_vl_chat_int4,
    'qwen/Qwen-VL-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={
        'get_qwen_function': get_model_tokenizer_qwen_vl,
        'gptq_bits': 4
    },
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen-VL-Chat-Int4')
@register_model(
    ModelType.qwen_14b_chat_int8,
    'qwen/Qwen-14B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-14B-Chat-Int8')
@register_model(
    ModelType.qwen_7b_chat_int8,
    'qwen/Qwen-7B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 8},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-7B-Chat-Int8')
@register_model(
    ModelType.qwen_14b_chat_int4,
    'qwen/Qwen-14B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-14B-Chat-Int4')
@register_model(
    ModelType.qwen_7b_chat_int4,
    'qwen/Qwen-7B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'gptq_bits': 4},
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='Qwen/Qwen-7B-Chat-Int4')
def get_model_tokenizer_qwen_intx(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    get_qwen_function = kwargs.pop('get_qwen_function', get_model_tokenizer_qwen_chat)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    return model, tokenizer


register_model(
    ModelType.skywork_13b,
    'skywork/Skywork-13B-base',
    LoRATM.llama,
    TemplateType.default_generation,
    get_model_tokenizer_from_repo,
    hf_model_id='Skywork/Skywork-13B-base')


@register_model(ModelType.skywork_13b_chat, 'skywork/Skywork-13B-chat', LoRATM.llama, TemplateType.skywork)
def get_skywork_model_tokenizer(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.add_tokens('[USER]')
    tokenizer.add_tokens('[BOT]')
    tokenizer.add_tokens('[SEP]')
    return model, tokenizer


@register_model(
    ModelType.codefuse_codellama_34b_chat,
    'codefuse-ai/CodeFuse-CodeLlama-34B',
    LoRATM.llama,
    TemplateType.codefuse_codellama,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding'],
    hf_model_id='codefuse-ai/CodeFuse-CodeLlama-34B')
def get_model_tokenizer_codellama(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


@register_model(
    ModelType.phi2_3b,
    'AI-ModelScope/phi-2',
    LoRATM.phi,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    tags=['coding'],
    hf_model_id='microsoft/phi-2')
@register_model(
    ModelType.telechat_12b,
    'TeleAI/TeleChat-12B',
    LoRATM.telechat,
    TemplateType.telechat,
    support_flash_attn=True,
    hf_model_id='Tele-AI/TeleChat-12B')
@register_model(
    ModelType.telechat_12b_v2,
    'TeleAI/TeleChat-12B-v2',
    LoRATM.telechat,
    TemplateType.telechat_v2,
    eos_token=2,
    support_flash_attn=True,
    hf_model_id='Tele-AI/TeleChat-12B-v2')
@register_model(
    ModelType.telechat_12b_v2_gptq_int4,
    'swift/TeleChat-12B-V2-GPTQ-Int4',
    LoRATM.telechat,
    TemplateType.telechat_v2,
    eos_token=2,
    requires=['auto_gptq>=0.5'],
    support_flash_attn=True,
    function_kwargs={'gptq_bits': 4})
def get_model_tokenizer_phi(model_dir: str,
                            torch_dtype: Dtype,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    model_config.flash_attn = use_flash_attn
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.telechat_7b,
    'TeleAI/TeleChat-7B',
    LoRATM.telechat,
    TemplateType.telechat,
    support_flash_attn=True,
    hf_model_id='Tele-AI/telechat-7B')
def get_model_tokenizer_telechat(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    if torch_dtype == torch.bfloat16:
        logger.info('telechat-7b does not support the bf16 dtype; the dtype is converted to fp16.')
        torch_dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    model_config.flash_attn = use_flash_attn
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.deepseek_moe_16b_chat,
    'deepseek-ai/deepseek-moe-16b-chat',
    LoRATM.llama,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-moe-16b-chat')
@register_model(
    ModelType.deepseek_moe_16b,
    'deepseek-ai/deepseek-moe-16b-base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='deepseek-ai/deepseek-moe-16b-base')
@register_model(
    ModelType.minicpm_moe_8x2b,
    'OpenBMB/MiniCPM-MoE-8x2B',
    LoRATM.llama,
    TemplateType.minicpm,
    requires=['transformers>=4.36.0'],
    support_flash_attn=True,
    support_vllm=True,
    hf_model_id='openbmb/MiniCPM-MoE-8x2B')
def get_model_tokenizer_deepseek_moe(model_dir: str,
                                     torch_dtype: Dtype,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__
        for module in model.modules():
            if isinstance(module, mlp_cls):
                if not hasattr(module, '__old_forward'):  # Avoid double patching
                    __old_forward = module._old_forward if hasattr(module, '_old_forward') else module.forward

                    def _new_forward(hidden_states, *, __old_forward) -> Tensor:
                        dtype = hidden_states.dtype
                        return __old_forward(hidden_states).to(dtype)

                    _new_forward = partial(_new_forward, __old_forward=__old_forward)
                    if hasattr(module, '_old_forward'):  # device_map
                        module._old_forward = _new_forward
                    else:
                        module.forward = _new_forward
                    module.__old_forward = __old_forward
    return model, tokenizer


@register_model(
    ModelType.yuan2_2b_instruct,
    'YuanLLM/Yuan2.0-2B-hf',
    LoRATM.llama,
    TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-2B-hf')
@register_model(
    ModelType.yuan2_51b_instruct,
    'YuanLLM/Yuan2.0-51B-hf',
    LoRATM.llama,
    TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-51B-hf')
@register_model(
    ModelType.yuan2_102b_instruct,
    'YuanLLM/Yuan2.0-102B-hf',
    LoRATM.llama,
    TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-102B-hf')
@register_model(
    ModelType.yuan2_2b_janus_instruct,
    'YuanLLM/Yuan2-2B-Janus-hf',
    LoRATM.llama,
    TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-2B-Janus-hf')
@register_model(
    ModelType.yuan2_m32,
    'YuanLLM/Yuan2-M32-hf',
    LoRATM.llama,
    TemplateType.yuan,
    support_gradient_checkpointing=False,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-M32-hf')
def get_model_tokenizer_yuan(model_dir: str,
                             torch_dtype: Dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_folder, model_name = os.path.split(model_dir)
    need_rename = '.' in model_name
    if need_rename:
        model_name = model_name.replace('.', '_')  # fix transformers_modules
        new_model_dir = os.path.join(model_folder, model_name)
        logger.info(f'Using new_model_dir: {new_model_dir}')
        os.rename(model_dir, new_model_dir)
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attention = kwargs.pop('use_flash_attn', False)
    model_config.use_flash_attention = use_flash_attention
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
    addi_tokens = [
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
        '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
        '<empty_output>'
    ]
    tokenizer.add_tokens(addi_tokens, special_tokens=True)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, tokenizer=tokenizer, **kwargs)
    if need_rename:
        os.rename(new_model_dir, model_dir)
    return model, tokenizer


@register_model(
    ModelType.orion_14b,
    'OrionStarAI/Orion-14B-Base',
    LoRATM.llama,
    TemplateType.default_generation,
    support_flash_attn=True,
    hf_model_id='OrionStarAI/Orion-14B-Base')
@register_model(
    ModelType.orion_14b_chat,
    'OrionStarAI/Orion-14B-Chat',
    LoRATM.llama,
    TemplateType.orion,
    support_flash_attn=True,
    ignore_file_pattern=[r'.+\.gguf$'],
    hf_model_id='OrionStarAI/Orion-14B-Chat')
def get_model_tokenizer_orion(model_dir: str,
                              torch_dtype: Dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config._flash_attn_2_enabled = kwargs.pop('use_flash_attn', False)
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.yi_vl_34b_chat,
    '01ai/Yi-VL-34B',
    LoRATM.llama,
    TemplateType.yi_vl,
    support_flash_attn=True,
    requires=['transformers>=4.34'],
    tags=['multi-modal', 'vision'],
    hf_model_id='01-ai/Yi-VL-34B')
@register_model(
    ModelType.yi_vl_6b_chat,
    '01ai/Yi-VL-6B',
    LoRATM.llama,
    TemplateType.yi_vl,
    support_flash_attn=True,
    requires=['transformers>=4.34'],
    tags=['multi-modal', 'vision'],
    hf_model_id='01-ai/Yi-VL-6B')
def get_model_tokenizer_yi_vl(model_dir: str,
                              torch_dtype: Dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = _git_clone_github('https://github.com/01-ai/Yi')
    sys.path.append(os.path.join(local_repo_path, 'VL'))
    from llava.model import LlavaLlamaForCausalLM, LlavaConfig
    from llava.model.constants import key_info

    model_config = LlavaConfig.from_pretrained(model_dir)
    mm_vision_tower = model_config.mm_vision_tower
    model_config.mm_vision_tower = os.path.join(model_dir, *mm_vision_tower.rsplit('/', maxsplit=2)[-2:])
    model_config.attention_dropout = 0.
    key_info['model_path'] = model_dir
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=LlavaLlamaForCausalLM,
        **kwargs)
    if model is not None:
        logger.info('Please ignore the above warning.')
        logger.info('Loading the parameters of vision_tower...')
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=model.device, dtype=torch_dtype)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
    return model, tokenizer


def _patch_minicpm_v_device_map(model) -> None:
    if not hasattr(model, 'hf_device_map') or len(model.hf_device_map.values()) == 1:
        return
    if hasattr(model.llm, '__old_forward'):
        # avoid double patching
        return
    device = list(model.hf_device_map.values())[0]
    if hasattr(model, 'get_vision_embedding'):  # minicpm-v-v2-chat
        _old_get_vision_embedding = model.get_vision_embedding

        def _get_vision_embedding(pixel_values):
            if len(pixel_values) == 0:
                return _old_get_vision_embedding(pixel_values)
            output = _old_get_vision_embedding(pixel_values)
            return output.to(device=device)

        model._old_get_vision_embedding = _old_get_vision_embedding
        model.get_vision_embedding = _get_vision_embedding

    if hasattr(model, 'resampler'):  # minicpm-v-v2_5-chat
        __old_resampler_forward = model.resampler.forward

        def _new_resampler_forward(*args, **kwargs) -> Tensor:
            output = __old_resampler_forward(*args, **kwargs)
            return output.to(device=device)

        model.resampler.forward = _new_resampler_forward

    __old_forward = model.llm.forward

    def _new_forward(*args, **kwargs) -> Tensor:
        inputs = kwargs.get('inputs_embeds')
        if inputs is None:
            inputs = kwargs.get('input_ids')
        device = inputs.device
        output = __old_forward(*args, **kwargs)
        if output.logits is not None:
            output.logits = output.logits.to(device)
        if output.loss is not None:
            output.loss = output.loss.to(device)
        return output

    model.llm.forward = _new_forward
    model.llm.__old_forward = __old_forward


@register_model(
    ModelType.minicpm_v_3b_chat,
    'OpenBMB/MiniCPM-V',
    LoRATM.llama,
    TemplateType.minicpm_v,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-V')
@register_model(
    ModelType.minicpm_v_v2_chat,
    'OpenBMB/MiniCPM-V-2',
    LoRATM.llama,
    TemplateType.minicpm_v,
    support_flash_attn=True,
    requires=['timm'],
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-V-2')
@register_model(
    ModelType.minicpm_v_v2_5_chat,
    'OpenBMB/MiniCPM-Llama3-V-2_5',
    LoRATM.llama,
    TemplateType.minicpm_v_v2_5,
    support_flash_attn=True,
    requires=['timm'],
    placeholder_tokens=['<unk>'],
    function_kwargs={'patching_embedding': True},
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-Llama3-V-2_5')
def get_model_tokenizer_minicpm_v(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    patching_embedding = kwargs.pop('patching_embedding', False)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if load_model:
        model.resampler.to(torch_dtype)  # fix float32
        _patch_minicpm_v_device_map(model)
        func_list = ['generate', 'get_input_embeddings', 'forward']
        _use_submodel_func(model, 'llm', func_list)
        if patching_embedding:
            embedding = model.get_input_embeddings()
            if not hasattr(embedding, '__old_forward'):  # Avoid double patching
                old_forward = embedding.forward

                @wraps(old_forward)
                def _new_forward(*args, **kwargs):
                    return old_forward(*args, **kwargs).requires_grad_(True).clone()

                embedding.__old_forward = old_forward
                embedding.forward = _new_forward
    return model, tokenizer


def _patch_llava(model):
    if hasattr(model, '__old_generate'):
        return
    generate = model.generate
    model.__old_generate = generate

    @wraps(generate)
    def _new_generate(inputs=None, *args, **kwargs):
        input_ids = kwargs.pop('input_ids', None)
        if inputs is None and input_ids is not None:
            inputs = input_ids
        return generate(inputs, *args, **kwargs)

    model.generate = _new_generate


@register_model(
    ModelType.llava1_6_yi_34b_instruct,
    'AI-ModelScope/llava-v1.6-34b',
    LoRATM.llama,
    TemplateType.llava_yi_instruct,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    function_kwargs={'llm_model_type': 'llama'},
    tags=['multi-modal', 'vision'],
    hf_model_id='liuhaotian/llava-v1.6-34b')
@register_model(
    ModelType.llava1_6_mistral_7b_instruct,
    'AI-ModelScope/llava-v1.6-mistral-7b',
    LoRATM.llama,
    TemplateType.llava_mistral_instruct,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    function_kwargs={'llm_model_type': 'mistral'},
    tags=['multi-modal', 'vision'],
    hf_model_id='liuhaotian/llava-v1.6-mistral-7b')
@register_model(
    ModelType.llama3_llava_next_8b,
    'AI-Modelscope/llama3-llava-next-8b',
    LoRATM.llama,
    TemplateType.llama_llava_next,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_llama'},
    hf_model_id='lmms-lab/llama3-llava-next-8b')
@register_model(
    ModelType.llava_next_72b,
    'AI-Modelscope/llava-next-72b',
    LoRATM.llama,
    TemplateType.llava_qwen_instruct,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_qwen'},
    hf_model_id='lmms-lab/llava-next-72b')
@register_model(
    ModelType.llava_next_110b,
    'AI-Modelscope/llava-next-110b',
    LoRATM.llama,
    TemplateType.llava_qwen_instruct,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_qwen'},
    hf_model_id='lmms-lab/llava-next-110b')
def get_model_tokenizer_llava(model_dir: str,
                              torch_dtype: Dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    llm_model_type = kwargs.pop('llm_model_type')
    if 'local_repo_path' in kwargs:
        repo_path = kwargs['local_repo_path']
    elif 'next' in llm_model_type:
        repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
    else:
        repo_path = 'https://github.com/haotian-liu/LLaVA'
    local_repo_path = _git_clone_github(repo_path)
    sys.path.append(os.path.join(local_repo_path))

    if llm_model_type == 'mistral':
        from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
        model_config = LlavaMistralConfig.from_pretrained(model_dir)
        automodel_class = LlavaMistralForCausalLM
    elif 'llama' in llm_model_type:  # llama
        from llava.model import LlavaLlamaForCausalLM, LlavaConfig
        if not hasattr(LlavaLlamaForCausalLM, '__old_forward'):  # Avoid double patching
            forward = LlavaLlamaForCausalLM.forward
            LlavaLlamaForCausalLM.__old_forward = forward

            @wraps(forward)
            def _new_forward(*args, **kwargs):
                kwargs.pop('cache_position', None)
                return forward(*args, **kwargs)

            LlavaLlamaForCausalLM.forward = _new_forward
        model_config = LlavaConfig.from_pretrained(model_dir)
        automodel_class = LlavaLlamaForCausalLM
    else:  # qwen
        from llava.model import LlavaQwenForCausalLM
        automodel_class = LlavaQwenForCausalLM
        model_config = AutoConfig.from_pretrained(model_dir)

    model_config.mm_vision_tower = snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=automodel_class,
        **kwargs)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        device_map = str(model_kwargs.get('device_map', str(model.device)))
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
        _patch_llava(model)
    return model, tokenizer


@register_model(
    ModelType.mplug_owl2_chat,
    'iic/mPLUG-Owl2',
    LoRATM.mplug_owl2,
    TemplateType.mplug_owl2,
    requires=['transformers<4.35', 'icecream'],
    eos_token='</s>',
    function_kwargs={'get_model_tokenizer_function': get_model_tokenizer_with_flash_attn},
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='MAGAer13/mplug-owl2-llama2-7b')
@register_model(
    ModelType.mplug_owl2_1_chat,
    'iic/mPLUG-Owl2.1',
    LoRATM.mplug_owl2_1,
    TemplateType.mplug_owl2,
    requires=['transformers<4.35', 'icecream'],
    eos_token='<|endoftext|>',
    function_kwargs={
        'vocab_size': 151851,
        'get_model_tokenizer_function': get_model_tokenizer_qwen
    },
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Mizukiluke/mplug_owl_2_1')
def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = _git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(os.path.join(local_repo_path))

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from mplug_owl2 import MPLUGOwl2LlamaForCausalLM
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function')
    model, tokenizer = get_model_tokenizer_function(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer.processor = processor
    return model, tokenizer


def fix_transformers_upgrade(module: PreTrainedModel) -> None:
    # from 4.35, transformers changes its arguments of _set_gradient_checkpointing
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        if isinstance(module, PreTrainedModel) and hasattr(module, '_set_gradient_checkpointing') \
                and 'value' in inspect.signature(module._set_gradient_checkpointing).parameters.keys():
            module._set_gradient_checkpointing = MethodType(PreTrainedModel._set_gradient_checkpointing, module)


def fix_gradient_checkpointing_warning() -> None:
    torch_version = version.parse(torch.__version__)
    if torch_version < version.parse('2'):
        return
    elif torch_version < version.parse('2.1'):
        # fix https://github.com/Dao-AILab/flash-attention/issues/341
        use_reentrant = True
    else:
        use_reentrant = False
    _old_checkpoint = torch.utils.checkpoint.checkpoint
    if not hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = update_wrapper(
            lambda *args, use_reentrant=use_reentrant, **kwargs: _old_checkpoint(
                *args, use_reentrant=use_reentrant, **kwargs),
            _old_checkpoint)
    try:
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, 'checkpoint'):
            transformers.modeling_utils.checkpoint = (lambda *args, use_reentrant=use_reentrant, **kwargs:
                                                      _old_checkpoint(*args, use_reentrant=use_reentrant, **kwargs))
    except ImportError:
        pass


def safe_snapshot_download(model_type: str,
                           model_id_or_path: Optional[str] = None,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           **kwargs) -> str:
    # Perform snapshot_download (ms or hf) based on model_type and model_id_or_path.
    model_info = MODEL_MAPPING[model_type]
    use_hf = strtobool(os.environ.get('USE_HF', 'False'))
    if model_id_or_path is None:
        model_dir = kwargs.pop('model_dir', None)  # compat with swift<1.7
        if model_dir is not None:
            model_id_or_path = model_dir
        else:
            model_id_or_path = model_info['hf_model_id' if use_hf else 'model_id_or_path']

    with safe_ddp_context():
        if model_id_or_path is not None and not os.path.exists(model_id_or_path):
            ignore_file_pattern = model_info['ignore_file_pattern']
            if download_model is False:
                if ignore_file_pattern is None:
                    ignore_file_pattern = []
                if use_hf:
                    ignore_file_pattern += ['*.bin', '*.safetensors']
                else:
                    ignore_file_pattern += [r'.+\.bin$', r'.+\.safetensors$']
            if use_hf:
                if revision is None or revision == 'master':
                    revision = 'main'
                logger.info(f'Downloading the model from HuggingFace Hub, model_id: {model_id_or_path}')
                use_hf_transfer = strtobool(os.environ.get('USE_HF_TRANSFER', 'False'))
                if use_hf_transfer:
                    import huggingface_hub._snapshot_download as hf_s
                    hf_s.HF_HUB_ENABLE_HF_TRANSFER = True
                from huggingface_hub import snapshot_download as hf_snapshot_download
                model_dir = hf_snapshot_download(
                    model_id_or_path, repo_type='model', revision=revision, ignore_patterns=ignore_file_pattern)
            else:
                if revision is None:
                    revision = model_info['revision']
                logger.info(f'Downloading the model from ModelScope Hub, model_id: {model_id_or_path}')
                model_dir = snapshot_download(model_id_or_path, revision, ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = model_id_or_path
        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.expanduser(model_dir)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def get_torch_dtype(model_dir: str) -> Dtype:
    model_config = PretrainedConfig.get_config_dict(model_dir)[0]
    torch_dtype = model_config.get('torch_dtype', None)
    if isinstance(torch_dtype, str):
        torch_dtype = eval(f'torch.{torch_dtype}')
    if torch_dtype in {torch.float32, None}:
        torch_dtype = torch.float16
    return torch_dtype


def get_model_tokenizer(model_type: str,
                        torch_dtype: Optional[Dtype] = None,
                        model_kwargs: Optional[Dict[str, Any]] = None,
                        load_model: bool = True,
                        *,
                        model_id_or_path: Optional[str] = None,
                        revision: Optional[str] = None,
                        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    torch_dtype: If you use None, it will retrieve the torch_dtype from the config.json file.
        However, if torch.float32 is retrieved, torch.float16 will be used.
    """
    model_dir = kwargs.pop('model_dir', None)  # compat with swift<1.7
    download_model = kwargs.pop('download_model', load_model)
    model_dir = safe_snapshot_download(
        model_type, model_id_or_path, revision=revision, download_model=download_model, model_dir=model_dir)

    model_info = MODEL_MAPPING[model_type]
    requires = model_info['requires']
    for require in requires:
        require_version(require)
    get_function = model_info['get_function']
    if model_kwargs is None:
        model_kwargs = {}
    if 'device_map' not in model_kwargs and not use_torchacc():
        model_kwargs['device_map'] = 'auto'

    if load_model:
        if model_info.get('torch_dtype') is not None:
            model_torch_dtype = model_info['torch_dtype']
            if torch_dtype is None:
                torch_dtype = model_torch_dtype
                logger.info(f'Setting torch_dtype: {torch_dtype}')
            else:
                assert torch_dtype == model_torch_dtype, f'please use `{model_torch_dtype}`'
        else:
            if torch_dtype is None:
                torch_dtype = get_torch_dtype(model_dir)
                logger.info(f'Setting torch_dtype: {torch_dtype}')
                quantization_config = model_kwargs.get('quantization_config')
                if (isinstance(quantization_config, BitsAndBytesConfig)
                        and quantization_config.bnb_4bit_compute_dtype is None):
                    quantization_config.bnb_4bit_compute_dtype = torch_dtype
                    logger.info(f'Setting quantization_config.bnb_4bit_compute_dtype: {torch_dtype}')
    kwargs['eos_token'] = model_info['eos_token']
    pad_token = model_info.get('pad_token')
    if pad_token is not None:
        kwargs['pad_token'] = pad_token
    placeholder_tokens = model_info.get('placeholder_tokens')
    if placeholder_tokens is not None:
        kwargs['placeholder_tokens'] = placeholder_tokens
    if 'is_training' not in kwargs:
        kwargs['is_training'] = False
    model, tokenizer = get_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        model.max_model_len = get_max_model_len(model.config)
        logger.info(f'model.max_model_len: {model.max_model_len}')
        model.model_type = model_type
        model.model_dir = model_dir
        fix_transformers_upgrade(model)
    fix_gradient_checkpointing_warning()
    tokenizer.model_type = model_type
    tokenizer.model_dir = model_dir
    assert tokenizer.eos_token is not None, 'tokenizer.eos_token has not been set.'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and model_dir is not None:
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        generation_config = getattr(model, 'generation_config', None)
        if os.path.isfile(generation_config_path) and generation_config is None:
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        generation_config = getattr(model, 'generation_config', None)
        # fix llama2 bug
        if (generation_config is not None and 0 < generation_config.temperature < 1
                and generation_config.do_sample is False):
            model.generation_config.do_sample = True
            logger.warning('Setting model.generation_config.do_sample: True')
    return model, tokenizer


def get_additional_saved_files(model_type: str) -> List[str]:
    files_mapping = {'qwen-vl': ['SimSun.ttf'], 'qwen-audio': ['mel_filters.npz'], 'yi-vl': ['vit']}
    for key, files_list in files_mapping.items():
        if key in model_type:
            return files_list
    return []


def get_default_template_type(model_type: str) -> Optional[str]:
    return MODEL_MAPPING[model_type].get('template')


def get_default_lora_target_modules(model_type: str) -> Optional[List[str]]:
    return MODEL_MAPPING[model_type].get('lora_target_modules')
