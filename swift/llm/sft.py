# Copyright (c) Alibaba, Inc. and its affiliates.

# 执行 swift sft 核心逻辑；被API调用的代码
import os,sys

proPath = os.getcwd()
sys.path.append(proPath)

from functools import partial
from typing import Any, Dict, Union

import json
import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available

from swift.trainers import Seq2SeqTrainer
from swift.trainers.utils import can_return_loss, find_labels
from swift.utils import (check_json_format, compute_acc_metrics, compute_nlg_metrics, get_dist_setting, get_logger,
                         get_main, get_model_info, is_ddp_plus_mp, is_dist, is_master, plot_images,
                         preprocess_logits_for_metrics, seed_everything, show_layers, use_torchacc)
from .accelerator import ta_accelerate
from .tuner import prepare_model
from .utils import (TEMPLATE_MAPPING, LazyLLMDataset, SftArguments, Template, dataset_map, get_dataset,
                    get_model_tokenizer, get_template, get_time_info, print_example, set_generation_config,
                    sort_by_max_length, stat_dataset)

logger = get_logger()


def llm_sft(args: SftArguments) -> Dict[str, Union[str, Any]]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    training_args = args.training_args
    if is_torch_npu_available():
        print(f'device_count: {torch.npu.device_count()}')
    else:
        print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, ' f'world_size: {world_size}, local_world_size: {local_world_size}')

    if args.gpu_memory_fraction is not None:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(max(min(args.gpu_memory_fraction, 1.0), 0.01), device=device_id)

    # Loading Model and Tokenizer
    if is_deepspeed_zero3_enabled():
        model_kwargs = {'device_map': None}
    elif is_torch_npu_available():
        model_kwargs = {'device_map': local_rank if local_rank >= 0 else 0}
    else:
        model_kwargs = {'low_cpu_mem_usage': True}
        if is_dist() and not is_ddp_plus_mp():
            model_kwargs['device_map'] = {'': local_rank}
        elif not use_torchacc():
            model_kwargs['device_map'] = 'auto'

    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config

    kwargs = {
        'max_length': args.max_length,
        'use_unsloth': args.tuner_backend == 'unsloth',
        'load_in_4bit': args.quantization_bit == 4
    }
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=args.model_id_or_path,
        revision=args.model_revision,
        is_training=True,
        **kwargs)
    logger.info(f'model_config: {model.config}')
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    set_generation_config(model, generation_config)
    training_args.generation_config = generation_config

    if use_torchacc():
        import torchacc as ta
        # Get `label` and `return_loss` before 'ta_accelerate' because it will
        # wrapper the model and make these properties wrong.
        label_names = find_labels(model)
        return_loss = can_return_loss(model)
        model = ta.patch_qwen_model(model)
    # Preparing LoRA
    model, callbacks = prepare_model(model, args)

    show_layers(model)
    logger.info(model)
    model_info = None
    if not is_deepspeed_zero3_enabled():
        model_info = get_model_info(model)
        logger.info(model_info)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
        model.enable_input_require_grads()

    if use_torchacc():
        model.config.use_cache = False
        logger.info('Setting model.config.use_cache: False')
        model = ta_accelerate(
            model,
            world_size,
            args.model_layer_cls_name,
            args.bf16,
            args.fp16,
            gradient_checkpointing=True,
            fsdp_flatten_parameters=False)

    # Loading Dataset
    train_dataset, val_dataset = get_dataset(
        args.dataset,
        args.dataset_test_ratio,
        args.dataset_seed,
        check_dataset_strategy=args.check_dataset_strategy,
        model_name=args.model_name,
        model_author=args.model_author)
    train_dataset, val_dataset = args._handle_dataset_compat(train_dataset, val_dataset)
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    template_kwargs = {}
    template_info = TEMPLATE_MAPPING[args.template_type]
    use_model = template_info.get('use_model', False)
    if use_model:
        template_kwargs['model'] = model
    template_kwargs['use_loss_scale'] = args.use_loss_scale
    template: Template = get_template(args.template_type, tokenizer, args.system, args.max_length,
                                      args.truncation_strategy, **template_kwargs)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    logger.info(f'args.lazy_tokenize: {args.lazy_tokenize}')
    if args.packing:
        from swift.llm.utils.utils import ConstantLengthDataset
        train_dataset = ConstantLengthDataset.get_packed_dataset(
            template, train_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
        if val_dataset is not None:
            val_dataset = ConstantLengthDataset.get_packed_dataset(
                template, val_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
        dataset_info = {}
        if not args.lazy_tokenize:
            dataset_info['train_dataset'] = stat_dataset(train_dataset)
            if val_dataset is not None:
                dataset_info['val_dataset'] = stat_dataset(val_dataset)
    elif not args.lazy_tokenize:
        dataset_info = {}
        logger.info(f'Using num_proc: {args.preprocess_num_proc}')
        train_dataset = dataset_map(train_dataset, template.encode, args.preprocess_num_proc)
        if val_dataset is not None:
            val_dataset = dataset_map(val_dataset, template.encode, args.preprocess_num_proc)
        if args.test_oom_error:
            train_dataset = sort_by_max_length(train_dataset, 20000)
        # Data analysis
        if train_dataset is None:
            logger.error('Error accessing train_dataset properties. '
                         'Please ensure that the dataset is properly initialized,'
                         'and every sample of the train_dataset not empty.')
            raise AttributeError('Failed to access dataset attributes,train_dataset is None. This might be because:\n'
                                 '(1) The dataset contains None for input or labels;\n'
                                 "(2) The 'max_length' setting is too short causing data truncation.")
        td0, tkwargs0 = train_dataset.data[0]
        print_example(td0, tokenizer, tkwargs0)
        dataset_info['train_dataset'] = stat_dataset(train_dataset)
        if val_dataset is not None:
            dataset_info['val_dataset'] = stat_dataset(val_dataset)
    else:
        dataset_info = None
        td0, tkwargs0 = template.encode(train_dataset[0])
        print_example(td0, tokenizer, tkwargs0)
        train_dataset = LazyLLMDataset(train_dataset, template)
        if val_dataset is not None:
            val_dataset = LazyLLMDataset(val_dataset, template)
    if val_dataset is None:
        training_args.evaluation_strategy = IntervalStrategy.NO
        training_args.do_eval = False

    padding_to = args.max_length if args.sft_type == 'longlora' else None
    data_collator = partial(template.data_collator, padding_to=padding_to)

    trian_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    if use_torchacc():
        trian_batch_size *= world_size
        eval_batch_size *= world_size
    training_args.per_device_train_batch_size = trian_batch_size
    training_args.per_device_eval_batch_size = eval_batch_size
    training_args.group_by_length = use_torchacc()

    # Trainer
    logger.info(f'training_args: {training_args}')

    trainer_kwargs = {}
    if args.predict_with_generate:
        trainer_kwargs['compute_metrics'] = partial(compute_nlg_metrics, tokenizer=tokenizer)
    else:
        compute_metrics = partial(compute_acc_metrics, acc_strategy=args.acc_strategy)
        trainer_kwargs['compute_metrics'] = compute_metrics
        trainer_kwargs['preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs)
    trainer.sft_args = args
    if use_torchacc():
        trainer.label_names = label_names
        trainer.can_return_loss = return_loss
    if is_master():
        for args_obj, fname in zip([args, training_args], ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args_obj.__dict__), f, ensure_ascii=False, indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
    trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint', None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    train_time = get_time_info(trainer.state.log_history, len(train_dataset))
    # Visualization
    if is_master() and not use_torchacc():
        if 'tensorboard' in args.training_args.report_to:
            images_dir = os.path.join(args.output_dir, 'images')
            logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignore(['images/'])
            trainer.push_to_hub()
    run_info = {
        'memory': trainer.perf['memory'],
        'gen_time': trainer.perf['gen_time'],
        'gen_len': trainer.perf['gen_len'],
        'train_time': train_time,
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        'model_info': model_info,
        'dataset_info': dataset_info,
    }
    jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(run_info) + '\n')
    return run_info


def get_sft_main(args, llm):
    if use_torchacc():
        logger.warning('TorchAcc is currently only available internally ' 'within Alibaba Cloud.')
        import torchacc as ta
        # This patch should be called before `llm_sft`.
        ta.accelerate_hf_trainer()
    return get_main(args, llm)


sft_main = get_sft_main(SftArguments, llm_sft)
