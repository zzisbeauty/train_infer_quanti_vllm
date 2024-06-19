from typing import Any, Dict, List, Optional

import torch
from transformers import trainer
from trl import KTOTrainer as HFKTOTrainer

from swift.llm.utils.template import Context, History, Template
from swift.llm.utils.utils import sort_by_max_length
from swift.utils import get_logger
from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
from .mixin import PushToMsHubMixin, SwiftMixin

logger = get_logger()


def encode_batch(batch: Dict[str, List[Any]], template: Template):
    """
    Encode a batch from KTO specific dataset with given template

    Args:
    batch: A dictionary containing:
        - prompt: The main prompt string
        - completion: The completion string
        - label: The label data
        - history (optional): A list of historical queries/responses
        - system (optional): A system string to use

    template: swift Template object

    Returns:
    A dictionary with encoded prompt, completion, and label.
    """

    query: Optional[str] = batch.get('prompt', None)
    history: Optional[History] = batch.get('history', None)
    system: Optional[str] = batch.get('system', None)
    if history is None:
        history = []
    if system is None:
        if template.use_default_system:
            system = template.default_system
    else:
        assert template.system_prefix is not None, 'not support `system`'

    res_context_list: List[Context] = []
    compute_loss_idx: List[float] = []

    if system is None:
        assert template.prefix != template.system_prefix, f'template.prefix: {template.prefix}'
        prefix = template.prefix
    else:
        prefix = template.system_prefix

    template._concat_context_list(prefix, res_context_list, compute_loss_idx, system=system)

    for i, (q, r) in enumerate(history):
        template._concat_context_list([*template.prompt, '{{RESPONSE}}', *template.chat_sep],
                                      res_context_list,
                                      compute_loss_idx,
                                      query=q,
                                      response=r,
                                      round0=i)
    template._concat_context_list(template.prompt, res_context_list, compute_loss_idx, query=query, round0=len(history))
    res_context_list, compute_loss_idx = template._simplify_context_list(res_context_list, compute_loss_idx)
    prompt = ''.join(res_context_list)

    return {'prompt': prompt, 'completion': batch['completion'], 'label': batch['label']}


class KTOTrainer(PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self, *args, template: Template, test_oom_error=False, **kwargs):
        eval_dataset = kwargs.get('eval_dataset', None)
        kwargs['train_dataset'] = kwargs['train_dataset'].map(
            encode_batch,
            fn_kwargs={'template': template},
            desc='Encode dataset with template',
        )
        if eval_dataset is not None:
            kwargs['eval_dataset'] = eval_dataset.map(
                encode_batch,
                fn_kwargs={'template': template},
                desc='Encode dataset with template',
            )
        super().__init__(*args, **kwargs)
        train_ds_info = self.stat_dataset(self.train_dataset)
        val_ds_info = self.stat_dataset(self.eval_dataset)
        self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time': 0.,
            'gen_len': 0,
            'memory': {},
            'model': self.model.get_trainable_parameters() if hasattr(self.model, 'get_trainable_parameters') else None,
        }

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

    @staticmethod
    def stat_dataset(llm_dataset) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            prompt_input_ids = llm_dataset['prompt_input_ids']
            answer_input_ids = llm_dataset['answer_input_ids']
            for pi, ai in zip(prompt_input_ids, answer_input_ids):
                _token_len.append(len(pi) + len(ai))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew
