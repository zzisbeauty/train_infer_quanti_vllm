import os

import gradio as gr
from packaging import version

from swift.ui.base import all_langs
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain

lang = os.environ.get('SWIFT_UI_LANG', all_langs[0])

locale_dict = {
    'title': {
        'zh': '🚀SWIFT: 轻量级大模型训练推理框架',
        'en': '🚀SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning and Inference'
    },
    'sub_title': {
        'zh':
        '请查看 <a href=\"https://github.com/modelscope/swift/tree/main/docs/source\" target=\"_blank\">'
        'SWIFT 文档</a>来查看更多功能',
        'en':
        'Please check <a href=\"https://github.com/modelscope/swift/tree/main/docs/source_en\" target=\"_blank\">'
        'SWIFT Documentation</a> for more usages',
    },
    'star_beggar': {
        'zh':
        '喜欢<a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>就动动手指给我们加个star吧🥺 ',
        'en':
        'If you like <a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>, '
        'please take a few seconds to star us🥺 '
    },
}

is_spaces = True if 'SPACE_ID' in os.environ else False
if is_spaces:
    is_shared_ui = True if 'modelscope/swift' in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False


def run_ui():
    LLMTrain.set_lang(lang)
    LLMInfer.set_lang(lang)
    with gr.Blocks(title='SWIFT WebUI') as app:
        gr.HTML(f"<h1><center>{locale_dict['title'][lang]}</center></h1>")
        gr.HTML(f"<h3><center>{locale_dict['sub_title'][lang]}</center></h3>")
        gr.HTML(f"<h3><center>{locale_dict['star_beggar'][lang]}</center></h3>")
        if is_shared_ui:
            gr.HTML(
                f'<div class="gr-prose" style="max-width: 80%"><p>If the waiting queue is too long, you can either run locally or duplicate the Space and run it on your own profile using a (paid) private A10G-large GPU for training. A A10G-large costs US$3.15/h. &nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{os.environ["SPACE_ID"]}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p></div>'  # noqa
            )
        with gr.Tabs():
            if is_shared_ui:
                LLMInfer.build_ui(LLMInfer)
                LLMTrain.build_ui(LLMTrain)
            else:
                LLMTrain.build_ui(LLMTrain)
                LLMInfer.build_ui(LLMInfer)

    port = os.environ.get('WEBUI_PORT', None)
    concurrent = {}
    if version.parse(gr.__version__) < version.parse('4.0.0') and os.environ.get('MODELSCOPE_ENVIRONMENT') != 'studio':
        concurrent = {'concurrency_count': 5}
    app.queue(**concurrent).launch(
        server_name=os.environ.get('WEBUI_SERVER', None),
        server_port=port if port is None else int(port),
        height=800,
        share=bool(int(os.environ.get('WEBUI_SHARE', '0'))))
