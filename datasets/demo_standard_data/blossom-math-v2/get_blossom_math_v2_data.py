from modelscope import MsDataset

dataset = MsDataset.load('AI-ModelScope/blossom-math-v2').to_hf_dataset()
print(dataset)