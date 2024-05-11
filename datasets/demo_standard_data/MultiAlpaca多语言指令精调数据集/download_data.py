import sys, os

sys.path.join(os.getcwd())

from modelscope.msdatasets import MsDataset

# ds =  MsDataset.load('damo/nlp_polylm_multialpaca_sft', subset_name='ja', split='train')
# print(next(iter(ds)))


# load data
ds =  MsDataset.load('jpdata-b0b2f75e06d2a09037eae53d7e07938d', subset_name='ja', split='train')
print(next(iter(ds)))