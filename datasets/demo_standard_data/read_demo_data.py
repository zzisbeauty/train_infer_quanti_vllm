import os,sys
import jsonlines

proPath = os.getcwd()
sys.path.append(proPath)

filePath = os.path.join(proPath, "datasets/demo_standard_data/giikin_jsonl_data_demo_standard_data/chatglm.jsonl")
filePath = '/home/train_infer_quanti_vllm/datasets/demo_standard_data/giikin_jsonl_data_demo_standard_data/chatgml.jsonl'
with open(filePath, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        print(item)