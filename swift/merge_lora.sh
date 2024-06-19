CUDA_VISIBLE_DEVICES=3 swift export --ckpt_dir '/home/train_infer_quanti_vllm/untrackfiles/output/chatglm3-6b-32k/v0-20240606-102752/checkpoint-1060' --merge_lora true
CUDA_VISIBLE_DEVICES=3 swift export --ckpt_dir '/home/train_infer_quanti_vllm/untrackfiles/output-old-glm3/chatglm3-6b-32k/v0-20240608-072736/checkpoint-1060' --merge_lora true
