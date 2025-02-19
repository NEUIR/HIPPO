# dpo training
export NCCL_P2P_DISABLE=1
nproc_per_node=2
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
--rlhf_type dpo \
--model #model_path \
--train_type lora \
--dataset # dpo_data \
--torch_dtype bfloat16 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-4 \
--lora_rank 8 \
--lora_alpha 32 \
--target_modules all-linear \
--gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
--eval_steps 100 \
--save_steps 100 \
--save_total_limit 5 \
--logging_steps 5 \
--max_length 5000 \
--output_dir # output_dir \
--add_version False
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--deepspeed zero2


# merge lora
CUDA_VISIBLE_DEVICES=0 swift export \
--ckpt_dir= #output_dir \
--merge_lora True