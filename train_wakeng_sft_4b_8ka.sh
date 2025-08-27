
export PDSH_RCMD_TYPE=ssh
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4 export NCCL_IB_TC=160
export GPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

accelerate launch --config_file platform_default_config.yaml.8core_multi src/train_bash.py \
    --do_train \
    --do_eval \
    --stage sft \
    --finetuning_type full \
    --template qwen3 \
    --model_name_or_path /home/huggingface_cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea \
    --dataset_dir data/wawa_sft/rawdata \
    --tokenized_path data/wawa_sft/cache \
    --dataset data_1129 \
    --val_size 0.2 \
    --seed 1129 \
    --bf16 True \
    --overwrite_cache False \
    --cutoff_len 2048 \
    --output_dir output/train_wakeng_sft_8core_qwen3-4b-thinking \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1\
    --eval_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.00 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --plot_loss \
    --preprocessing_num_workers 4 \
    --overwrite_output_dir True \
    --save_only_model False \
    --enable_thinking True \
    --max_grad_norm 1.0 