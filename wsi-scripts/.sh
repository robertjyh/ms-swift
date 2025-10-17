export NCCL_P2P_DISABLE=1
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=7 \
swift pt \
    --model_type qwen3-optimus \
    --model Qwen/Qwen3-0.6B \
    --custom_dataset_info /home/lxy/Documents/ms-swift/wsi-scripts/wsi_pt_dataset_info.json \
    --dataset q1m \
    --train_type full \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    --trainable_parameters vision_projector \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --max_length 8192 \
    --deepspeed zero2 \
    --output_dir ./outputs/optimus-qwen3-pt \
    --report_to wandb \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --streaming true \
    --max_steps 1 \
    --save_strategy epoch \
    --use_hf true \
    --attn_impl flash_attn
    > ./outputs/optimus-qwen0.6-pt/example.log 2>&1

    # --per_device_train_batch_size  1 \
    # --save_steps 50 \
    # --save_total_limit 2 \
    # --logging_steps 5 \
