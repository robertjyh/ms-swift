export MASTER_PORT=29501
export NCCL_P2P_DISABLE=1
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift pt \
    --model_type qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --custom_dataset_info /home/lxy/Documents/ms-swift/wsi-scripts/wsi_pt_dataset_info.json \
    --dataset q1m \
    --train_type full \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    --trainable_parameters vision_projector \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --max_length 8192 \
    --deepspeed zero3 \
    --output_dir /data1/liuxiaoyu/outputs/qwen2.5vl-pt \
    --report_to wandb \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --streaming true \
    --max_steps 50 \
    --save_strategy epoch \
    --use_hf true \
    --attn_impl flash_attn \
    #--master_port 29505 
    #> ./outputs/optimus-qwen0.6-pt/example.log 2>&1

    # --per_device_train_batch_size  1 \
    # --save_steps 50 \
    # --save_total_limit 2 \
    # --logging_steps 5 \
