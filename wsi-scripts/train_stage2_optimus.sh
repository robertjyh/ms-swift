export MASTER_PORT=29502
export NCCL_P2P_DISABLE=1
WANDB_ENTITY=robertjyh-nanjing-university WANDB_PROJECT=wsi_stage2 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
python -m swift.cli.main sft \
    --model_type qwen3-optimus \
    --model /data1/liuxiaoyu/outputs/optimus-qwen3-pt/v33-20250926-102407/checkpoint-7600 \
    --custom_dataset_info /home/lxy/Documents/ms-swift/wsi-scripts/wsi_pt_dataset_info.json \
    --dataset pathcap_qa \
    --train_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --trainable_parameters vision_projector \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 1 \
    --max_length 2048 \
    --deepspeed zero3 \
    --output_dir /data1/liuxiaoyu/outputs/optimus-qwen3-sft-stage2 \
    --report_to wandb \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --streaming true \
    --max_steps 7600 \
    --save_strategy epoch \
    --use_hf true \
    --attn_impl flash_attn \
    #--master_port 29505 
    #> ./outputs/optimus-qwen0.6-pt/example.log 2>&1

    # --per_device_train_batch_size  1 \
    # --save_steps 50 \
    # --save_total_limit 2 \
    # --logging_steps 5 \
