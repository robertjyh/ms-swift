export MASTER_PORT=29501
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=5 \
python -m swift.cli.main infer \
    --model /data1/liuxiaoyu/outputs/optimus-qwen3-sft-stage2/v53-20251010-101057/checkpoint-7600 \
    --infer_backend pt \
    --temperature 0 \
    --val_dataset /data1/liuxiaoyu/wrap/pathcap_qa_ms_swift_test.jsonl \
    --gpu_memory_utilization 0.9 \
    --max_batch_size 1