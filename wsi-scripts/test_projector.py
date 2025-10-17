from safetensors.torch import load_file
import os

ckpt_path = "/data1/liuxiaoyu/outputs/optimus-qwen3-sft-stage2/v2-20250929-132917/checkpoint-100/model.safetensors"

# 加载到第7张显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
state_dict = load_file(ckpt_path, device='cuda')  # 加载为 CPU tensor

# 假设 projector 的参数名都以 "vision_projector." 开头
projector_state_dict = {k.replace("vision_projector.", ""): v
                        for k, v in state_dict.items() if k.startswith("vision_projector.")}

# 确保 projector_state_dict 不为空
if len(projector_state_dict) == 0:
    print("⚠️ 没有找到 vision_projector 权重，可能需要训练后单独保存。")
else:
    model.vision_projector.load_state_dict(projector_state_dict)
    print("✅ vision_projector 权重加载成功")
