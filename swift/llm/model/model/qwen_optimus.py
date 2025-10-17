
# -*- coding: utf-8 -*-

from typing import Any, Dict

import torch.nn as nn 

from swift.utils import get_env_args
from ..utils import ModelInfo
from ..register import (Model, ModelGroup, ModelMeta, register_model)
from ..constant import MLLMModelType
from .qwen import get_model_tokenizer_qwen
from swift.llm import TemplateType
from safetensors.torch import load_file
import os

import json
import torch
from safetensors.torch import save_file
from safetensors import safe_open

def load_vision_projector(model, model_dir, device="cuda", cache_dir=None, force_reload=False):
    """
    加载 vision_projector.* 的权重。
    - 自动识别单文件 / 多分片模型
    - 支持缓存机制：首次提取后保存为独立 safetensors 文件，下次直接加载
    - 避免加载整个模型，几乎不占显存
    """

    prefix = "vision_projector."
    cache_dir = cache_dir or model_dir
    cache_path = os.path.join(cache_dir, "vision_projector.safetensors")

    # 如果缓存存在且未强制刷新，则直接加载
    if os.path.exists(cache_path) and not force_reload:
        print(f"[INFO] Loading cached projector weights from {cache_path}")
        state_dict = torch.load(cache_path, map_location=device)
        model.vision_projector.load_state_dict(state_dict, strict=False)
        model.vision_projector.to(device)
        return model

    # 否则从原始权重提取
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    safetensor_files = []

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        safetensor_files = sorted(set(index["weight_map"].values()))
    else:
        safetensor_files = ["model.safetensors"]

    projector_state_dict = {}
    for filename in safetensor_files:
        ckpt_path = os.path.join(model_dir, filename)
        if not os.path.exists(ckpt_path):
            continue
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    new_key = key.replace(prefix, "")
                    projector_state_dict[new_key] = f.get_tensor(key)

    if len(projector_state_dict) == 0:
        print(f"[WARN] No {prefix} weights found in {model_dir}")
        return model

    # 保存缓存以便下次直接加载
    torch.save(projector_state_dict, cache_path)
    print(f"[INFO] Extracted and cached {len(projector_state_dict)} {prefix} weights to {cache_path}")

    # 加载到模型
    model.vision_projector.load_state_dict(projector_state_dict, strict=False)
    model.vision_projector.to(device)
    return model




def get_model_tokenizer_qwen3_optimus(model_dir: str,
                                      model_info: ModelInfo,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # 1. Load Qwen3 base model
    model, tokenizer = get_model_tokenizer_qwen(model_dir, model_info, model_kwargs, load_model, **kwargs)
   
    # 2. Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '<|image_start|>', '<|image_end|>',
            '<|patch_start|>', '<|patch_end|>',
            '<|image_pad|>', '<|patch_pad|>',
            '<|image_idx_start|>', '<|image_idx_end|>',
            '<|patch_idx_start|>', '<|patch_idx_end|>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load H-optimus-1 vision encoder
    import timm
    vision_tower = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1", 
        pretrained=True, 
        init_values=1e-5, 
        dynamic_img_size=False
    )
    vision_tower.eval()

    # 4. Initialize Adapter (Projector)
    hidden_size = model.config.hidden_size  # Qwen3 hidden size
    vision_hidden_size = 1536  # H-optimus-1 output size
    downsample_ratio = get_env_args('downsample_ratio', float, 0.5)

    model.vision_tower = vision_tower
    model.vision_projector = nn.Sequential(
        nn.LayerNorm(vision_hidden_size * int(1 / downsample_ratio) ** 2),
        nn.Linear(vision_hidden_size * int(1 / downsample_ratio) ** 2, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )

    if os.path.exists(model_dir):
        model = load_vision_projector(model, model_dir, device='cuda', cache_dir=model_dir)
        # ckpt_path = model_dir + '/model.safetensors'
        # state_dict = load_file(ckpt_path, device='cuda')  
        # projector_state_dict = {k.replace("vision_projector.", ""): v
        #                 for k, v in state_dict.items() if k.startswith("vision_projector.")}        
        # model.vision_projector.load_state_dict(projector_state_dict)

    return model, tokenizer

# 注册 Optimus Qwen3 模型
register_model(
    ModelMeta(
        MLLMModelType.optimus_qwen3,
        [ModelGroup([
            Model('Qwen/Qwen3-8B', 'Qwen/Qwen3-8B'),
            Model('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B'),
        ])],
        template=TemplateType.optimus_qwen3,
        get_function=get_model_tokenizer_qwen3_optimus,
        architectures=['Qwen3ForCausalLM'],
    ))
