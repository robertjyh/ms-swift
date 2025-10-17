
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np
from torchvision import transforms

from swift.utils import get_env_args
from ..base import Template
from ..template_inputs import StdTemplateInputs
from ..constant import MLLMTemplateType
from ..register import register_template
from .qwen import QwenTemplateMeta

def split_image_to_patches(image: Image.Image, patch_size: int = 224, max_patches: int = 64):
    width, height = image.size
    max_cols = max(1, width // patch_size)
    max_rows = max(1, height // patch_size)
    max_possible = max_cols * max_rows
    num_patches = min(max_patches, max_possible)
    if num_patches <= 0:
        return [], (0, 0)

    aspect_ratio = width / max(1, height)
    cols = max(1, int(round(np.sqrt(num_patches * aspect_ratio))))
    rows = max(1, int(np.ceil(num_patches / cols)))
    cols = min(cols, max_cols)
    rows = min(rows, max_rows)
    # ensure rows*cols >= num_patches
    while rows * cols < num_patches:
        if cols < max_cols:
            cols += 1
        elif rows < max_rows:
            rows += 1
        else:
            break

    target_width = cols * patch_size
    target_height = rows * patch_size
    image = image.resize((target_width, target_height), Image.LANCZOS)

    patches = []
    for i in range(rows):
        for j in range(cols):
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patch = image.crop(box)
            patch_tensor = transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            )(transforms.ToTensor()(patch))
            patches.append(patch_tensor)
    return patches, (rows, cols)


# def split_image_to_patches(image: Image.Image, patch_size: int = 224, 
#                          max_patches: int = 64) -> tuple[list[torch.Tensor], tuple[int, int]]:
#     """
#     Splits an image into fixed-size patches.

#     Args:
#         image: The PIL Image.
#         patch_size: The size of each patch (default: 224).
#         max_patches: The maximum number of patches.

#     Returns:
#         A tuple containing:
#         - patches: A list of normalized patch tensors.
#         - grid_size: A tuple representing the grid size (rows, cols).
#     """
#     # 1. Calculate the optimal grid
#     width, height = image.size
#     aspect_ratio = width / height
    
    
#     # Dynamically calculate grid size (inspired by InternVL)
#     num_patches = min(max_patches, (width // patch_size) * (height // patch_size))

#     if aspect_ratio > 1:
#         cols = int(np.sqrt(num_patches * aspect_ratio))
#         rows = num_patches // cols if cols > 0 else 0
#     else:
#         rows = int(np.sqrt(num_patches / aspect_ratio)) if aspect_ratio > 0 else 0
#         cols = num_patches // rows if rows > 0 else 0

#     if rows == 0 or cols == 0:
#         return [], (0, 0)

#     # 2. Resize the image to the grid dimensions
#     target_width = cols * patch_size
#     target_height = rows * patch_size
#     image = image.resize((target_width, target_height), Image.LANCZOS)
    
#     # 3. Split into patches
#     normalize = transforms.Normalize(
#         mean=(0.707223, 0.578729, 0.703617),  # H-optimus-0 normalization parameters
#         std=(0.211883, 0.230117, 0.177517)
#     )
#     to_tensor = transforms.ToTensor()
    
#     patches = []
#     for i in range(rows):
#         for j in range(cols):
#             box = (j * patch_size, i * patch_size, 
#                    (j + 1) * patch_size, (i + 1) * patch_size)
#             patch = image.crop(box)
#             patch_tensor = normalize(to_tensor(patch))
#             patches.append(patch_tensor)
    
#     return patches, (rows, cols)


@dataclass
class OptimusQwen3Meta(QwenTemplateMeta):
    """沿用 Qwen-VL 的 ChatML 模板，保证与 Qwen3 tokenizer 的特殊符号一致。"""
    # 仅覆盖系统提示，其余符号沿用 QwenTemplateMeta（ChatML 风格）
    default_system: str = 'You are a multimodal AI assistant capable of understanding images.'

class OptimusQwen3Template(Template):
    use_model = True  # Requires model for encoding

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define placeholders
        self.image_placeholder = ['<|image_start|>', '<|image_pad|>', '<|image_end|>']
        self.patch_placeholder = ['<|patch_start|>', '<|patch_pad|>', '<|patch_end|>']

        # Get special token IDs
        self.image_start_id, self.image_end_id, self.image_pad_id = self.tokenizer.convert_tokens_to_ids(['<|image_start|>', '<|image_end|>', '<|image_pad|>'])
        self.patch_start_id, self.patch_end_id, self.patch_pad_id = self.tokenizer.convert_tokens_to_ids(['<|patch_start|>', '<|patch_end|>', '<|patch_pad|>'])

        # Get extract feature parameters
        self.select_layer = get_env_args('select_layer', int, -1)
        self.downsample_ratio = get_env_args('downsample_ratio', float, 0.5)
        self.input_size = 224
        self.max_num = get_env_args('max_num', int, 100)
        self.num_patch_token = int((self.input_size // 14)**2 * (0.5**2))

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs) -> List[str]:
        """Replaces media placeholders."""
        if media_type == 'image':
            return self.image_placeholder
        return super().replace_tag(media_type, index, inputs)
    
    def _pixel_shuffle(self, x, scale_factor=0.5):
        n, h, w, c = x.size()  # N, H, W, C
        # N, H, W, C --> N, H, W * scale, C // scale
        x = x.view(n, h, int(w * scale_factor), int(c / scale_factor))
        # N, H, W * scale, C // scale --> N, W * scale, H, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, W * scale, H, C // scale --> N, W * scale, H * scale, C // (scale ** 2)
        x = x.view(n, int(w * scale_factor), int(h * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """Encodes both text and vision."""
        #print("encode")
        # 1. Base class handles the text part
        encoded = super()._encode(inputs)

        if not inputs.images:
            return encoded

        # 2. Process images
        all_patches = []
        patch_counts = []

        for image in inputs.images:
            patches, _ = split_image_to_patches(image, self.input_size, self.max_num)
            all_patches.extend(patches)
            patch_counts.append(len(patches))

        # 3. Expand image_pad to the actual patch sequence
        input_ids = encoded['input_ids']
        labels = encoded.get('labels')
        attention_mask = encoded.get('attention_mask')

        new_input_ids = []
        new_labels = [] if labels is not None else None
        new_attention_mask = [] if attention_mask is not None else None

        img_idx = 0
        i = 0
        while i < len(input_ids):
            if input_ids[i] == self.image_pad_id and img_idx < len(patch_counts):
                n_patches = patch_counts[img_idx]
                for _ in range(n_patches):
                    patch_id_list = [self.patch_start_id] + [self.patch_pad_id] * self.num_patch_token + [self.patch_end_id]
                    new_input_ids.extend(patch_id_list)
                    if new_labels is not None:
                        new_labels.extend([-100] * (self.num_patch_token + 2))  # Ignore loss for visual tokens
                    if new_attention_mask is not None:
                        new_attention_mask.extend([1] * (self.num_patch_token + 2))
                img_idx += 1
            elif input_ids[i] in [self.image_start_id, self.image_end_id]:
                # ignore <|image_start|> <|image_end|> in loss
                new_input_ids.append(input_ids[i])
                if new_labels is not None:
                    new_labels.append(-100)
                if new_attention_mask is not None:
                    new_attention_mask.append(attention_mask[i])
            else:
                new_input_ids.append(input_ids[i])
                if new_labels is not None:
                    new_labels.append(labels[i])
                if new_attention_mask is not None:
                    new_attention_mask.append(attention_mask[i])
            i += 1

        # 4. Update encoded dict
        encoded['input_ids'] = new_input_ids
        if new_labels is not None:
            encoded['labels'] = new_labels
        if new_attention_mask is not None:
            encoded['attention_mask'] = new_attention_mask

        # 5. Add vision data
        if all_patches:
            encoded['pixel_values'] = torch.stack(all_patches)
            encoded['num_patches'] = torch.tensor(patch_counts, dtype=torch.long)

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """Collates batch data and handles pixel_values list->tensor conversion."""
        # 先处理可能的嵌套 list -> tensor 转换
        for item in batch:
            if 'pixel_values' in item and isinstance(item['pixel_values'], list):
                # 将嵌套的 list 转换为 torch.Tensor
                item['pixel_values'] = torch.as_tensor(item['pixel_values'])
            if 'num_patches' in item and isinstance(item['num_patches'], list):
                # 同样处理 num_patches
                item['num_patches'] = torch.as_tensor(item['num_patches'])
        
        # 调用父类的 _data_collator
        result = super()._data_collator(batch, padding_to=padding_to)

        # 汇总视觉特征（有些 batch 样本可能没有图像，此时列表为空需要跳过）
        all_pixel_values = [item['pixel_values'] for item in batch if item.get('pixel_values') is not None]
        all_num_patches = [item['num_patches'] for item in batch if item.get('num_patches') is not None]

        if len(all_pixel_values) > 0:
            # 只有当至少存在一个样本包含 pixel_values 时，才进行拼接
            result['pixel_values'] = torch.cat(all_pixel_values, dim=0)
            if len(all_num_patches) > 0:
                result['num_patches'] = torch.cat(all_num_patches, dim=0)

        return result
    
    def _extract_feature(self, model, pixel_values):
        print("feature")
        vit_embeds = model.vision_tower.forward_intermediates(
            pixel_values,
            indices=[self.select_layer],
            output_fmt='NCHW',
            return_prefix_tokens=False,  # remove 1 CLS token + 4 register token
            intermediates_only=True,
            stop_early=True,
            norm=False
        )
        vit_embeds = vit_embeds[0][:, :, :, :]  # N, C, H, W
        vit_embeds = vit_embeds.permute(0, 2, 3, 1)  # N, H, W, C
        vit_embeds = self._pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = model.vision_projector(vit_embeds)
        return vit_embeds

    def _post_encode(self, model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processing: injects vision features into language embeddings."""
        print("****")
        if 'pixel_values' not in inputs:
            return inputs
        print("*******")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        pixel_values = inputs['pixel_values'].to(device=device, dtype=dtype)
        input_ids = inputs['input_ids']
        
        # 获取语言模型的 embedding 层
        embedding = model.get_input_embeddings()
        inputs_embeds = embedding(input_ids.to(device))

        with torch.no_grad():
            if hasattr(model, 'vision_tower'):
                vision_embeds = self._extract_feature(model, pixel_values)
                print(vision_embeds)

        # 找到所有 patch_pad token 的位置并替换为视觉特征
        patch_pad_mask = (input_ids == self.patch_pad_id)
        if patch_pad_mask.any():
            vision_embeds = vision_embeds.to(device=device, dtype=inputs_embeds.dtype)
            inputs_embeds[patch_pad_mask] = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        return {'inputs_embeds': inputs_embeds}

register_template(
    OptimusQwen3Meta(
        MLLMTemplateType.optimus_qwen3,
        template_cls=OptimusQwen3Template,
    )
)

