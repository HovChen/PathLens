import torch
import argparse
from PIL import Image

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def add_special_tokens_and_resize_model(tokenizer, model):
    special_tokens = {
        'additional_special_tokens': (['<think>', '</think>', '<answer>', '</answer>'])
    }
    num_new = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return num_new

com_vision_args = argparse.Namespace(
    freeze_backbone=False,
    mm_patch_merge_type='flat',
    mm_projector_type='mlp2x_gelu',
    mm_use_im_patch_token=False,
    mm_use_im_start_end=False,
    mm_vision_select_feature='patch',
    mm_vision_select_layer=-2,
    model_name_or_path=None,
    pretrain_mm_mlp_adapter=None,
    tune_mm_mlp_adapter=False,
    version=None,
    vision_tower=None
)

def load_weights(model, hlora_path, vocab_proj_path=None):
    hlora_weights = torch.load(hlora_path)
    hlora_unexpected_keys = model.load_state_dict(hlora_weights, strict=False)[1]
    if hlora_unexpected_keys:
        print(f"Warning: Unexpected keys in hlora checkpoint: {hlora_unexpected_keys}")

    if vocab_proj_path:
        vocab_proj_weights = torch.load(vocab_proj_path)
        vocab_proj_unexpected_keys = model.load_state_dict(vocab_proj_weights, strict=False)[1]
        if vocab_proj_unexpected_keys:
            print(f"Warning: Unexpected keys in vocab_proj checkpoint: {vocab_proj_unexpected_keys}")

    return model
