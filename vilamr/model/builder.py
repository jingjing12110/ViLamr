#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig
)
from vilamr.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from vilamr.model.language_model.llava_llama import LlavaLlamaForCausalLM


def load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=False,
        vision_tower_mix=False,
        for_debug=False,
        **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'vilamr' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. '
                'If you are loading a LoRA model, please provide the `model_base` argument. '
                'Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.'
            )
        if 'lora' in model_name.lower() and model_base is not None:
            from vilamr.model.language_model.llava_llama import LlavaConfig

            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base,
                use_fast=False
            )

            print(f'Loading LLaVA from base model: {model_base}')
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **kwargs
            )
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(
                    token_num, token_dim,
                    device=model.device, dtype=model.dtype
                ))
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, token_dim,
                        device=model.device, dtype=model.dtype
                    )
                )

            print('Loading additional LLaVA weights, e.g., mm_projector...')
            non_lora_file = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_file):
                non_lora_trainables = torch.load(
                    non_lora_file, map_location='cpu'
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')

                non_lora_trainables = load_from_hf(
                    model_path,
                    'non_lora_trainables.bin'
                )

            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v for k, v in
                non_lora_trainables.items()
            }

            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v for k, v in
                    non_lora_trainables.items()
                }

            # todo 【model.model.load_state_dict 在本地测试】
            # model.model.load_state_dict(non_lora_trainables, strict=False)  # acc=0 ?
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path, **kwargs)

            print('Merging LoRA weights...')
            model = model.merge_and_unload()

            print('Model is loaded...')

        elif model_base is not None:
            # this may be mm projector only
            print('Loading model from base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False
            )
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=cfg_pretrained,
                **kwargs
            )

            mm_projector_weights = torch.load(
                os.path.join(model_path, 'mm_projector.bin'),
                map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in
                                    mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # e.g., full fine-tuning model
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False
            )
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(
                model_base,
                use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs)

    # init image_processor
    image_processor = None
    if 'vilamr' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        # print(vision_tower.state_dict()['vision_tower.vision_model.post_layernorm.bias'])

        if vision_tower_mix:
            # device = vision_tower.state_dict()['vision_tower.vision_model.post_layernorm.bias']
            print(f"\033[91m{'=' * 120}\033[0m")
            print("Loading convnext_xxl!")  
            convnext_xxl = model.get_vision_tower_mix()
            if not convnext_xxl.is_loaded:
                convnext_xxl.load_model()
            # print(convnext_xxl.device)
            convnext_xxl.to(vision_tower.state_dict()['vision_tower.vision_model.post_layernorm.bias'])

            print(f"\033[91m{'=' * 120}\033[0m")

            if for_debug:
                convnext_xxl.to(device=device, dtype=torch.float16)

            if device_map != 'auto':
                convnext_xxl.to(device=device_map, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
