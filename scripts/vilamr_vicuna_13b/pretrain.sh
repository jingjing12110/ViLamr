#!/bin/bash

# ==============================================================================
# ViLamr-13b
it_file=sharegpt4v_filtered_676k.json

per_device_bs=16
accumulation_step=4

mm_projector_type=gate_mixer
vision_tower_mix=True
num_register_tokens=24

# ==============================================================================
output=playground/checkpoints/vilamr-vicuna-13b-pretrain

deepspeed vilamr/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path playground/checkpoints/vicuna-13b-v1.5 \
    --version plain \
    --data_path playground/data/pretrain/${it_file} \
    --image_folder playground/data/images \
    --vision_tower playground/checkpoints/openai/clip-vit-large-patch14-336 \
    --mm_projector_type ${mm_projector_type} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vision_tower_mix $vision_tower_mix \
    --num_register_tokens $num_register_tokens \
    --output_dir ${output} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $accumulation_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
