#!/bin/bash

# *****************************************************************************
model_max_length=2048
vision_tower_mix=True
mm_projector_type=gate_mixer
num_register_tokens=24

PROMPT_VERSION="llava_llama_3_1"
# *****************************************************************************
mit_file=mcot_instruct_v1_266k.json

# bs=128, lr=2e-5
tbs=128
accumulation_step=2
bs=8
epoch=2
lr=2e-5

# *****************************************************************************
output=experiments/sft-266k_v1-ep${epoch}_bs${tbs}_lr${lr}/vilamr-llama3-8b
mkdir -p $output
mkdir -p $output/result/
cp $0 $output/run.sh

deepspeed vilamr/train/train_mem.py \
    --deepspeed scripts/vilamr_llama3_8b/zero3.json \
    --model_name_or_path playground/checkpoints/Meta-Llama-3.1-8B-Instruct \
    --llm_backbone llama_3 \
    --llm_pad_token pad \
    --version ${PROMPT_VERSION} \
    --model_max_length $model_max_length \
    --data_path playground/data/vilamr_it/${mit_file} \
    --image_folder playground/data/images \
    --vision_tower playground/checkpoints/openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter playground/checkpoints/vilamr-llama3-8b-pretrain/mm_projector.bin \
    --vision_tower_mix $vision_tower_mix \
    --mm_projector_type $mm_projector_type \
    --num_register_tokens $num_register_tokens \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --output_dir ${output} \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $accumulation_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb


