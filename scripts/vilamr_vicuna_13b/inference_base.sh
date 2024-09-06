#!/bin/bash

# **************************************************************************
output=""
echo "Evaluating LMMs: ${output}"
# **************************************************************************

vision_tower_mix=True
max_new_token=512
PROMPT_VERSION="vicuna_v1"

eval_tasks='mmstar mmmu rwqa mmb_dev gd_vcr wemath geoqa m3cot'
cot_prompt_types='mr_detail mr_detail mr_detail mr_detail mr_detail mr_detail mr_rat mr_rat'

GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python vilamr/eval/inference_vicuna.py \
    --vision_tower_mix $vision_tower_mix \
    --max_new_tokens $max_new_token \
    --model_path ${output} \
    --eval_tasks ${eval_tasks} \
    --cot_prompt_types ${cot_prompt_types} \
    --answers_file_id ${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv_mode ${PROMPT_VERSION} &
done
wait
