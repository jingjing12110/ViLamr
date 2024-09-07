#!/bin/bash

# **************************************************************************
output=""
echo "Testing LMMs: ${output}"

eval_base_dir="playground/data/vilamr_eval"
# **************************************************************************

cot_prompt_type=mr_detail
# cot_prompt_type=mr_rat

# ********************************************************************************************
GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}

# ********************************************************************************************
# mmstar [1500]
output_file=$output/result/mmstar-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/mmstar-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/mmstar-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/mmstar/test.json


# ********************************************************************************************
# mmmu [900]
output_file=$output/result/mmmu-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/mmmu-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/mmmu-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/mmmu/val.json

# ********************************************************************************************
# rwqa [765]
output_file=$output/result/rwqa-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/rwqa-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/rwqa-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/rwqa/test.json

# ********************************************************************************************
# mmb_dev [4,329]
output_file=$output/result/mmb_dev-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/mmb_dev-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/mmb_dev-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/mm_bench/dev_en.json


# ********************************************************************************************
# GeoQA [754]
output_file=$output/result/geoqa-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/geoqa-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/geoqa-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/geoqa/test.json

# ********************************************************************************************
# gd_vcr [886]
output_file=$output/result/gd_vcr-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/gd_vcr-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/gd_vcr-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/gd_vcr/val.json

# ********************************************************************************************
# wemath [1,740]
output_file=$output/result/wemath-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/wemath-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/wemath-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/wemath/testmini.json

# ********************************************************************************************
# m3cot [2,318]
output_file=$output/result/m3cot-cot_${cot_prompt_type}-merge.jsonl
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output/result/m3cot-cot_${cot_prompt_type}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

python vilamr/eval/compute_accuracy.py \
    --result_file $output/result/m3cot-cot_${cot_prompt_type}-merge.jsonl \
    --ann_file ${eval_base_dir}/m3cot/test.json
