#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

home_dir=''

CKPT=${1}
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.eval.model_vqa_loader \
        --model-path "${home_dir}"/ckpts/llava_checkpoints/"${1}" \
        --question-file "${home_dir}"/llava/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder "${home_dir}"/llava/playground/data/eval/vqav2/test2015 \
        --answers-file "${home_dir}"/llava/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ds_math &
done

wait

output_file="${home_dir}"/llava/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${home_dir}"/llava/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir "${home_dir}"/llava/playground/data/eval/vqav2/

