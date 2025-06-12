#!/bin/bash

home_dir=''

python3 -m llava.eval.model_vqa_loader \
    --model-path "${home_dir}"/ckpts/llava_checkpoints/"${1}" \
    --question-file "${home_dir}"/llava/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder "${home_dir}"/llava/playground/data/eval/pope/val2014 \
    --answers-file "${home_dir}"/llava/playground/data/eval/pope/answers/"${1}".jsonl \
    --temperature 0 \
    --conv-mode ds_math

python3 llava/eval/eval_pope.py \
    --annotation-dir "${home_dir}"/llava/playground/data/eval/pope/coco \
    --question-file "${home_dir}"/llava/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file "${home_dir}"/llava/playground/data/eval/pope/answers/"${1}".jsonl
