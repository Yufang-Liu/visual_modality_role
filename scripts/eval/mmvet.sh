#!/bin/bash

home_dir=''

python3 -m llava.eval.model_vqa \
    --model-path ${home_dir}/ckpts/llava_checkpoints/"${1}" \
    --question-file ${home_dir}/llava/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${home_dir}/llava/playground/data/eval/mm-vet/images \
    --answers-file ${home_dir}/llava/playground/data/eval/mm-vet/answers/"${1}".jsonl \
    --temperature 0 \
    --conv-mode ds_math

mkdir -p ${home_dir}/llava/playground/data/eval/mm-vet/results

python3 scripts/convert_mmvet_for_eval.py \
    --src ${home_dir}/llava/playground/data/eval/mm-vet/answers/"${1}".jsonl \
    --dst ${home_dir}/llava/playground/data/eval/mm-vet/results/"${1}".json

