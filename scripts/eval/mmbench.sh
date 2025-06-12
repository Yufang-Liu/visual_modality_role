#!/bin/bash

home_dir=''

pip install openpyxl

SPLIT="mmbench_dev_20230712"

python3 -m llava.eval.model_vqa_mmbench \
    --model-path ${home_dir}/ckpts/llava_checkpoints/"${1}" \
    --question-file ${home_dir}/llava/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ${home_dir}/llava/playground/data/eval/mmbench/answers/$SPLIT/"${1}".jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ds_math

mkdir -p ${home_dir}/llava/playground/data/eval/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${home_dir}/llava/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ${home_dir}/llava/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ${home_dir}/llava/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment "${1}"
