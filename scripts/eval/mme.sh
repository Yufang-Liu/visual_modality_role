#!/bin/bash

home_dir=''


python3 -m llava.eval.model_vqa_loader \
    --model-path ${home_dir}/ckpts/llava_checkpoints/"${1}" \
    --question-file ${home_dir}/llava/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ${home_dir}/llava/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ${home_dir}/llava/playground/data/eval/MME/answers/"${1}".jsonl \
    --temperature 0 \
    --conv-mode ds_math

cd ${home_dir}/llava/playground/data/eval/MME

python3 convert_answer_to_mme.py --experiment "${1}"

cd eval_tool

python3 calculation.py --results_dir answers/"${1}"
