#!/bin/bash

home_dir=''

python3 -m llava.eval.model_vqa_loader \
    --model-path "${home_dir}"/ckpts/llava_checkpoints/"${1}" \
    --question-file "${home_dir}"/llava/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder "${home_dir}"/llava/playground/data/eval/textvqa/train_images \
    --answers-file "${home_dir}"/llava/playground/data/eval/textvqa/answers/"${1}".jsonl \
    --temperature 0 \
    --conv-mode ds_math

python3 -m llava.eval.eval_textvqa \
    --annotation-file "${home_dir}"/llava/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${home_dir}"/llava/playground/data/eval/textvqa/answers/"${1}".jsonl
