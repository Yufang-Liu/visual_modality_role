#!/bin/bash

home_dir=''
pretrained_dir=${home_dir}/ckpts/model


deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${pretrained_dir}/deepseek-math-7b-rl \
    --version plain \
    --dataset_use MAVISPretrain \
    --vision_tower ${pretrained_dir}/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ${home_dir}/ckpts/llava_checkpoints/mavis-llava-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${pretrained_dir}/deepseek-math-7b-rl \
    --version ds_math \
    --dataset_use FineTune \
    --vision_tower ${pretrained_dir}/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${home_dir}/ckpts/llava_checkpoints/mavis-llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${home_dir}/ckpts/llava_checkpoints/mavis-llava-pretrain-sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none

deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${home_dir}/ckpts/llava_checkpoints/mavis-llava-pretrain-sft \
    --version ds_math \
    --dataset_use MAVISFineTune \
    --vision_tower ${pretrained_dir}/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${home_dir}/ckpts/llava_checkpoints/mavis-llava-pretrain-sft-mathsft \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none