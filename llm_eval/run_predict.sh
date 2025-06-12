#!/bin/bash

home_dir=''
ckpt_dir=''

cd llm_eval

MATHVERSE_DATASET="${home_dir}/rl_math/data/MathVerse/testmini.json"
MATHVERSE_IMG_ROOT="${home_dir}/rl_math/data/MathVerse/images/"
MATHVISTA_DATASET="${home_dir}/rl_math/data/MathVista/query.json"
MATHVISTA_IMG_ROOT="${home_dir}/rl_math/data/MathVista/"
MATHVISTA_IMG_FILE="${home_dir}/rl_math/data/MathVista/testmini.json"
WE_MATH_DATASET="${home_dir}/rl_math/data/We-Math/testmini.json"
WE_MATH_IMG_ROOT="${home_dir}/rl_math/data/We-Math/data/"
MATHVISION_DATASET="${home_dir}/rl_math/data/MathVision/testmini.json"
MATHVISION_IMG_ROOT="${home_dir}/rl_math/data/MathVision/"
GEOQA_DATASET="${home_dir}/gllava/playground/data/test_questions.jsonl"
GEOQA_IMG_ROOT="${home_dir}/gllava/playground/data/images/"
HC_M3D_DATASET="${home_dir}/rl_math/data/HC-M3D/data.json"
HC_M3D_IMG_ROOT="${home_dir}/rl_math/data/HC-M3D/raw_data/"
MATH_POPE_DATASET="${home_dir}/rl_math/data/generated_data/polling_based_questions/"
MATH_IMG_ROOT="${home_dir}/data/gllava_data/images/geo3k"


eval_ckpts(){
  python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name mathverse \
  --dataset_path ${MATHVERSE_DATASET} \
  --image_root ${MATHVERSE_IMG_ROOT} \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name HC-M3D \
  --dataset_path ${HC_M3D_DATASET} \
  --image_root ${HC_M3D_IMG_ROOT} \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name mathvista \
  --dataset_path ${MATHVISTA_DATASET} \
  --image_root ${MATHVISTA_IMG_ROOT} \
  --image_file ${MATHVISTA_IMG_FILE} \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

   python3 predict.py \
   --model_name "${1}" \
   --model_path ${ckpt_dir}/"${2}" \
   --dataset_name wemath \
   --dataset_path ${WE_MATH_DATASET} \
   --image_root ${WE_MATH_IMG_ROOT} \
   --with_image "${3}" \
   --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
   --model_name "${1}" \
   --model_path ${ckpt_dir}/"${2}" \
   --dataset_name mathvision \
   --dataset_path ${MATHVISION_DATASET} \
   --image_root ${MATHVISION_IMG_ROOT} \
   --with_image "${3}" \
   --output_dir "${home_dir}"/rl_math/output/llm_results/

   python3 predict.py \
   --model_name "${1}" \
   --model_path ${ckpt_dir}/"${2}" \
   --dataset_name geoqa \
   --dataset_path ${GEOQA_DATASET} \
   --image_root ${GEOQA_IMG_ROOT} \
   --with_image "${3}" \
   --output_dir "${home_dir}"/rl_math/output/llm_results/

wait
}

eval_ckpts_shuffle_option(){
  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathverse \
  --dataset_path "${home_dir}/rl_math/data/MathVerse/testmini_shuffle_option.json" \
  --image_root ${MATHVERSE_IMG_ROOT} \
  --with_image "${3}" \
  --split shuffle_option \
  --output_dir "${home_dir}"/rl_math/output/llm_results/ 

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathvista \
  --dataset_path "${home_dir}/rl_math/data/MathVista/query_shuffle_option.json" \
  --image_root ${MATHVISTA_IMG_ROOT} \
  --image_file ${MATHVISTA_IMG_FILE} \
  --with_image "${3}" \
  --split shuffle_option \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name wemath \
  --dataset_path "${home_dir}/rl_math/data/We-Math/testmini_shuffle_option.json" \
  --image_root ${WE_MATH_IMG_ROOT} \
  --split shuffle_option \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathvision \
  --dataset_path "${home_dir}/rl_math/data/MathVision/testmini_shuffle_option.json" \
  --image_root ${MATHVISION_IMG_ROOT} \
  --split shuffle_option \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name geoqa \
  --dataset_path "${home_dir}/gllava/playground/data/test_questions_shuffle_option.jsonl" \
  --image_root ${GEOQA_IMG_ROOT} \
  --with_image "${3}" \
  --split shuffle_option \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

wait
}

eval_ckpts_free_form(){
  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathverse \
  --dataset_path "${home_dir}/rl_math/data/MathVerse/testmini_free_form.json" \
  --image_root ${MATHVERSE_IMG_ROOT} \
  --with_image "${3}" \
  --split free_form \
  --output_dir "${home_dir}"/rl_math/output/llm_results/ &

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathvista \
  --dataset_path "${home_dir}/rl_math/data/MathVista/query_free_form.json" \
  --image_root ${MATHVISTA_IMG_ROOT} \
  --image_file "${home_dir}/rl_math/data/MathVista/testmini_free_form.json" \
  --with_image "${3}" \
  --split free_form \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name wemath \
  --dataset_path "${home_dir}/rl_math/data/We-Math/testmini_free_form.json" \
  --image_root ${WE_MATH_IMG_ROOT} \
  --split free_form \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/ 

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name mathvision \
  --dataset_path "${home_dir}/rl_math/data/MathVision/testmini_free_form.json" \
  --image_root ${MATHVISION_IMG_ROOT} \
  --split free_form \
  --with_image "${3}" \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  python3 predict.py \
  --model_name "${1}" \
  --model_path "${ckpt_dir}"/"${2}" \
  --dataset_name geoqa \
  --dataset_path "${home_dir}/gllava/playground/data/test_questions_free_form.jsonl" \
  --image_root ${GEOQA_IMG_ROOT} \
  --with_image "${3}" \
  --split free_form \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

wait
}

eval_ckpts_POPE(){
  CUDA_VISIBLE_DEVICES=0 python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name math_pope_adversarial \
  --dataset_path ${MATH_POPE_DATASET}/geo3k_test_point_adversarial.json \
  --image_root ${MATH_IMG_ROOT} \
  --with_image 1 \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  CUDA_VISIBLE_DEVICES=1 python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name math_pope_popular \
  --dataset_path ${MATH_POPE_DATASET}/geo3k_test_point_popular.json \
  --image_root ${MATH_IMG_ROOT} \
  --with_image 1 \
  --output_dir "${home_dir}"/rl_math/output/llm_results/

  CUDA_VISIBLE_DEVICES=2 python3 predict.py \
  --model_name "${1}" \
  --model_path ${ckpt_dir}/"${2}" \
  --dataset_name math_pope_random \
  --dataset_path ${MATH_POPE_DATASET}/geo3k_test_point_random.json \
  --image_root ${MATH_IMG_ROOT} \
  --with_image 1 \
  --output_dir "${home_dir}"/rl_math/output/llm_results/
}

eval_ckpts_POPE gllava-clip-base gllava-clip-base-pretrain-sft
#echo -----gllava-mathsft--------
#eval_ckpts_POPE gllava-mathsft gllava-pretrain-sft-mathsft
#echo -------gllava-dino-clip-concat-mathsft-------
#eval_ckpts_POPE gllava-dino-clip-concat-mathsft gllava-dino-clip-concat-pretrain-sft-mathsft
#echo ------------gllava-siglip-clip-cat-mathsft-----------
#eval_ckpts_POPE gllava-siglip-clip-cat-mathsft gllava-siglip-clip-cat-pretrain-sft-mathsft
#echo ------------gllava-dino-clip-siglip-concat-mathsft---------
#eval_ckpts_POPE gllava-dino-clip-siglip-concat-mathsft gllava-dino-clip-siglip-concat-pretrain-sft-mathsft
#echo ------------gllava-siglip-clip-mathsft--------------
#eval_ckpts_POPE gllava-siglip-clip-mathsft gllava-siglip-clip-pretrain-sft-mathsft
#echo ------------gllava-siglip-clip-dino---------------
#eval_ckpts_POPE gllava-siglip-clip-dino-mathsft gllava-siglip-clip-dino-pretrain-sft-mathsft
#echo -----------gllava-dino-clip-mathsft----------------
#eval_ckpts_POPE gllava-dino-clip-mathsft gllava-dino-clip-pretrain-sft-mathsft


#eval_ckpts gllava-clip-base gllava-clip-base-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip-siglip-concat-exist2 gllava-dino-clip-siglip-concat-exist2-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip-concat gllava-dino-clip-concat-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip-cat gllava-siglip-clip-cat-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip-siglip-concat gllava-dino-clip-siglip-concat-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip gllava-siglip-clip-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip-dino gllava-siglip-clip-dino-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip gllava-dino-clip-pretrain-sft-mathsft 1

#eval_ckpts gllava gllava-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip-cat gllava-siglip-clip-cat-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip-siglip-concat gllava-dino-clip-siglip-concat-pretrain-sft-mathsft 1
# eval_ckpts gllava-dino-clip-concat gllava-dino-clip-concat-pretrain-sft-mathsft 1
#eval_ckpts gllava-dino-clip gllava-dino-clip-pretrain-sft-mathsft 1
#eval_ckpts mathllava-dino-clip mathllava-dino-clip-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip  gllava-siglip-clip-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip-dino  gllava-siglip-clip-dino-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip gllava-siglip-clip-pretrain-sft-mathsft 1
#eval_ckpts gllava-siglip-clip_dino gllava-siglip-clip-dino-pretrain-sft-mathsft 1
# eval_ckpts gllava_clip_encoder gllava-pretrain-sft-mathsft_clip_finetune_3e-7_10_1_25_image_encoder_mix_2w 1
# eval_ckpts gllava_clip gllava-pretrain-sft-mathsft_clip_finetune_1e-6_10_1_25_mix_2w 1
#eval_ckpts gpt4o gpt-4o-2024-08-06 0
#eval_ckpts gpt4o-image gpt-4o-2024-08-06 1
# eval_ckpts_shuffle_option gpt4o-image gpt-4o-2024-08-06 1
# eval_ckpts_free_form gpt4o-image gpt-4o-2024-08-06 1
# eval_ckpts deepseek-math-7b-rl deepseek-math-7b-rl 0
#eval_ckpts math-shepherd-mistral-7b-rl math-shepherd-mistral-7b-rl 0
#eval_ckpts Qwen2.5-Math-7B-Instruct Qwen/Qwen2.5-Math-7B-Instruct/main 0
#eval_ckpts OpenMath2-Llama3.1-8B nvidia/OpenMath2-Llama3.1-8B/main 0
#eval_ckpts dart-math-dsmath-7b-prop2diff hkust-nlp/dart-math-dsmath-7b-prop2diff/main 0
#eval_ckpts Qwen2-VL-7B-Instruct Qwen2-VL-7B-Instruct 1
#eval_ckpts Qwen2-VL-72B-Instruct Qwen2-VL-72B-Instruct/main 1

#eval_ckpts Meta-Llama-3.1-70B-Instruct meta-llama/Meta-Llama-3.1-70B-Instruct 0
#eval_ckpts OpenMath2-Llama3.1-70B nvidia/OpenMath2-Llama3.1-70B/main 0

#eval_ckpts Qwen2.5-Math-72B-Instruct Qwen/Qwen2.5-Math-72B-Instruct/main 0
#eval_ckpts dart-math-llama3-70b-prop2diff hkust-nlp/dart-math-llama3-70b-prop2diff/main 0
#eval_ckpts Qwen2-72B-Instruct-Step-DPO xinlai/Qwen2-72B-Instruct-Step-DPO/main 0
#eval_ckpts internvl2-8B OpenGVLab/InternVL2-8B 1
#eval_ckpts Internvl2-Llama3-76B InternVL2-Llama3-76B/main 1
# eval_ckpts Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-7B-Instruct/main 1
#eval_ckpts Qwen2-VL-72B-Instruct Qwen/Qwen2-VL-72B-Instruct/main 1
# eval_ckpts_shuffle_option Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-7B-Instruct/main 1
# eval_ckpts_free_form Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-7B-Instruct/main 1
# eval_ckpts_shuffle_option Qwen2-VL-72B-Instruct Qwen2-VL-72B-Instruct/main 1
# eval_ckpts_free_form Qwen2-VL-72B-Instruct Qwen2-VL-72B-Instruct/main 1

