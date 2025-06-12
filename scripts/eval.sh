
home_dir=''
ckpt_dir=''


 pip install shortuuid
 pip install accelerate -U
 export PATH=$PATH:~/.local/bin


eval_ckpt(){
  bash scripts/eval_multi.sh \
${ckpt_dir}/"${1}" \
${home_dir}/gllava/playground/data/test_questions.jsonl \
${home_dir}/gllava/results_try/"${1}"-test \
${home_dir}/gllava/playground/data/images/ \
8 0 "$2"

python3 scripts/geo_acc_calculate.py  \
--ground_truth_file \
${home_dir}/gllava/playground/data/test_answers.jsonl \
--predictions_file ${home_dir}/gllava/results_try/"${1}"-test_merged.jsonl

bash scripts/eval_multi.sh \
${home_dir}/gllava/checkpoints/gllava_mavis_qa_tuned_"$1" \
${home_dir}/data/rl_math_data/PGPS9K_all/pgp_test_questions.jsonl \
${home_dir}/gllava/results_try/gllava_mavis_qa_tuned_pgp_"$1"-test \
${home_dir}/rl_math/data/PGPS9K_all/ \
8 0 "${2}"

python scripts/geo_acc_calculate.py  \
--ground_truth_file \
${home_dir}/data/rl_math_data/PGPS9K_all/pgp_test_answers.jsonl \
--predictions_file ${home_dir}/gllava/results_try/gllava_mavis_qa_tuned_pgp_"$1"-test_merged.jsonl
}

eval_gllava_trained(){
   bash scripts/eval_multi.sh \
 ${home_dir}/ckpts/llava_checkpoints/"${1}" \
 ${home_dir}/gllava/playground/data/test_questions.jsonl \
 ${home_dir}/ckpts/llava_checkpoints/results/"${1}"-test \
 ${home_dir}/gllava/playground/data/images/ \
 1 0 "${2}"

 python3 scripts/geo_acc_calculate.py  \
 --ground_truth_file \
 ${home_dir}/gllava/playground/data/test_answers.jsonl \
 --predictions_file ${home_dir}/ckpts/llava_checkpoints/results/"${1}"-test_merged.jsonl

# bash scripts/v1_5/eval_multi.sh \
# ${home_dir}/ckpts/llava_checkpoints/"${1}" \
# ${home_dir}/data/rl_math_data/PGPS9K_all/pgp_test_questions.jsonl \
# ${home_dir}/ckpts/llava_checkpoints/results/"$1"-test \
# ${home_dir}/rl_math/data/PGPS9K_all/ \
# 8 0 "${2}"

#python scripts/v1_5/geo_acc_calculate.py  \
#--ground_truth_file \
#${home_dir}/data/rl_math_data/PGPS9K_all/pgp_test_answers.jsonl \
#--predictions_file ${home_dir}/ckpts/llava_checkpoints/results/"$1"-test_merged.jsonl
}

eval_gllava_trained gllava-clip-base-pretrain-sft-mathsft ds_math
#eval_gllava_trained llava-deepseek-math-7b_math-pretrain-sft-mathsft ds_math
#eval_gllava_trained llava-deepseek-math-7b_math-pretrain-sft-mathsft_clip_finetune_3e-7_10_1_25_image_encoder_2w ds_math
#eval_gllava_trained llava-deepseek-math-7b-pretrain-sft-mathsft ds_math
#eval_gllava_trained mathllava-pretrain-sft-mathsft ds_math
#eval_gllava_trained gllava-dino-clip-pretrain-sft-mathsft ds_math
#eval_gllava_trained mathllava-dino-clip-pretrain-sft-mathsft ds_math
#echo 'gllava-siglip-clip-pretrain-sft-mathsft'
#eval_gllava_trained gllava-siglip-clip-pretrain-sft-mathsft ds_math
#echo 'gllava-siglip-clip-dino-pretrain-sft-mathsft'
#eval_gllava_trained gllava-siglip-clip-dino-pretrain-sft-mathsft ds_math

#eval_gllava_trained gllava-dino-clip-concat-pretrain-sft-mathsft ds_math
#eval_gllava_trained gllava-siglip-clip-cat-pretrain-sft-mathsft ds_math
#echo 'gllava-dino-clip-siglip-concat-pretrain-sft-mathsft'
#eval_gllava_trained gllava-dino-clip-siglip-concat-pretrain-sft-mathsft ds_math