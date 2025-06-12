
pip install shortuuid
pip install accelerate -U
export PATH=$PATH:~/.local/bin

#model_list=(gllava-pretrain-sft gllava-dino-clip-concat-pretrain-sft gllava-siglip-clip-cat-pretrain-sft
#gllava-dino-clip-siglip-concat-pretrain-sft gllava-dino-clip-pretrain-sft gllava-siglip-clip-pretrain-sft
#gllava-siglip-clip-dino-pretrain-sft)
#model_list=(gllava-siglip-clip-pretrain-sft gllava-siglip-clip-dino-pretrain-sft)
model_list=(gllava-clip-base-pretrain-sft)

for(( i=0;i<${#model_list[@]};i++)) do
   echo '---------MME eval '${model_list[i]}'--------------'
   sh scripts/eval/mme.sh ${model_list[i]};

  echo '---------pope eval '${model_list[i]}'--------------'
  sh scripts/eval/pope.sh ${model_list[i]};

   echo '---------mmbench eval '${model_list[i]}'--------------'
   sh scripts/eval/mmbench.sh ${model_list[i]};

   echo '---------mmvet eval '${model_list[i]}'--------------'
   sh scripts/eval/mmvet.sh ${model_list[i]};

   echo '---------vqav2 eval '${model_list[i]}'--------------'
   sh scripts/eval/vqav2.sh ${model_list[i]};
done;