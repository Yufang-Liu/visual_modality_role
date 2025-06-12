from .dataset_config import *


DataConfig = {
    "Pretrain": [LLaVA_Pretrain],
    "GeoPretrain": [LLaVA_Pretrain, Geo170K_align],
    "MAVISPretrain": [LLaVA_Pretrain, MAVIS_align],
    "MathPretrain": [LLaVA_Pretrain, Geo170K_align, MultiMath_caption_ZH, MultiMath_caption_EN],
    "FineTune": [LLaVA_Instruct],
    "FineTuneExistence": [LLaVA_Instruct, math_object_existence],
    "FineTuneImageShuffle": [LLaVA_Instruct_image_shuffle],
    "FineTuneImageDeleted": [LLaVA_Instruct_image_deleted],
    "GeoFineTune": [Geo170K_qa],
    "GeoFineTuneExistence": [Geo170K_qa, math_object_existence],
    "GeoFineTuneImageShuffle": [Geo170k_qa_image_shuffle],
    "GeoFineTuneImageDeleted": [Geo170k_qa_image_deleted],
    "Math360FineTune": [MathV360K],
    "Math360FineTuneImageShuffle": [MathV360K_image_shuffle],
    "Math360FineTuneImageDeleted": [MathV360K_image_delete],
    "MAVISFineTune": [MAVIS_qa],
    "MAVISFineTuneImageShuffle": [MAVIS_qa_image_shuffle],
    "MAVISFineTuneImageDeleted": [MAVIS_qa_image_delete],
    "MultiFT": [LLaVA_Instruct, gsm8k_train, math_train, cmath_dev],
    "MathFineTune": [Geo170K_qa, MathV360K, MultiMath_solution_EN, MultiMath_solution_ZH],
    "MathFineTuneImageShuffle": [Geo170k_qa_image_shuffle, MathV360K_image_shuffle, MultiMath_solution_EN_image_shuffle, MultiMath_solution_ZH_image_shuffle],
    "MathFineTuneImageDeleted": [Geo170k_qa_image_deleted, MathV360K_image_delete, MultiMath_solution_EN_image_delete, MultiMath_solution_ZH_image_delete],
    "VisionTextMathFineTune": [Geo170K_qa, MathV360K, MultiMath_solution_ZH, MultiMath_solution_EN, gsm8k_train, math_train, cmath_dev]
}
