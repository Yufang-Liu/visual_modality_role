import torch
import json
from tqdm import tqdm
import sys

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


# adapted from model_vqa.py, only change the args to avoid loading models multiple times and return output
def eval_model(args, tokenizer, model, image_processor):
    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(args.image_file).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).bfloat16().cuda(),
            # no_repeat_ngram_size=3,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


def generate_output(model_path, all_data, output_file):
    print("Loading model from model_path {}".format(model_path))
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    model.to("cuda")

    fwrite = open(output_file, "w")
    for data in tqdm(all_data):
        prompt = data["text"]

        model_args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": "ds_math",
            "image_file": data['image'],
            "max_new_tokens": 512
        })()

        data["model_answer"] = eval_model(model_args, tokenizer, model, image_processor)
        fwrite.write(json.dumps(data) + "\n")
    fwrite.close()