import torch
import json
from tqdm import tqdm
import multiprocessing
import openai
import os
import backoff
import logging
import glob
from functools import partial
from tqdm import tqdm
import time
import base64
from PIL import Image
import io

openai.api_key = ""
openai.api_base = ""


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def encode_image(image_path):
 with open(image_path, "rb") as image_file:
   return base64.b64encode(image_file.read()).decode('utf-8')

# def encode_image(image_path, quality=10):
#     with Image.open(image_path) as img:
#         buffered = io.BytesIO()
#         img = img.convert('RGB')
#         img.save(buffered, format="JPEG", quality=quality)
#         img_byte_data = buffered.getvalue()
#         return base64.b64encode(img_byte_data).decode('utf-8')


def request_openai(data, model='gpt-4o-eva'):
    try:
        if 'image' not in data:
            output = completions_with_backoff(
                model=model,
                messages=[
                    {"role": "system", "content": "Be a helpful assistant."},
                    {"role": "user", "content": data['text']}
                ],
            )["choices"][0]["message"]["content"]
        else:
            # Getting the base64 string
            base64_image = encode_image(data['image'])
            output = completions_with_backoff(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": data['text']
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "detail": "low",
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            }
                        ]
                    },
                ]
            )["choices"][0]["message"]["content"]
        return output
    except Exception as e:
        print("Error : {}".format(e))
        time.sleep(60)
        request_openai(data, model=model)


def generate(data, model):
    output = request_openai(data, model=model)

    return {
        **data,
        "model_answer": output
    }


def generate_output(model_path, all_data, output_file, num_process=10):
    model_path = model_path.split('/')[-1]
    with open(output_file, 'w') as writer:
        with multiprocessing.Pool(num_process) as pool:
            for case in tqdm(pool.imap(
                    partial(generate, model=model_path), all_data), total=len(all_data)):
                writer.write(json.dumps(case) + '\n')
