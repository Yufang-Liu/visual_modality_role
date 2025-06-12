from tqdm import tqdm
import json
import torch
from vllm import LLM, SamplingParams


def generate_output(model_path, all_data, output_file):
    llm = LLM(model=model_path, gpu_memory_utilization=0.95)
    print("Loading model from {}".format(model_path))
    tokenizer = llm.get_tokenizer()

    try:
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': item['text']}],
            tokenize=False,
        ) for item in all_data]
    except Exception:
        prompts = [item['text'] for item in all_data]
    sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params)

    fwrite = open(output_file, "w")
    for output, data in zip(outputs, all_data):
        generated_text = output.outputs[0].text
        data['model_answer'] = generated_text
        print(data)
        fwrite.write(json.dumps(data) + "\n")
    fwrite.close()
