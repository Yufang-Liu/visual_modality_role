from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from tqdm import tqdm
import json

def generate_output(model_path, all_data, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,  bf16=True).eval()
    
    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.generation_config.max_new_tokens = 1024

    fwrite = open(output_file, "w")
    for data in tqdm(all_data):
        query = tokenizer.from_list_format([
            {'image': data['image']},
            {'text': data['text']},
        ])
        response = model.chat(tokenizer, query=query, history=None)
        data['model_answer'] = response
        fwrite.write(json.dumps(data) + "\n")
    fwrite.close()