import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def generate_output(model_path, all_data, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    fwrite = open(output_file, "w")
    for data in tqdm(all_data):
        messages = [
            {"role": "user", "content": data['text']}
        ]
        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1024)

        result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        data['model_answer'] = result
        print(data)

        fwrite.write(json.dumps(data) + "\n")
    fwrite.close()
