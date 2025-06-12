import torch
import json
from tqdm import tqdm
import transformers


def generate_output(model_path, all_data, output_file):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    fwrite = open(output_file, "w")
    for data in tqdm(all_data):
        messages = [
            {
                "role": "user",
                "content": data['text']},
        ]

        outputs = pipeline(messages, max_new_tokens=1024)
        data['model_answer'] = outputs[0]['generated_text'][-1]['content']
        print(data)

        fwrite.write(json.dumps(data) + "\n")
    fwrite.close()
