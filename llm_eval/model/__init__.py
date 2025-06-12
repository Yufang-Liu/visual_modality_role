

def generate_output(model_name, model_path, all_data, output_file):
    if model_name in ['deepseek-math-7b-rl']:
        from model.deepseek import generate_output as deepseek_generate_output
        return deepseek_generate_output(model_path, all_data, output_file)
    elif model_name in ['Qwen2.5-Math-7B-Instruct', 'Qwen2.5-Math-72B-Instruct',
                        'Qwen2-72B-Instruct-Step-DPO']:
        from model.qwen import generate_output as qwen_generate_output
        return qwen_generate_output(model_path, all_data, output_file)
    elif model_name in ['math-shepherd-mistral-7b-rl', ]:
        from model.math_shepherd import generate_output as math_shepherd_generate_output
        return math_shepherd_generate_output(model_path, all_data, output_file)
    elif model_name in ['OpenMath2-Llama3.1-8B', 'Meta-Llama-3.1-70B-Instruct',
                        'OpenMath2-Llama3.1-70B', 'dart-math-dsmath-7b-prop2diff',
                        'dart-math-llama3-70b-prop2diff']:
        from model.llama import generate_output as llama_generate_output
        return llama_generate_output(model_path, all_data, output_file)
    elif model_name in ['internvl-8B', 'Internvl2-Llama3-76B']:
        from model.internvl import generate_output as internvl_generate_output
        return internvl_generate_output(model_path, all_data, output_file)
    elif model_name in ['Qwen2-VL-7B-Instruct', 'Qwen2-VL-72B-Instruct']:
        from model.qwen2vl import generate_output as qwen2vl_generate_output
        return qwen2vl_generate_output(model_path, all_data, output_file)
    elif 'llava' in model_name and 'dino' not in model_name and 'siglip' not in model_name:
        from model.llava import generate_output as llava_generate_output
        return llava_generate_output(model_path, all_data, output_file)
    elif 'llava' in model_name and ('dino' in model_name or 'siglip' in model_name):
        from model.llava_enhance import generate_output as llava_enhance_generate_output
        return llava_enhance_generate_output(model_path, all_data, output_file)
    elif model_name in ['gpt4o', 'gpt4o-image']:
        from model.gpt import generate_output as gpt_generate_output
        return gpt_generate_output(model_path, all_data, output_file)
    else:
        raise ValueError(f"Unknown model {model_name}")
