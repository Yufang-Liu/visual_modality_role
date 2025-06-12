import json
import os
import shortuuid
import re


def load_text(dataset_path, image_root, image_file, with_image):
    print("Loading data from input_file {}".format(dataset_path))
    all_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data['image'] = os.path.join(image_root, data['image'])
            if not with_image:
                del data['image']
                pattern = r'<image\d*>'
                # 替换为空字符串
                data['text'] = re.sub(pattern+r'\n', '', data['text'])
                data['text'] = re.sub(r'\n'+pattern, '', data['text'])
            all_data.append(data)
    return all_data


def load_query(dataset_path, image_root, image_file, with_image):
    print("Loading data from input_file {}".format(dataset_path))
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    all_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        item['text'] = item['query_cot']
        item['image'] = os.path.join(image_root, item['image'])
        if not with_image:
            del item['image']
            pattern = r'<image\d*>'
            # 替换为空字符串
            item['text'] = re.sub(pattern+r'\n', '', item['text'])
            item['text'] = re.sub(r'\n'+pattern, '', item['text'])
        all_data.append(item)
    return all_data


def load_mathvista(dataset_path, image_root, image_file, with_image):
    print("Loading data from input_file {}".format(image_file))
    with open(image_file, 'r') as f:
        dataset = json.load(f)

    with open(dataset_path, 'r') as f:
        query_data = json.load(f)

    all_data = []
    for i in dataset.keys():
        dataset[i]['pid'] = i
        dataset[i]['image'] = os.path.join(image_root, dataset[i]['image'])
        dataset[i]['text'] = query_data[i]
        if not with_image:
            del dataset[i]['image']
            pattern = r'<image\d*>'
            # 替换为空字符串
            dataset[i]['text'] = re.sub(pattern+r'\n', '', dataset[i]['text'])
            dataset[i]['text'] = re.sub(r'\n'+pattern, '', dataset[i]['text'])
        all_data.append(dataset[i])
    return all_data


def load_hcm3d(dataset_path, image_root, image_file, with_image):
    print("Loading data from input_file {}".format(dataset_path))
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    all_data = []
    for i in range(len(dataset)):
        item = dataset[i]
        item['text'] = item['question']
        if not with_image:
            del item['image']
            pattern = r'<image\d*>'
            # 替换为空字符串
            item['text'] = re.sub(pattern + r'\n', '', item['text'])
            item['text'] = re.sub(r'\n' + pattern, '', item['text'])
        item['image'] = os.path.join(image_root, item['image'])
        all_data.append(item)
    return all_data


def load_pope(dataset_path, image_root, image_file, with_image):
    print("Loading data from input_file {}".format(dataset_path))
    all_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data['image'] = os.path.join(image_root, 'test',
                                         data['image'], 'img_diagram.png')
            data['image'] = os.path.abspath(data['image'])
            all_data.append(data)
    return all_data

