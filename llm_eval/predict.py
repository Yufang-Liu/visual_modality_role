import argparse
from utils import *
from model import *


def calc_accuracy(ans_file):

    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(ans_file, 'r')]

    for answer in answers:
        text = answer['model_answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['model_answer'] = 'no'
        else:
            answer['model_answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['model_answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))

def load_dataset(dataset_name, dataset_path, image_root, image_file=None, with_image=False):
    if dataset_name == 'geoqa':
        return load_text(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name == 'mathverse':
        return load_query(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name == 'mathvista':
        return load_mathvista(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name == 'mathvision':
        return load_query(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name == 'wemath':
        return load_query(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name == 'HC-M3D':
        return load_hcm3d(dataset_path, image_root, image_file, with_image=with_image)
    elif dataset_name.startswith('math_pope'):
        return load_pope(dataset_path, image_root, image_file, with_image=with_image)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='geoqa',
                        help='The name of the dataset')
    parser.add_argument('--dataset_path', type=str, default='./data/geoqa',
                        help='The path to the dataset')
    parser.add_argument('--split', type=str, default='testmini',
                        help='The split of the dataset')
    parser.add_argument('--image_root', type=str, default='./data/imgs',
                        help='The path to the image folder')
    parser.add_argument('--image_file', type=str, default=None,
                        help='The path to the image file')
    parser.add_argument('--model_name', type=str, default='deepseek',
                        help='The name of the model')
    parser.add_argument('--model_path', type=str, default='./checkpoints/xx.pt',
                        help='The path to the model')
    parser.add_argument('--output_dir', type=str, default='../output/llm_results/',
                        help='The path to the output file')
    parser.add_argument('--with_image', type=int, default=0,
                        help='Whether to use image')
    args = parser.parse_args()

    all_data = load_dataset(args.dataset_name, args.dataset_path,
                            args.image_root, args.image_file, args.with_image)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = args.output_dir + '{}_{}_{}.json'.format(args.model_name, args.dataset_name, args.split)
        
    generate_output(args.model_name, args.model_path, all_data, output_file)

    if args.dataset_name.startswith('math_pope'):
        calc_accuracy(output_file)



if __name__ == '__main__':
    main()