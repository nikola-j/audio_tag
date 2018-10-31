import argparse
import json

import numpy as np
from keras.models import load_model

from audioset.vggish_inference import run_inference_vgg


def run_inference_model(input_wav, checkpoint):
    processed_wav = run_inference_vgg(input_wav)

    full_x = []
    for sec_x in processed_wav:  # Create new data point for each second of wav
        full_x.append(sec_x)

    full_x = np.array(full_x)

    model = load_model(checkpoint)

    return model.predict(full_x)


def labels_to_classes(labels_list):
    with open('audioset/output.json', 'r') as labelfile:
        labels = json.load(labelfile)

    label_to_class = {}

    for class_name, label in labels.items():
        label_to_class[label] = class_name

    return [label_to_class[i] for i in labels_list]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, default='trained_model.ckpt', help='What model to train')
    arg_parser.add_argument('--input_file', type=str, help='On what file to predict')
    args = arg_parser.parse_args()

    inference_result = run_inference_model(args.input_file, args.model)
    print(labels_to_classes(np.argmax(inference_result, axis=1)))
