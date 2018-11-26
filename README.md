# Audiotag

Automatic .wav file audio tagging using VGGish.
A simple POC.

## Installation

Install requirements using:
```
pip install -r requiremnts.txt
```

Download VGGish model:
```
# Download data files into the audioset directory
cd audioset
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
```

## Dataset

Download dataset from this competition: https://www.kaggle.com/c/freesound-audio-tagging

## Pre-compute vggish audio features

Use batch_inference.py to compute audio features from audio signals
eg:
```bash
python batch_inference.py --wav_train [train files] --wav_csv [train csv file]
```

## Train

Use train.py to train.py a model, choose a batch size and model to use

## Inference

Use jupyter to run inference, open 'Sound tag.ipynb'

## External code

Audioset used from here: https://github.com/tensorflow/models

Batched vggish inference used from here: https://github.com/knstmrd/vggish-batch
