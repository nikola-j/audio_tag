# Audiotag

Automatic .wav file audio tagging using VGGish.


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

## Train

Use train.py to train.py a model, choose a batch size and model to use

## Inference

Use jupyter to run inference, open 'Sound tag.ipynb'

## External code

Audioset used from here: https://github.com/tensorflow/models

Batched vggish inference used from here: https://github.com/knstmrd/vggish-batch
